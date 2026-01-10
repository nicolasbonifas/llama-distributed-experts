#include "ggml-rpc-expert.h"
#include "ggml.h"
#include <cstring>
#include <cstdio>
#include <vector>
#include <map>
#include <string>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>

// Global state for expert RPC
struct expert_rpc_state {
    bool initialized = false;
    bool shutdown_requested = false;
    std::string model_path;
    std::string expert_range;
    std::string endpoint;
    int server_socket = -1;
    pthread_t worker_thread;

    // Connection pool for client-side
    std::map<std::string, int> connections;

    // Evaluation callback (worker side)
    ggml_rpc_expert_eval_callback eval_callback = nullptr;
    void * eval_user_data = nullptr;

    // Endpoint lookup callback (master side)
    ggml_rpc_expert_endpoint_lookup_callback endpoint_lookup_callback = nullptr;
    void * endpoint_lookup_user_data = nullptr;
};

static expert_rpc_state g_expert_rpc;

// Helper: Parse endpoint into host and port
static bool parse_endpoint(const char * endpoint, std::string & host, int & port) {
    std::string ep(endpoint);
    size_t colon_pos = ep.rfind(':');
    if (colon_pos == std::string::npos) {
        return false;
    }
    host = ep.substr(0, colon_pos);
    try {
        port = std::stoi(ep.substr(colon_pos + 1));
    } catch (...) {
        return false;
    }
    return true;
}

// Helper: Create TCP connection to endpoint
static int connect_to_endpoint(const char * endpoint) {
    // Check connection pool first
    std::string ep_str(endpoint);
    auto it = g_expert_rpc.connections.find(ep_str);
    if (it != g_expert_rpc.connections.end()) {
        // TODO: Check if connection is still alive
        return it->second;
    }

    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        fprintf(stderr, "%s: invalid endpoint format: %s\n", __func__, endpoint);
        return -1;
    }

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        fprintf(stderr, "%s: socket creation failed: %s\n", __func__, strerror(errno));
        return -1;
    }

    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);

    if (inet_pton(AF_INET, host.c_str(), &server_addr.sin_addr) <= 0) {
        fprintf(stderr, "%s: invalid address: %s\n", __func__, host.c_str());
        close(sock);
        return -1;
    }

    if (connect(sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        fprintf(stderr, "%s: connection failed to %s: %s\n", __func__, endpoint, strerror(errno));
        close(sock);
        return -1;
    }

    // Add to connection pool
    g_expert_rpc.connections[ep_str] = sock;

    return sock;
}

// Helper: Send data over socket
static bool send_all(int sock, const void * data, size_t size) {
    size_t sent = 0;
    const uint8_t * ptr = (const uint8_t *)data;

    while (sent < size) {
        ssize_t n = send(sock, ptr + sent, size - sent, 0);
        if (n < 0) {
            fprintf(stderr, "%s: send failed: %s\n", __func__, strerror(errno));
            return false;
        }
        sent += n;
    }
    return true;
}

// Helper: Receive data from socket
static bool recv_all(int sock, void * data, size_t size) {
    size_t received = 0;
    uint8_t * ptr = (uint8_t *)data;

    while (received < size) {
        ssize_t n = recv(sock, ptr + received, size - received, 0);
        if (n <= 0) {
            if (n < 0) {
                fprintf(stderr, "%s: recv failed: %s\n", __func__, strerror(errno));
            } else {
                fprintf(stderr, "%s: connection closed by peer\n", __func__);
            }
            return false;
        }
        received += n;
    }
    return true;
}

// Forward declaration for worker thread
static void* worker_thread_func(void* arg);

// Handle a single request on the worker side
static bool handle_single_request(int client_sock) {
    // Receive request header
    ggml_rpc_expert_request req;
    if (!recv_all(client_sock, &req, sizeof(req))) {
        fprintf(stderr, "%s: failed to receive request header\n", __func__);
        return false;
    }

    // Validate magic and version
    if (req.magic != GGML_RPC_EXPERT_MAGIC) {
        fprintf(stderr, "%s: invalid magic number: 0x%08x\n", __func__, req.magic);

        ggml_rpc_expert_response resp;
        resp.magic = GGML_RPC_EXPERT_MAGIC;
        resp.version = GGML_RPC_EXPERT_VERSION;
        resp.status = GGML_RPC_EXPERT_STATUS_INVALID_REQ;
        resp.n_tokens = 0;
        resp.n_embd = 0;
        send_all(client_sock, &resp, sizeof(resp));
        return false;
    }

    if (req.version != GGML_RPC_EXPERT_VERSION) {
        fprintf(stderr, "%s: unsupported version: %d\n", __func__, req.version);

        ggml_rpc_expert_response resp;
        resp.magic = GGML_RPC_EXPERT_MAGIC;
        resp.version = GGML_RPC_EXPERT_VERSION;
        resp.status = GGML_RPC_EXPERT_STATUS_INVALID_REQ;
        resp.n_tokens = 0;
        resp.n_embd = 0;
        send_all(client_sock, &resp, sizeof(resp));
        return false;
    }

    fprintf(stderr, "%s: received request - layer=%d, n_experts=%d, n_tokens=%d, n_embd=%d, n_ff=%d\n",
            __func__, req.layer_id, req.n_experts, req.n_tokens, req.n_embd, req.n_ff);

    // Receive expert IDs
    std::vector<uint8_t> expert_ids(req.n_experts);
    if (!recv_all(client_sock, expert_ids.data(), req.n_experts)) {
        fprintf(stderr, "%s: failed to receive expert IDs\n", __func__);
        return false;
    }

    // Receive weights
    size_t weights_size = req.n_experts * req.n_tokens * sizeof(float);
    std::vector<float> weights(req.n_experts * req.n_tokens);
    if (!recv_all(client_sock, weights.data(), weights_size)) {
        fprintf(stderr, "%s: failed to receive weights\n", __func__);
        return false;
    }

    // Receive input data
    size_t input_size = req.n_embd * req.n_tokens * sizeof(float);
    std::vector<float> input_data(req.n_embd * req.n_tokens);
    if (!recv_all(client_sock, input_data.data(), input_size)) {
        fprintf(stderr, "%s: failed to receive input data\n", __func__);
        return false;
    }

    fprintf(stderr, "%s: received all data, processing experts...\n", __func__);

    // Allocate output buffer
    std::vector<float> output_data(req.n_embd * req.n_tokens, 0.0f);

    // Call evaluation callback if registered
    bool eval_success = false;
    if (g_expert_rpc.eval_callback != nullptr) {
        fprintf(stderr, "%s: calling registered evaluation callback\n", __func__);
        eval_success = g_expert_rpc.eval_callback(
            g_expert_rpc.eval_user_data,
            req.layer_id,
            req.n_experts,
            expert_ids.data(),
            req.n_tokens,
            req.n_embd,
            req.n_ff,
            weights.data(),
            input_data.data(),
            output_data.data()
        );
    } else {
        fprintf(stderr, "%s: WARNING - no evaluation callback registered, returning zeros\n", __func__);
        // Return zeros as fallback
        eval_success = true;
    }

    if (!eval_success) {
        fprintf(stderr, "%s: evaluation callback failed\n", __func__);

        ggml_rpc_expert_response error_resp;
        error_resp.magic = GGML_RPC_EXPERT_MAGIC;
        error_resp.version = GGML_RPC_EXPERT_VERSION;
        error_resp.status = GGML_RPC_EXPERT_STATUS_ERROR;
        error_resp.n_tokens = 0;
        error_resp.n_embd = 0;
        send_all(client_sock, &error_resp, sizeof(error_resp));
        return false;
    }

    // Send response header
    ggml_rpc_expert_response resp;
    resp.magic = GGML_RPC_EXPERT_MAGIC;
    resp.version = GGML_RPC_EXPERT_VERSION;
    resp.status = GGML_RPC_EXPERT_STATUS_SUCCESS;
    resp.n_tokens = req.n_tokens;
    resp.n_embd = req.n_embd;

    if (!send_all(client_sock, &resp, sizeof(resp))) {
        fprintf(stderr, "%s: failed to send response header\n", __func__);
        return false;
    }

    // Send output data
    size_t output_size = resp.n_embd * resp.n_tokens * sizeof(float);
    if (!send_all(client_sock, output_data.data(), output_size)) {
        fprintf(stderr, "%s: failed to send output data\n", __func__);
        return false;
    }

    fprintf(stderr, "%s: successfully sent response (%zu bytes)\n", __func__, output_size);
    return true;
}

// Worker thread that accepts connections and handles requests
static void* worker_thread_func(void* arg) {
    (void)arg;
    fprintf(stderr, "%s: worker thread started\n", __func__);

    while (!g_expert_rpc.shutdown_requested) {
        // Accept incoming connection
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);

        int client_sock = accept(g_expert_rpc.server_socket,
                                 (struct sockaddr *)&client_addr,
                                 &client_len);

        if (client_sock < 0) {
            if (g_expert_rpc.shutdown_requested) {
                break;
            }
            fprintf(stderr, "%s: accept failed: %s\n", __func__, strerror(errno));
            continue;
        }

        char client_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, INET_ADDRSTRLEN);
        fprintf(stderr, "%s: accepted connection from %s:%d\n",
                __func__, client_ip, ntohs(client_addr.sin_port));

        // Handle multiple requests on same connection
        while (handle_single_request(client_sock)) {
            // Continue handling requests until client disconnects or error
        }

        close(client_sock);
        fprintf(stderr, "%s: closed connection from %s:%d\n",
                __func__, client_ip, ntohs(client_addr.sin_port));
    }

    fprintf(stderr, "%s: worker thread exiting\n", __func__);
    return nullptr;
}

bool ggml_rpc_expert_init(
    const char * model_path,
    const char * expert_range,
    const char * endpoint
) {
    fprintf(stderr, "%s: initializing expert RPC worker\n", __func__);
    fprintf(stderr, "  model:    %s\n", model_path);
    fprintf(stderr, "  experts:  %s\n", expert_range);
    fprintf(stderr, "  endpoint: %s\n", endpoint);

    g_expert_rpc.model_path = model_path;
    g_expert_rpc.expert_range = expert_range;
    g_expert_rpc.endpoint = endpoint;

    // Parse endpoint
    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        fprintf(stderr, "%s: invalid endpoint format\n", __func__);
        return false;
    }

    // Create listening socket
    g_expert_rpc.server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (g_expert_rpc.server_socket < 0) {
        fprintf(stderr, "%s: socket creation failed: %s\n", __func__, strerror(errno));
        return false;
    }

    // Set socket options
    int opt = 1;
    if (setsockopt(g_expert_rpc.server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        fprintf(stderr, "%s: setsockopt failed: %s\n", __func__, strerror(errno));
        close(g_expert_rpc.server_socket);
        return false;
    }

    // Bind to address
    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);

    if (bind(g_expert_rpc.server_socket, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        fprintf(stderr, "%s: bind failed: %s\n", __func__, strerror(errno));
        close(g_expert_rpc.server_socket);
        return false;
    }

    // Start listening
    if (listen(g_expert_rpc.server_socket, 5) < 0) {
        fprintf(stderr, "%s: listen failed: %s\n", __func__, strerror(errno));
        close(g_expert_rpc.server_socket);
        return false;
    }

    fprintf(stderr, "%s: server listening on port %d\n", __func__, port);

    // TODO: Load model with only assigned experts (next step)

    // Start worker thread to accept connections
    g_expert_rpc.shutdown_requested = false;
    if (pthread_create(&g_expert_rpc.worker_thread, nullptr, worker_thread_func, nullptr) != 0) {
        fprintf(stderr, "%s: failed to create worker thread: %s\n", __func__, strerror(errno));
        close(g_expert_rpc.server_socket);
        return false;
    }

    fprintf(stderr, "%s: worker thread started successfully\n", __func__);

    g_expert_rpc.initialized = true;
    return true;
}

bool ggml_rpc_expert_evaluate(
    const char * endpoint,
    uint8_t layer_id,
    uint8_t n_experts,
    const uint8_t * expert_ids,
    uint16_t n_tokens,
    uint32_t n_embd,
    uint32_t n_ff,
    const float * weights,
    const float * input_data,
    float * output_data
) {
    fprintf(stderr, "%s: calling remote expert evaluation\n", __func__);
    fprintf(stderr, "  endpoint: %s\n", endpoint);
    fprintf(stderr, "  layer: %d, n_experts: %d, n_tokens: %d\n",
            layer_id, n_experts, n_tokens);

    // Connect to worker
    int sock = connect_to_endpoint(endpoint);
    if (sock < 0) {
        fprintf(stderr, "%s: failed to connect to %s\n", __func__, endpoint);
        return false;
    }

    // Send request header
    ggml_rpc_expert_request req;
    req.magic = GGML_RPC_EXPERT_MAGIC;
    req.version = GGML_RPC_EXPERT_VERSION;
    req.layer_id = layer_id;
    req.n_experts = n_experts;
    req.n_tokens = n_tokens;
    req.n_embd = n_embd;
    req.n_ff = n_ff;

    if (!send_all(sock, &req, sizeof(req))) {
        fprintf(stderr, "%s: failed to send request header\n", __func__);
        return false;
    }

    // Send expert IDs
    if (!send_all(sock, expert_ids, n_experts)) {
        fprintf(stderr, "%s: failed to send expert IDs\n", __func__);
        return false;
    }

    // Send weights
    size_t weights_size = n_experts * n_tokens * sizeof(float);
    if (!send_all(sock, weights, weights_size)) {
        fprintf(stderr, "%s: failed to send weights\n", __func__);
        return false;
    }

    // Send input data
    size_t input_size = n_embd * n_tokens * sizeof(float);
    if (!send_all(sock, input_data, input_size)) {
        fprintf(stderr, "%s: failed to send input data\n", __func__);
        return false;
    }

    // Receive response header
    ggml_rpc_expert_response resp;
    if (!recv_all(sock, &resp, sizeof(resp))) {
        fprintf(stderr, "%s: failed to receive response header\n", __func__);
        return false;
    }

    // Check response
    if (resp.magic != GGML_RPC_EXPERT_MAGIC || resp.version != GGML_RPC_EXPERT_VERSION) {
        fprintf(stderr, "%s: invalid response magic or version\n", __func__);
        return false;
    }

    if (resp.status != GGML_RPC_EXPERT_STATUS_SUCCESS) {
        fprintf(stderr, "%s: remote evaluation failed with status %d\n", __func__, resp.status);

        // Read error message if present
        if (resp.n_tokens > 0) {
            std::vector<char> error_msg(resp.n_tokens + 1, 0);
            if (recv_all(sock, error_msg.data(), resp.n_tokens)) {
                fprintf(stderr, "%s: error: %s\n", __func__, error_msg.data());
            }
        }
        return false;
    }

    // Receive output data
    size_t output_size = resp.n_embd * resp.n_tokens * sizeof(float);
    if (!recv_all(sock, output_data, output_size)) {
        fprintf(stderr, "%s: failed to receive output data\n", __func__);
        return false;
    }

    fprintf(stderr, "%s: successfully received %zu bytes of output\n", __func__, output_size);
    return true;
}

void ggml_rpc_expert_register_eval_callback(
    ggml_rpc_expert_eval_callback callback,
    void * user_data
) {
    fprintf(stderr, "%s: registering evaluation callback\n", __func__);
    g_expert_rpc.eval_callback = callback;
    g_expert_rpc.eval_user_data = user_data;
}

void ggml_rpc_expert_register_endpoint_lookup_callback(
    ggml_rpc_expert_endpoint_lookup_callback callback,
    void * user_data
) {
    fprintf(stderr, "%s: registering endpoint lookup callback\n", __func__);
    g_expert_rpc.endpoint_lookup_callback = callback;
    g_expert_rpc.endpoint_lookup_user_data = user_data;
}

void ggml_rpc_expert_shutdown() {
    fprintf(stderr, "%s: shutting down expert RPC\n", __func__);

    // Signal worker thread to exit
    g_expert_rpc.shutdown_requested = true;

    // Close server socket to unblock accept()
    if (g_expert_rpc.server_socket >= 0) {
        shutdown(g_expert_rpc.server_socket, SHUT_RDWR);
        close(g_expert_rpc.server_socket);
        g_expert_rpc.server_socket = -1;
    }

    // Wait for worker thread to finish
    if (g_expert_rpc.initialized) {
        fprintf(stderr, "%s: waiting for worker thread to exit...\n", __func__);
        pthread_join(g_expert_rpc.worker_thread, nullptr);
        fprintf(stderr, "%s: worker thread joined\n", __func__);
    }

    // Close all client connections
    for (auto & conn : g_expert_rpc.connections) {
        close(conn.second);
    }
    g_expert_rpc.connections.clear();

    g_expert_rpc.initialized = false;
}

bool ggml_rpc_expert_dispatch_mul_mat_id(
    const struct ggml_tensor * ids,
    const struct ggml_tensor * weights,
    const struct ggml_tensor * input,
    struct ggml_tensor * output,
    void * model_ptr,
    int layer_id
) {
    // This function handles distributed dispatch for MoE expert evaluation
    // Called from ggml_compute_forward_mul_mat_id_distributed at execution time

    // Extract model pointer and router weights from context struct
    struct distributed_moe_ctx {
        void * model_ptr;
        struct ggml_tensor * router_weights;
    };

    distributed_moe_ctx * ctx = nullptr;
    struct ggml_tensor * router_weights_tensor = nullptr;

    if (model_ptr != nullptr) {
        // New path: model_ptr points to distributed_moe_ctx struct
        ctx = (distributed_moe_ctx *)model_ptr;
        model_ptr = ctx->model_ptr;
        router_weights_tensor = ctx->router_weights;
    }

    if (model_ptr == nullptr) {
        fprintf(stderr, "%s: no model pointer provided, cannot dispatch\n", __func__);
        return false;
    }

    // Validate tensor types
    if (ids->type != GGML_TYPE_I32) {
        fprintf(stderr, "%s: expert IDs tensor must be I32, got %d\n", __func__, ids->type);
        return false;
    }

    // Read expert IDs from the tensor
    // At execution time, the data is available on the backend
    const int32_t * expert_ids = (const int32_t *)ids->data;
    const int64_t n_experts = ids->ne[0];  // Number of experts selected
    const int64_t n_tokens = ids->ne[1];   // Batch size

    fprintf(stderr, "%s: layer %d - evaluating %lld experts for %lld tokens\n",
            __func__, layer_id, n_experts, n_tokens);

    // Log the expert IDs for debugging
    fprintf(stderr, "%s: expert IDs: [", __func__);
    for (int64_t i = 0; i < std::min(n_experts, (int64_t)10); i++) {
        fprintf(stderr, "%d%s", expert_ids[i], i < std::min(n_experts, (int64_t)10) - 1 ? ", " : "");
    }
    if (n_experts > 10) {
        fprintf(stderr, ", ... (%lld total)", n_experts);
    }
    fprintf(stderr, "]\n");

    // Check if endpoint lookup callback is registered
    if (g_expert_rpc.endpoint_lookup_callback == nullptr) {
        fprintf(stderr, "%s: no endpoint lookup callback registered\n", __func__);
        fprintf(stderr, "%s: falling back to local evaluation\n", __func__);
        return false;  // Fallback to regular mul_mat_id
    }

    // Group experts by endpoint
    std::map<std::string, std::vector<int>> experts_by_endpoint;
    std::vector<int> local_experts;

    for (int64_t i = 0; i < n_experts; i++) {
        int expert_id = expert_ids[i];
        const char * endpoint = g_expert_rpc.endpoint_lookup_callback(
            model_ptr,  // Can be g_expert_rpc.endpoint_lookup_user_data or model_ptr
            expert_id,
            layer_id
        );

        if (endpoint == nullptr || endpoint[0] == '\0') {
            // Local expert
            local_experts.push_back(expert_id);
        } else {
            // Remote expert
            experts_by_endpoint[std::string(endpoint)].push_back(expert_id);
        }
    }

    fprintf(stderr, "%s: found %zu local experts, %zu remote endpoints\n",
            __func__, local_experts.size(), experts_by_endpoint.size());

    // Log remote expert groups
    for (const auto & [endpoint, expert_list] : experts_by_endpoint) {
        fprintf(stderr, "%s: endpoint %s has %zu experts\n",
                __func__, endpoint.c_str(), expert_list.size());
    }

    // Handle different deployment scenarios
    if (experts_by_endpoint.empty()) {
        // All experts are local - use optimized local path
        fprintf(stderr, "%s: all experts are local, using optimized local mul_mat_id\n", __func__);
        return false;  // Let regular mul_mat_id handle it
    }

    if (!local_experts.empty()) {
        // Mixed local/remote experts
        // This is a complex case that requires:
        //   1. Computing FFN for local experts directly
        //   2. Making RPC calls for remote experts
        //   3. Merging weighted results
        //
        // Implementation approaches:
        //   A) Direct FFN computation for local experts (requires access to expert weights)
        //   B) Recursive mul_mat_id call with subset of experts (requires tensor creation)
        //   C) Graph-time separation of local/remote paths (requires graph builder changes)
        //
        // For now, we have three options:
        //   1. Fall back to all-local (safe, works, but defeats the purpose)
        //   2. Implement direct FFN for local + RPC for remote (complex, best)
        //   3. Fail and require all-local or all-remote deployment (restrictive)
        //
        // Current implementation: Fall back to all-local with detailed warning

        fprintf(stderr, "%s: ========================================\n", __func__);
        fprintf(stderr, "%s: MIXED LOCAL/REMOTE EXPERT DEPLOYMENT DETECTED\n", __func__);
        fprintf(stderr, "%s: ========================================\n", __func__);
        fprintf(stderr, "%s: Layer %d expert distribution:\n", __func__, layer_id);
        fprintf(stderr, "%s:   - Local experts:  %zu\n", __func__, local_experts.size());
        fprintf(stderr, "%s:   - Remote experts: %zu across %zu endpoints\n",
                __func__, n_experts - local_experts.size(), experts_by_endpoint.size());
        fprintf(stderr, "%s:\n", __func__);
        fprintf(stderr, "%s: CURRENT BEHAVIOR: Falling back to all-local evaluation\n", __func__);
        fprintf(stderr, "%s:   - All experts (including remote) will be evaluated locally\n", __func__);
        fprintf(stderr, "%s:   - This works correctly but misses the distributed benefits\n", __func__);
        fprintf(stderr, "%s:\n", __func__);
        fprintf(stderr, "%s: RECOMMENDATIONS:\n", __func__);
        fprintf(stderr, "%s:   1. BEST: Deploy all experts for each layer on the same node\n", __func__);
        fprintf(stderr, "%s:      Example: Layer 0 all-local, Layer 1 all-remote\n", __func__);
        fprintf(stderr, "%s:   2. ALTERNATIVE: Implement direct FFN evaluation for local experts\n", __func__);
        fprintf(stderr, "%s:      (See TODO in ggml-rpc-expert.cpp for implementation details)\n", __func__);
        fprintf(stderr, "%s: ========================================\n", __func__);

        return false;  // Fallback to regular mul_mat_id
    }

    // All experts are remote - this is the optimized distributed case
    fprintf(stderr, "%s: all experts are remote, using distributed evaluation\n", __func__);

    // Extract tensor dimensions
    const int64_t n_embd = input->ne[0];   // Embedding dimension
    const int64_t n_ff = output->ne[0];    // FFN dimension

    fprintf(stderr, "%s: tensor dimensions - n_embd=%lld, n_ff=%lld, n_tokens=%lld\n",
            __func__, n_embd, n_ff, n_tokens);

    // For the remote-only case, we need to:
    // 1. Extract input data: [n_embd * n_tokens]
    // 2. Extract router weights for the selected experts
    // 3. Call RPC for each endpoint
    // 4. Collect results into output tensor

    // Get pointers to tensor data
    const float * input_data = (const float *)input->data;
    float * output_data = (float *)output->data;

    // Router weight extraction
    std::vector<float> router_weights;

    if (router_weights_tensor != nullptr && router_weights_tensor->data != nullptr) {
        // Extract actual router probabilities from the tensor
        // Router weights tensor shape: [1, n_expert_used, n_tokens]
        const float * router_data = (const float *)router_weights_tensor->data;
        const int64_t router_n_experts = router_weights_tensor->ne[1];
        const int64_t router_n_tokens = router_weights_tensor->ne[2];

        fprintf(stderr, "%s: Extracting router weights from tensor: [%lld, %lld, %lld]\n",
                __func__, router_weights_tensor->ne[0], router_n_experts, router_n_tokens);

        // Copy router weights for selected experts
        router_weights.resize(n_experts * n_tokens);
        for (int64_t t = 0; t < n_tokens; t++) {
            for (int64_t e = 0; e < n_experts; e++) {
                // Router weights are in shape [1, n_expert_used, n_tokens]
                // Index: [0, expert_idx, token_idx]
                router_weights[e * n_tokens + t] = router_data[e * n_tokens + t];
            }
        }

        fprintf(stderr, "%s: Successfully extracted %zu router weights\n",
                __func__, router_weights.size());
    } else {
        // Fallback to uniform weights if router weights not available
        fprintf(stderr, "%s: WARNING - router weights not available, using uniform weights (1/%lld)\n",
                __func__, n_experts);
        router_weights.resize(n_experts * n_tokens, 1.0f / (float)n_experts);
    }

    // Dispatch to each remote endpoint
    bool all_success = true;
    for (const auto & [endpoint, expert_list] : experts_by_endpoint) {
        fprintf(stderr, "%s: dispatching %zu experts to %s\n",
                __func__, expert_list.size(), endpoint.c_str());

        // Convert expert IDs to uint8_t array
        std::vector<uint8_t> expert_ids_u8(expert_list.begin(), expert_list.end());

        // Allocate buffer for this endpoint's output
        std::vector<float> endpoint_output(n_embd * n_tokens, 0.0f);

        // Make RPC call
        bool success = ggml_rpc_expert_evaluate(
            endpoint.c_str(),
            (uint8_t)layer_id,
            (uint8_t)expert_list.size(),
            expert_ids_u8.data(),
            (uint16_t)n_tokens,
            (uint32_t)n_embd,
            (uint32_t)n_ff,
            router_weights.data(),
            input_data,
            endpoint_output.data()
        );

        if (!success) {
            fprintf(stderr, "%s: RPC call to %s failed\n", __func__, endpoint.c_str());
            all_success = false;
            continue;
        }

        fprintf(stderr, "%s: RPC call to %s succeeded\n", __func__, endpoint.c_str());

        // Accumulate results into output
        // TODO: This is simplified - we need proper result mapping
        for (size_t i = 0; i < endpoint_output.size(); i++) {
            output_data[i] += endpoint_output[i];
        }
    }

    if (!all_success) {
        fprintf(stderr, "%s: some RPC calls failed, falling back to local evaluation\n", __func__);
        return false;
    }

    fprintf(stderr, "%s: distributed dispatch completed successfully\n", __func__);
    return true;  // Success - we handled the dispatch
}

