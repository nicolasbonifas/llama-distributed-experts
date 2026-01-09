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

// Global state for expert RPC
struct expert_rpc_state {
    bool initialized = false;
    std::string model_path;
    std::string expert_range;
    std::string endpoint;
    int server_socket = -1;

    // Connection pool for client-side
    std::map<std::string, int> connections;
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

    // TODO: Load model with only assigned experts
    // TODO: Start worker thread to accept connections and handle requests

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

void ggml_rpc_expert_shutdown() {
    fprintf(stderr, "%s: shutting down expert RPC\n", __func__);

    // Close all client connections
    for (auto & conn : g_expert_rpc.connections) {
        close(conn.second);
    }
    g_expert_rpc.connections.clear();

    // Close server socket
    if (g_expert_rpc.server_socket >= 0) {
        close(g_expert_rpc.server_socket);
        g_expert_rpc.server_socket = -1;
    }

    g_expert_rpc.initialized = false;
}
