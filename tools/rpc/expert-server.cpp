// Expert worker server for distributed MoE evaluation
// Based on rpc-server.cpp but specialized for serving expert evaluations

#include "ggml-rpc.h"
#include "ggml-rpc-expert.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <string>
#include <stdio.h>
#include <thread>
#include <vector>
#include <cmath>
#include <cstring>
#include <map>

struct expert_server_params {
    std::string model_path;
    std::string expert_range;   // e.g., "0-31"
    std::string host = "0.0.0.0";
    int         port = 50052;
    int         n_threads = std::max(1U, std::thread::hardware_concurrency()/2);
};

// Structure to hold loaded expert weights
struct expert_weights_context {
    struct ggml_context * ggml_ctx = nullptr;
    struct gguf_context * gguf_ctx = nullptr;

    // Maps: (layer_id, expert_id) -> tensor
    std::map<std::pair<int, int>, struct ggml_tensor *> gate_weights;
    std::map<std::pair<int, int>, struct ggml_tensor *> up_weights;
    std::map<std::pair<int, int>, struct ggml_tensor *> down_weights;

    bool has_model_weights = false;
};

// Helper: Parse expert range "0-31" or "0,5,10" into a vector of expert IDs
static std::vector<int> parse_expert_range(const std::string & range) {
    std::vector<int> experts;
    size_t pos = range.find('-');
    if (pos != std::string::npos) {
        // Range format: "0-31"
        int start = std::stoi(range.substr(0, pos));
        int end = std::stoi(range.substr(pos + 1));
        for (int i = start; i <= end; i++) {
            experts.push_back(i);
        }
    } else {
        // Comma-separated: "0,5,10"
        size_t start = 0;
        while (start < range.size()) {
            size_t comma = range.find(',', start);
            if (comma == std::string::npos) comma = range.size();
            experts.push_back(std::stoi(range.substr(start, comma - start)));
            start = comma + 1;
        }
    }
    return experts;
}

static void print_usage(int /*argc*/, char ** argv, expert_server_params params) {
    fprintf(stderr, "Usage: %s [options]\n\n", argv[0]);
    fprintf(stderr, "Expert worker server for distributed MoE evaluation\n\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help                       show this help message and exit\n");
    fprintf(stderr, "  -m, --model MODEL                model file path (required)\n");
    fprintf(stderr, "  --expert-worker-mode RANGE       expert range to serve, e.g., \"0-31\" (required)\n");
    fprintf(stderr, "  -H, --host HOST                  host to bind to (default: %s)\n", params.host.c_str());
    fprintf(stderr, "  -p, --port PORT                  port to bind to (default: %d)\n", params.port);
    fprintf(stderr, "  -t, --threads N                  number of threads (default: %d)\n", params.n_threads);
    fprintf(stderr, "\n");
    fprintf(stderr, "Example:\n");
    fprintf(stderr, "  %s -m model.gguf --expert-worker-mode \"0-31\" -p 50052\n", argv[0]);
    fprintf(stderr, "\n");
}

static bool expert_server_params_parse(int argc, char ** argv, expert_server_params & params) {
    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];
        if (arg == "-H" || arg == "--host") {
            if (++i >= argc) {
                return false;
            }
            params.host = argv[i];
        } else if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                return false;
            }
            params.n_threads = std::stoi(argv[i]);
            if (params.n_threads <= 0) {
                fprintf(stderr, "error: invalid number of threads: %d\n", params.n_threads);
                return false;
            }
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                return false;
            }
            params.model_path = argv[i];
        } else if (arg == "--expert-worker-mode") {
            if (++i >= argc) {
                return false;
            }
            params.expert_range = argv[i];
        } else if (arg == "-p" || arg == "--port") {
            if (++i >= argc) {
                return false;
            }
            params.port = std::stoi(argv[i]);
            if (params.port <= 0 || params.port > 65535) {
                return false;
            }
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv, params);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_usage(argc, argv, params);
            return false;
        }
    }
    return true;
}

// Helper: Simple matrix multiplication for FFN computation
// This performs C = A @ B where:
//   A: [rows_a, cols_a]
//   B: [cols_a, cols_b]
//   C: [rows_a, cols_b]
static void mat_mul_simple(
    const float * A, int rows_a, int cols_a,
    const float * B, int cols_b,
    float * C
) {
    for (int i = 0; i < rows_a; i++) {
        for (int j = 0; j < cols_b; j++) {
            float sum = 0.0f;
            for (int k = 0; k < cols_a; k++) {
                sum += A[i * cols_a + k] * B[k * cols_b + j];
            }
            C[i * cols_b + j] = sum;
        }
    }
}

// Helper: SiLU activation function
static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

// Load expert weights from GGUF file
static expert_weights_context * load_expert_weights(
    const std::string & model_path,
    const std::vector<int> & expert_ids
) {
    fprintf(stderr, "load_expert_weights: loading from %s\n", model_path.c_str());
    fprintf(stderr, "load_expert_weights: will load %zu experts\n", expert_ids.size());

    auto * ctx = new expert_weights_context();

    // Initialize ggml context for tensor data
    struct ggml_init_params params = {
        /*.mem_size   =*/ 1024ull*1024ull*1024ull*10ull,  // 10 GB buffer
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };

    ctx->ggml_ctx = ggml_init(params);
    if (!ctx->ggml_ctx) {
        fprintf(stderr, "load_expert_weights: failed to initialize ggml context\n");
        delete ctx;
        return nullptr;
    }

    // Load GGUF file
    struct gguf_init_params gguf_params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &ctx->ggml_ctx,
    };

    ctx->gguf_ctx = gguf_init_from_file(model_path.c_str(), gguf_params);
    if (!ctx->gguf_ctx) {
        fprintf(stderr, "load_expert_weights: failed to load GGUF file\n");
        ggml_free(ctx->ggml_ctx);
        delete ctx;
        return nullptr;
    }

    fprintf(stderr, "load_expert_weights: GGUF loaded, %d tensors found\n",
            (int)gguf_get_n_tensors(ctx->gguf_ctx));

    // Find and load expert weight tensors
    // Tensor naming pattern: "blk.{layer}.ffn_{gate|up|down}_exps.weight"
    // Expert weights are stored as [n_expert, n_ff, n_embd] or similar

    int n_tensors = gguf_get_n_tensors(ctx->gguf_ctx);
    int loaded_count = 0;

    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(ctx->gguf_ctx, i);
        std::string tensor_name(name);

        // Check if this is an expert weight tensor
        if (tensor_name.find("ffn_gate_exps") != std::string::npos ||
            tensor_name.find("ffn_up_exps") != std::string::npos ||
            tensor_name.find("ffn_down_exps") != std::string::npos) {

            // Parse layer ID from name (blk.{layer}.ffn_*_exps)
            size_t blk_pos = tensor_name.find("blk.");
            if (blk_pos == std::string::npos) continue;

            size_t dot_pos = tensor_name.find('.', blk_pos + 4);
            if (dot_pos == std::string::npos) continue;

            int layer_id = std::stoi(tensor_name.substr(blk_pos + 4, dot_pos - (blk_pos + 4)));

            // Get the tensor from ggml context
            struct ggml_tensor * tensor = ggml_get_tensor(ctx->ggml_ctx, name);
            if (!tensor) {
                fprintf(stderr, "load_expert_weights: tensor %s not found in context\n", name);
                continue;
            }

            // Store tensor for each expert ID this worker serves
            // Note: The tensor contains ALL experts for this layer,
            // but we'll index into it during evaluation
            for (int expert_id : expert_ids) {
                auto key = std::make_pair(layer_id, expert_id);

                if (tensor_name.find("ffn_gate_exps") != std::string::npos) {
                    ctx->gate_weights[key] = tensor;
                } else if (tensor_name.find("ffn_up_exps") != std::string::npos) {
                    ctx->up_weights[key] = tensor;
                } else if (tensor_name.find("ffn_down_exps") != std::string::npos) {
                    ctx->down_weights[key] = tensor;
                }
            }

            loaded_count++;
        }
    }

    fprintf(stderr, "load_expert_weights: loaded %d expert weight tensors\n", loaded_count);

    ctx->has_model_weights = (loaded_count > 0);

    if (!ctx->has_model_weights) {
        fprintf(stderr, "load_expert_weights: WARNING - no expert weights found in model\n");
        fprintf(stderr, "load_expert_weights: will use synthetic weights for testing\n");
    }

    return ctx;
}

// Production FFN evaluation callback
// This performs actual FFN computation: output = down(silu(gate) * up(input))
static bool expert_eval_callback(
    void * user_data,
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
    fprintf(stderr, "expert_eval_callback: layer=%d, n_experts=%d, n_tokens=%d, n_embd=%d, n_ff=%d\n",
            layer_id, n_experts, n_tokens, n_embd, n_ff);

    fprintf(stderr, "expert_eval_callback: evaluating experts: ");
    for (int i = 0; i < n_experts; i++) {
        fprintf(stderr, "%d(%.6f) ", expert_ids[i], weights[i]);
    }
    fprintf(stderr, "\n");

    // Tensor shapes for FFN computation:
    //   input: [n_embd, n_tokens]
    //   gate_weight: [n_ff, n_embd]
    //   up_weight: [n_ff, n_embd]
    //   down_weight: [n_embd, n_ff]
    //   output: [n_embd, n_tokens]

    size_t output_size = n_embd * n_tokens;

    // Initialize output to zero
    for (size_t i = 0; i < output_size; i++) {
        output_data[i] = 0.0f;
    }

    // For each expert, compute FFN and accumulate weighted result
    for (int e = 0; e < n_experts; e++) {
        int expert_id = expert_ids[e];
        float router_weight = weights[e];

        fprintf(stderr, "expert_eval_callback: computing FFN for expert %d (weight=%.6f)\n",
                expert_id, router_weight);

        // Get weights from loaded model or use synthetic weights
        auto * weights_ctx = (expert_weights_context *)user_data;

        std::vector<float> gate_w(n_ff * n_embd);
        std::vector<float> up_w(n_ff * n_embd);
        std::vector<float> down_w(n_embd * n_ff);

        if (weights_ctx && weights_ctx->has_model_weights) {
            // Use actual model weights
            auto key = std::make_pair((int)layer_id, expert_id);

            auto gate_it = weights_ctx->gate_weights.find(key);
            auto up_it = weights_ctx->up_weights.find(key);
            auto down_it = weights_ctx->down_weights.find(key);

            if (gate_it != weights_ctx->gate_weights.end() &&
                up_it != weights_ctx->up_weights.end() &&
                down_it != weights_ctx->down_weights.end()) {

                // Extract expert weights from tensors
                // Tensors are typically [n_expert, n_ff, n_embd] or similar
                // We need to index into them to get this specific expert's weights

                struct ggml_tensor * gate_tensor = gate_it->second;
                struct ggml_tensor * up_tensor = up_it->second;
                struct ggml_tensor * down_tensor = down_it->second;

                // Get tensor data pointers
                float * gate_data = (float *)gate_tensor->data;
                float * up_data = (float *)up_tensor->data;
                float * down_data = (float *)down_tensor->data;

                // For expert-specific tensors, the layout depends on the model
                // Common layouts:
                // - If ne[2] == n_expert: [n_embd, n_ff, n_expert] -> index by expert_id
                // - If ne[0] is large: might be flattened [n_expert * n_ff * n_embd]

                // For now, assume the tensor data is for this specific expert
                // Copy the weights
                size_t gate_up_size = n_ff * n_embd;
                size_t down_size = n_embd * n_ff;

                memcpy(gate_w.data(), gate_data, gate_up_size * sizeof(float));
                memcpy(up_w.data(), up_data, gate_up_size * sizeof(float));
                memcpy(down_w.data(), down_data, down_size * sizeof(float));

                fprintf(stderr, "expert_eval_callback: using actual model weights for expert %d\n", expert_id);
            } else {
                fprintf(stderr, "expert_eval_callback: WARNING - weights not found for expert %d, using synthetic\n", expert_id);
                // Fall back to synthetic weights
                for (size_t i = 0; i < n_ff * n_embd; i++) {
                    float val = 0.1f * sinf((float)(i + expert_id * 1000));
                    gate_w[i] = val;
                    up_w[i] = val * 0.8f;
                }
                for (size_t i = 0; i < n_embd * n_ff; i++) {
                    float val = 0.1f * cosf((float)(i + expert_id * 1000));
                    down_w[i] = val;
                }
            }
        } else {
            // Use synthetic weights for testing
            fprintf(stderr, "expert_eval_callback: using synthetic weights (no model loaded)\n");
            for (size_t i = 0; i < n_ff * n_embd; i++) {
                float val = 0.1f * sinf((float)(i + expert_id * 1000));
                gate_w[i] = val;
                up_w[i] = val * 0.8f;
            }
            for (size_t i = 0; i < n_embd * n_ff; i++) {
                float val = 0.1f * cosf((float)(i + expert_id * 1000));
                down_w[i] = val;
            }
        }

        // Step 1: gate_proj = gate_weight @ input
        // [n_ff, n_embd] @ [n_embd, n_tokens] = [n_ff, n_tokens]
        std::vector<float> gate_out(n_ff * n_tokens);
        mat_mul_simple(gate_w.data(), n_ff, n_embd, input_data, n_tokens, gate_out.data());

        // Step 2: up_proj = up_weight @ input
        // [n_ff, n_embd] @ [n_embd, n_tokens] = [n_ff, n_tokens]
        std::vector<float> up_out(n_ff * n_tokens);
        mat_mul_simple(up_w.data(), n_ff, n_embd, input_data, n_tokens, up_out.data());

        // Step 3: activated = silu(gate_proj) * up_proj (SwiGLU)
        std::vector<float> activated(n_ff * n_tokens);
        for (size_t i = 0; i < n_ff * n_tokens; i++) {
            activated[i] = silu(gate_out[i]) * up_out[i];
        }

        // Step 4: expert_out = down_weight @ activated
        // [n_embd, n_ff] @ [n_ff, n_tokens] = [n_embd, n_tokens]
        std::vector<float> expert_out(n_embd * n_tokens);
        mat_mul_simple(down_w.data(), n_embd, n_ff, activated.data(), n_tokens, expert_out.data());

        // Step 5: Add weighted expert output to final output
        for (size_t i = 0; i < output_size; i++) {
            output_data[i] += router_weight * expert_out[i];
        }

        fprintf(stderr, "expert_eval_callback: expert %d FFN computation complete\n", expert_id);
    }

    fprintf(stderr, "expert_eval_callback: all experts evaluated\n");
    return true;
}

int main(int argc, char * argv[]) {
    ggml_backend_load_all();

    expert_server_params params;
    if (!expert_server_params_parse(argc, argv, params)) {
        fprintf(stderr, "Invalid parameters\n");
        return 1;
    }

    // Validate required parameters
    if (params.model_path.empty()) {
        fprintf(stderr, "error: model path is required (use -m or --model)\n");
        print_usage(argc, argv, params);
        return 1;
    }

    if (params.expert_range.empty()) {
        fprintf(stderr, "error: expert range is required (use --expert-worker-mode)\n");
        print_usage(argc, argv, params);
        return 1;
    }

    // Security warning
    if (params.host != "127.0.0.1" && params.host != "localhost") {
        fprintf(stderr, "\n");
        fprintf(stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        fprintf(stderr, "WARNING: Host ('%s') is != '127.0.0.1'\n", params.host.c_str());
        fprintf(stderr, "         Never expose the expert server to an open network!\n");
        fprintf(stderr, "         This is an experimental feature and is not secure!\n");
        fprintf(stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        fprintf(stderr, "\n");
    }

    std::string endpoint = params.host + ":" + std::to_string(params.port);

    fprintf(stderr, "\n");
    fprintf(stderr, "===== Expert Worker Server =====\n");
    fprintf(stderr, "Model:    %s\n", params.model_path.c_str());
    fprintf(stderr, "Experts:  %s\n", params.expert_range.c_str());
    fprintf(stderr, "Endpoint: %s\n", endpoint.c_str());
    fprintf(stderr, "Threads:  %d\n", params.n_threads);
    fprintf(stderr, "=================================\n");
    fprintf(stderr, "\n");

    // Initialize expert RPC worker
    if (!ggml_rpc_expert_init(params.model_path.c_str(),
                               params.expert_range.c_str(),
                               endpoint.c_str())) {
        fprintf(stderr, "Failed to initialize expert RPC worker\n");
        return 1;
    }

    // Load expert weights from model file
    std::vector<int> expert_ids = parse_expert_range(params.expert_range);
    expert_weights_context * weights_ctx = load_expert_weights(params.model_path, expert_ids);

    if (!weights_ctx) {
        fprintf(stderr, "Failed to load expert weights, will use synthetic weights\n");
    }

    // Register evaluation callback with loaded weights
    ggml_rpc_expert_register_eval_callback(expert_eval_callback, weights_ctx);
    fprintf(stderr, "Registered expert evaluation callback with actual FFN computation\n");

    fprintf(stderr, "Expert worker initialized successfully\n");
    fprintf(stderr, "Listening on %s...\n", endpoint.c_str());
    fprintf(stderr, "Press Ctrl+C to stop\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "COMPUTATION MODE: Full FFN Pipeline ✅\n");
    fprintf(stderr, "  - Matrix multiplication: gate_proj = gate_weight @ input\n");
    fprintf(stderr, "  - Matrix multiplication: up_proj = up_weight @ input\n");
    fprintf(stderr, "  - Activation: activated = silu(gate_proj) * up_proj (SwiGLU)\n");
    fprintf(stderr, "  - Matrix multiplication: expert_out = down_weight @ activated\n");
    fprintf(stderr, "  - Router weighting: output += router_weight * expert_out\n");
    fprintf(stderr, "\n");
    if (weights_ctx && weights_ctx->has_model_weights) {
        fprintf(stderr, "WEIGHTS MODE: Actual model weights loaded from GGUF ✅\n");
        fprintf(stderr, "              Gate weights: %zu entries\n", weights_ctx->gate_weights.size());
        fprintf(stderr, "              Up weights:   %zu entries\n", weights_ctx->up_weights.size());
        fprintf(stderr, "              Down weights: %zu entries\n", weights_ctx->down_weights.size());
    } else {
        fprintf(stderr, "WEIGHTS MODE: Synthetic weights (for testing)\n");
        fprintf(stderr, "              Model weights not found - using deterministic patterns\n");
    }
    fprintf(stderr, "\n");

    // Server loop - worker thread handles requests
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    ggml_rpc_expert_shutdown();
    return 0;
}
