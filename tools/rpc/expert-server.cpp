// Expert worker server for distributed MoE evaluation
// Based on rpc-server.cpp but specialized for serving expert evaluations

#include "ggml-rpc.h"
#include "ggml-rpc-expert.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <string>
#include <stdio.h>
#include <thread>
#include <vector>
#include <cmath>

struct expert_server_params {
    std::string model_path;
    std::string expert_range;   // e.g., "0-31"
    std::string host = "0.0.0.0";
    int         port = 50052;
    int         n_threads = std::max(1U, std::thread::hardware_concurrency()/2);
};

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

        // In production with loaded model:
        // auto * model = (llama_model *)user_data;
        // float * gate_w = get_expert_weight(model, layer_id, expert_id, "gate");
        // float * up_w = get_expert_weight(model, layer_id, expert_id, "up");
        // float * down_w = get_expert_weight(model, layer_id, expert_id, "down");

        // For demonstration/testing without loaded model:
        // Create synthetic expert weights (identity-like transformations)
        // This allows us to test the computation pipeline

        std::vector<float> gate_w(n_ff * n_embd);
        std::vector<float> up_w(n_ff * n_embd);
        std::vector<float> down_w(n_embd * n_ff);

        // Initialize with small random-like values based on expert_id
        // This creates different behavior per expert while being deterministic
        for (size_t i = 0; i < n_ff * n_embd; i++) {
            float val = 0.1f * sinf((float)(i + expert_id * 1000));
            gate_w[i] = val;
            up_w[i] = val * 0.8f;
        }
        for (size_t i = 0; i < n_embd * n_ff; i++) {
            float val = 0.1f * cosf((float)(i + expert_id * 1000));
            down_w[i] = val;
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

    // Register evaluation callback
    // Using actual FFN computation with synthetic weights (for testing/demonstration)
    // For production with real model weights:
    //   llama_model_params model_params = llama_model_default_params();
    //   llama_model * model = llama_load_model_from_file(params.model_path.c_str(), model_params);
    //   ggml_rpc_expert_register_eval_callback(expert_eval_callback, model);
    //   Then update callback to use model->layers[layer_id].ffn_*_exps[expert_id] weights
    ggml_rpc_expert_register_eval_callback(expert_eval_callback, nullptr);
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
    fprintf(stderr, "CURRENT WEIGHTS: Synthetic (deterministic per-expert patterns)\n");
    fprintf(stderr, "PRODUCTION MODE: Load actual model to use real expert weights\n");
    fprintf(stderr, "                 (See commented code in expert_eval_callback)\n");
    fprintf(stderr, "\n");

    // Server loop - worker thread handles requests
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    ggml_rpc_expert_shutdown();
    return 0;
}
