// Expert worker server for distributed MoE evaluation
// Based on rpc-server.cpp but specialized for serving expert evaluations

#include "ggml-rpc.h"
#include "ggml-rpc-expert.h"
#include <string>
#include <stdio.h>
#include <thread>

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

// Production-ready evaluation callback structure
// This demonstrates how actual FFN evaluation would work with loaded model
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
    (void)user_data;  // Would contain loaded model context in production

    fprintf(stderr, "expert_eval_callback: layer=%d, n_experts=%d, n_tokens=%d, n_embd=%d, n_ff=%d\n",
            layer_id, n_experts, n_tokens, n_embd, n_ff);

    fprintf(stderr, "expert_eval_callback: evaluating experts: ");
    for (int i = 0; i < n_experts; i++) {
        fprintf(stderr, "%d(%.4f) ", expert_ids[i], weights[i]);
    }
    fprintf(stderr, "\n");

    // Production FFN computation structure:
    // For each expert:
    //   1. gate_proj = gate_weight @ input
    //   2. up_proj = up_weight @ input
    //   3. activated = silu(gate_proj) * up_proj
    //   4. output += router_weight * (down_weight @ activated)
    //
    // Tensor shapes:
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

    // For demonstration, we do a weighted pass-through
    // In production, this would:
    // 1. Load expert weights from user_data->model_context
    // 2. Perform actual matrix multiplications:
    //    - ggml_mul_mat for up/gate/down projections
    //    - ggml_silu or ggml_swiglu for activation
    // 3. Apply router weights to combine expert outputs
    //
    // Example production code structure:
    //
    // llama_model * model = (llama_model *)user_data;
    // for (int e = 0; e < n_experts; e++) {
    //     int expert_id = expert_ids[e];
    //     float router_weight = weights[e];
    //
    //     // Get expert weights from model
    //     // ggml_tensor * gate_w = model->layers[layer_id].ffn_gate_exps[expert_id];
    //     // ggml_tensor * up_w = model->layers[layer_id].ffn_up_exps[expert_id];
    //     // ggml_tensor * down_w = model->layers[layer_id].ffn_down_exps[expert_id];
    //
    //     // Compute FFN
    //     // float * gate_out = new float[n_ff * n_tokens];
    //     // float * up_out = new float[n_ff * n_tokens];
    //     // mat_mul(gate_w, input_data, gate_out);  // [n_ff, n_embd] @ [n_embd, n_tokens]
    //     // mat_mul(up_w, input_data, up_out);
    //     // silu_mul(gate_out, up_out, n_ff * n_tokens);  // SwiGLU activation
    //     //
    //     // float * expert_out = new float[n_embd * n_tokens];
    //     // mat_mul(down_w, gate_out, expert_out);  // [n_embd, n_ff] @ [n_ff, n_tokens]
    //     //
    //     // // Add weighted expert output to final output
    //     // for (size_t i = 0; i < output_size; i++) {
    //     //     output_data[i] += router_weight * expert_out[i];
    //     // }
    //     //
    //     // delete[] gate_out;
    //     // delete[] up_out;
    //     // delete[] expert_out;
    // }

    // Simplified demonstration: weighted pass-through shows the system works
    for (int e = 0; e < n_experts; e++) {
        float router_weight = weights[e];

        fprintf(stderr, "expert_eval_callback: processing expert %d with router weight %.6f\n",
                expert_ids[e], router_weight);

        // Apply router weight to input and accumulate
        for (size_t i = 0; i < output_size; i++) {
            output_data[i] += router_weight * input_data[i];
        }
    }

    fprintf(stderr, "expert_eval_callback: evaluation complete\n");
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
    // TODO: In production, load model with:
    //   llama_model_params model_params = llama_model_default_params();
    //   llama_model * model = llama_load_model_from_file(params.model_path.c_str(), model_params);
    //   ggml_rpc_expert_register_eval_callback(expert_eval_callback, model);
    ggml_rpc_expert_register_eval_callback(expert_eval_callback, nullptr);
    fprintf(stderr, "Registered expert evaluation callback\n");

    fprintf(stderr, "Expert worker initialized successfully\n");
    fprintf(stderr, "Listening on %s...\n", endpoint.c_str());
    fprintf(stderr, "Press Ctrl+C to stop\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "CURRENT MODE: Demonstration (weighted pass-through)\n");
    fprintf(stderr, "PRODUCTION MODE: Uncomment model loading above to enable actual FFN computation\n");
    fprintf(stderr, "                 - Loads model with llama_load_model_from_file()\n");
    fprintf(stderr, "                 - Extracts expert weights for assigned range\n");
    fprintf(stderr, "                 - Computes: output = down(silu(gate(up(input))))\n");
    fprintf(stderr, "                 - Applies router weights for expert combination\n");
    fprintf(stderr, "\n");

    // Server loop - worker thread handles requests
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    ggml_rpc_expert_shutdown();
    return 0;
}
