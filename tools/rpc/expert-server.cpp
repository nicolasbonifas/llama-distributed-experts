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

// Simple evaluation callback for testing
// In production, this would load the model and do actual FFN computation
static bool simple_eval_callback(
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
    (void)user_data;
    (void)n_ff;

    fprintf(stderr, "simple_eval_callback: layer=%d, n_experts=%d, n_tokens=%d, n_embd=%d\n",
            layer_id, n_experts, n_tokens, n_embd);

    fprintf(stderr, "simple_eval_callback: evaluating experts: ");
    for (int i = 0; i < n_experts; i++) {
        fprintf(stderr, "%d ", expert_ids[i]);
    }
    fprintf(stderr, "\n");

    // Simple demonstration: apply weighted pass-through
    // In production, this would be: up -> gate -> down FFN computation
    // For now, we just apply the expert weights to show the system works

    size_t output_size = n_embd * n_tokens;

    // Initialize output to zero
    for (size_t i = 0; i < output_size; i++) {
        output_data[i] = 0.0f;
    }

    // For each expert, add weighted input to output
    for (int e = 0; e < n_experts; e++) {
        float expert_weight = weights[e];  // Simplified: using first weight for each expert

        fprintf(stderr, "simple_eval_callback: expert %d weight: %.4f\n",
                expert_ids[e], expert_weight);

        // Add weighted input to output
        for (size_t i = 0; i < output_size; i++) {
            output_data[i] += expert_weight * input_data[i];
        }
    }

    fprintf(stderr, "simple_eval_callback: evaluation complete\n");
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
    // TODO: In production, load model and use real FFN evaluation
    ggml_rpc_expert_register_eval_callback(simple_eval_callback, nullptr);
    fprintf(stderr, "Registered simple evaluation callback (weighted pass-through)\n");

    fprintf(stderr, "Expert worker initialized successfully\n");
    fprintf(stderr, "Listening on %s...\n", endpoint.c_str());
    fprintf(stderr, "Press Ctrl+C to stop\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "NOTE: Using simple demonstration callback\n");
    fprintf(stderr, "      For production use, implement actual FFN evaluation\n");
    fprintf(stderr, "\n");

    // Server loop - worker thread handles requests
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    ggml_rpc_expert_shutdown();
    return 0;
}
