// Simple test for expert RPC protocol
// Tests that client-server communication works end-to-end

#include "ggml-rpc-expert.h"
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <thread>
#include <chrono>

// Simple evaluation callback that returns the input scaled by 2.0
static bool test_eval_callback(
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
    (void)layer_id;
    (void)expert_ids;
    (void)n_ff;
    (void)weights;

    fprintf(stderr, "test_eval_callback: evaluating %d experts for %d tokens\n", n_experts, n_tokens);
    fprintf(stderr, "test_eval_callback: n_embd=%d, n_ff=%d\n", n_embd, n_ff);

    // Simple test: scale input by 2.0
    size_t n = n_embd * n_tokens;
    for (size_t i = 0; i < n; i++) {
        output_data[i] = input_data[i] * 2.0f;
    }

    fprintf(stderr, "test_eval_callback: returning scaled output\n");
    return true;
}

int main() {
    fprintf(stderr, "=== Expert RPC Protocol Test ===\n\n");

    const char * endpoint = "127.0.0.1:50099";
    const int n_embd = 4;
    const int n_ff = 8;
    const int n_tokens = 2;
    const int n_experts = 2;

    // Start server in background thread
    fprintf(stderr, "Starting expert RPC server on %s...\n", endpoint);
    std::thread server_thread([endpoint]() {
        bool success = ggml_rpc_expert_init("test_model", "0-7", endpoint);
        if (!success) {
            fprintf(stderr, "Failed to initialize expert RPC server\n");
            return;
        }

        // Register evaluation callback
        ggml_rpc_expert_register_eval_callback(test_eval_callback, nullptr);

        // Server runs indefinitely (we'll kill it when test is done)
        fprintf(stderr, "Server running...\n");
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    });

    // Give server time to start
    fprintf(stderr, "Waiting for server to start...\n");
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Prepare test data
    fprintf(stderr, "\nPreparing test data...\n");
    uint8_t expert_ids[] = {0, 1};
    float weights[n_experts * n_tokens];
    for (int i = 0; i < n_experts * n_tokens; i++) {
        weights[i] = 1.0f / n_experts;  // Uniform weights
    }

    float input_data[n_embd * n_tokens];
    for (int i = 0; i < n_embd * n_tokens; i++) {
        input_data[i] = (float)(i + 1);  // 1.0, 2.0, 3.0, ...
    }

    float output_data[n_embd * n_tokens];
    memset(output_data, 0, sizeof(output_data));

    // Make RPC call
    fprintf(stderr, "\nMaking RPC call to %s...\n", endpoint);
    bool success = ggml_rpc_expert_evaluate(
        endpoint,
        0,  // layer_id
        n_experts,
        expert_ids,
        n_tokens,
        n_embd,
        n_ff,
        weights,
        input_data,
        output_data
    );

    if (!success) {
        fprintf(stderr, "\n❌ TEST FAILED: RPC call failed\n");
        return 1;
    }

    // Verify results
    fprintf(stderr, "\nVerifying results...\n");
    fprintf(stderr, "Input:  ");
    for (int i = 0; i < n_embd * n_tokens; i++) {
        fprintf(stderr, "%.1f ", input_data[i]);
    }
    fprintf(stderr, "\n");

    fprintf(stderr, "Output: ");
    for (int i = 0; i < n_embd * n_tokens; i++) {
        fprintf(stderr, "%.1f ", output_data[i]);
    }
    fprintf(stderr, "\n");

    // Expected: input * 2.0
    bool correct = true;
    for (int i = 0; i < n_embd * n_tokens; i++) {
        float expected = input_data[i] * 2.0f;
        if (output_data[i] != expected) {
            fprintf(stderr, "Mismatch at index %d: expected %.1f, got %.1f\n",
                    i, expected, output_data[i]);
            correct = false;
        }
    }

    if (correct) {
        fprintf(stderr, "\n✅ TEST PASSED: RPC protocol works correctly!\n");
    } else {
        fprintf(stderr, "\n❌ TEST FAILED: Output mismatch\n");
        return 1;
    }

    fprintf(stderr, "\nTest complete. Cleaning up...\n");
    ggml_rpc_expert_shutdown();

    return 0;
}
