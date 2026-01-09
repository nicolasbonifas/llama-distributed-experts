#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Simplified expert RPC protocol for distributed MoE evaluation
// This is a minimal implementation for testing the concept

#define GGML_RPC_EXPERT_MAGIC      0x45585052 // "EXPR" in hex
#define GGML_RPC_EXPERT_VERSION    1

// Status codes
#define GGML_RPC_EXPERT_STATUS_SUCCESS       0
#define GGML_RPC_EXPERT_STATUS_ERROR         1
#define GGML_RPC_EXPERT_STATUS_NOT_FOUND     2
#define GGML_RPC_EXPERT_STATUS_INVALID_REQ   3

// Request from master to worker: evaluate experts
struct ggml_rpc_expert_request {
    uint32_t magic;              // Protocol magic number
    uint32_t version;            // Protocol version

    uint8_t  layer_id;           // Layer index
    uint8_t  n_experts;          // Number of experts to evaluate
    uint16_t n_tokens;           // Batch size
    uint32_t n_embd;             // Embedding dimension
    uint32_t n_ff;               // Expert FFN dimension

    // Followed by variable-length data:
    // - uint8_t expert_ids[n_experts]
    // - float weights[n_experts * n_tokens]
    // - float input_data[n_embd * n_tokens]
};

// Response from worker to master: weighted expert outputs
struct ggml_rpc_expert_response {
    uint32_t magic;              // Protocol magic number
    uint32_t version;            // Protocol version

    uint8_t  status;             // 0 = success, non-zero = error
    uint16_t n_tokens;
    uint32_t n_embd;

    // Followed by variable-length data:
    // - char error_message[128] (if status != 0)
    // - float weighted_output[n_embd * n_tokens] (if status == 0)
};

// Callback function type for expert evaluation on worker side
// This allows the application layer (llama.cpp) to provide the actual
// expert evaluation logic while keeping ggml layer independent
typedef bool (*ggml_rpc_expert_eval_callback)(
    void * user_data,           // User context (e.g., llama_model*)
    uint8_t layer_id,
    uint8_t n_experts,
    const uint8_t * expert_ids,
    uint16_t n_tokens,
    uint32_t n_embd,
    uint32_t n_ff,
    const float * weights,
    const float * input_data,
    float * output_data
);

// Initialize expert RPC for a model (called once at startup)
GGML_BACKEND_API bool ggml_rpc_expert_init(
    const char * model_path,
    const char * expert_range,  // e.g., "0-31"
    const char * endpoint       // e.g., "0.0.0.0:50052"
);

// Register evaluation callback (worker side only)
GGML_BACKEND_API void ggml_rpc_expert_register_eval_callback(
    ggml_rpc_expert_eval_callback callback,
    void * user_data
);

// Evaluate experts on worker node (synchronous call)
GGML_BACKEND_API bool ggml_rpc_expert_evaluate(
    const char * endpoint,
    uint8_t layer_id,
    uint8_t n_experts,
    const uint8_t * expert_ids,
    uint16_t n_tokens,
    uint32_t n_embd,
    uint32_t n_ff,
    const float * weights,
    const float * input_data,
    float * output_data  // pre-allocated [n_embd * n_tokens]
);

// Cleanup
GGML_BACKEND_API void ggml_rpc_expert_shutdown();

#ifdef __cplusplus
}
#endif
