# Distributed MoE Expert Implementation Status

## ✅ Completed

1. **Command-Line Arguments** (`common/common.h`, `common/arg.cpp`)
   - Added `--expert-worker-mode "0-31"` for worker nodes
   - Added `--expert-rpc-servers "host:port:range,..."` for master node

2. **Expert Range Parsing** (`common/common-expert.cpp`)
   - `parse_expert_range()` - parses "0-31" into set of IDs
   - `is_expert_tensor()` - detects expert tensors
   - Added to CMakeLists.txt

3. **RPC Protocol** (`ggml/include/ggml-rpc-expert.h`, `ggml/src/ggml-rpc/ggml-rpc-expert.cpp`)
   - Basic protocol structures defined
   - Stub implementation (returns zeros for now)
   - Added to ggml-rpc CMakeLists.txt

4. **Model Loader Note** (`src/llama-model-loader.cpp`)
   - Added TODO comment about selective loading
   - Current architecture loads all experts (merged tensors)

## 🚧 TODO: Core Functionality

### 5. Modify `build_moe_ffn()` in `src/llama-graph.cpp`

**Location**: Lines 936-1203

**Current flow**:
```cpp
// Select top-k experts
selected_experts = ggml_argsort_top_k(selection_probs, n_expert_used);
weights = ggml_get_rows(probs, selected_experts);

// Evaluate ALL experts locally using build_lora_mm_id
up = build_lora_mm_id(up_exps, cur, selected_experts);
cur = build_lora_mm_id(gate_exps, cur, selected_experts);
experts = build_lora_mm_id(down_exps, cur, selected_experts);

// Weight and aggregate
experts = ggml_mul(experts, weights);
for (i...) { moe_out = ggml_add(moe_out, cur_experts[i]); }
```

**Required changes**:
1. After selecting experts, split them into local vs remote
2. Evaluate local experts using existing code path
3. For remote experts, call `ggml_rpc_expert_evaluate()`
4. Merge local and remote results

**Pseudocode**:
```cpp
// After line 1051: weights = ggml_get_rows(probs, selected_experts);

// NEW: Check if we have remote experts configured
if (model.has_remote_experts) {
    // Split experts into local/remote sets
    std::vector<int> local_expert_indices;
    std::map<std::string, std::vector<int>> remote_experts_by_endpoint;

    for (int i = 0; i < n_expert_used; ++i) {
        int expert_id = get_expert_id(selected_experts, i);
        std::string endpoint = model.find_expert_endpoint(expert_id, il);

        if (endpoint.empty()) {
            local_expert_indices.push_back(i);
        } else {
            remote_experts_by_endpoint[endpoint].push_back(i);
        }
    }

    // Evaluate local experts
    ggml_tensor* local_output = nullptr;
    if (!local_expert_indices.empty()) {
        // Use existing build_lora_mm_id path for local experts
        local_output = evaluate_local_experts(...);
    }

    // Evaluate remote experts
    std::vector<ggml_tensor*> remote_outputs;
    for (auto& [endpoint, indices] : remote_experts_by_endpoint) {
        // Call RPC
        ggml_tensor* remote_out = call_remote_experts(
            endpoint, il, cur, indices, weights);
        remote_outputs.push_back(remote_out);
    }

    // Merge all outputs
    moe_out = merge_outputs(local_output, remote_outputs);
} else {
    // Original code path for non-distributed mode
    ...
}
```

### 6. Create Expert Worker Server (`tools/rpc/expert-server.cpp`)

Copy `tools/rpc/rpc-server.cpp` and modify:
1. Parse `--expert-worker-mode` argument
2. Call `ggml_rpc_expert_init(model_path, expert_range, endpoint)`
3. Start RPC server to listen for expert evaluation requests

### 7. Add Expert Routing to llama_model

**File**: `src/llama-model.h`

Add structure to track expert-to-endpoint mapping:
```cpp
struct llama_model {
    // ... existing fields ...

    // NEW: Expert distribution
    struct expert_endpoint {
        std::string endpoint;      // "host:port"
        std::set<int> expert_ids;  // experts on this endpoint
    };
    std::vector<expert_endpoint> expert_endpoints;

    std::string find_expert_endpoint(int expert_id, int layer) const;
    bool has_remote_experts = false;
};
```

Parse `--expert-rpc-servers` in `llama_model_load()` and populate this structure.

## Testing Steps

1. **Build**:
   ```bash
   mkdir build && cd build
   cmake .. -DGGML_RPC=ON
   make -j
   ```

2. **Download test model**:
   ```bash
   # Qwen3-30B-A3B or similar MoE model
   ```

3. **Test worker node** (verify it loads only assigned experts):
   ```bash
   ./expert-server --model qwen3-30b.gguf \
                   --expert-worker-mode "0-31" \
                   --rpc "0.0.0.0:50052"
   ```

4. **Test master node** (verify it connects and dispatches):
   ```bash
   ./llama-cli --model qwen3-30b.gguf \
               --expert-rpc-servers "localhost:50052:0-31" \
               --prompt "Hello world"
   ```

## Next Steps for Full Implementation

1. Implement actual network communication in `ggml-rpc-expert.cpp` (socket/gRPC)
2. Implement worker-side expert evaluation logic
3. Add tensor serialization/deserialization
4. Add error handling, timeouts, retries
5. Implement load balancing for redundant experts
6. Add metrics and logging
7. Optimize with partial weighted aggregation on workers

## Known Limitations

- Expert tensors stored in merged form (all in one tensor)
- Can't selectively load individual experts with current architecture
- Need to implement actual network protocol (currently stubs)
- No error handling or timeout logic yet

## Files Modified

- `common/common.h` - Added expert params
- `common/arg.cpp` - Added arguments
- `common/common-expert.cpp` - NEW: parsing utilities
- `common/CMakeLists.txt` - Added common-expert.cpp
- `ggml/include/ggml-rpc-expert.h` - NEW: protocol header
- `ggml/src/ggml-rpc/ggml-rpc-expert.cpp` - NEW: stub implementation
- `ggml/src/ggml-rpc/CMakeLists.txt` - Added expert RPC
- `src/llama-model-loader.cpp` - Added TODO comment
- `src/llama-graph.cpp` - TODO: modify build_moe_ffn
- `src/llama-model.h` - TODO: add expert routing
- `tools/rpc/expert-server.cpp` - TODO: create worker executable
