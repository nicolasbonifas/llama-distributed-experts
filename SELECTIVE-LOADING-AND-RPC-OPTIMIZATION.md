# Implementation Deep Dive: Selective Loading & RPC Optimization

## 1. Selective Expert Loading - Implementation Guide

### Understanding Expert Tensor Layout

Expert tensors in llama.cpp are stored as **3D tensors** with all experts concatenated:

```cpp
// Example for ffn_gate_exps tensor
// Shape: [n_ff, n_embd, n_expert]
//
// For Qwen3-30B-A3B with 64 experts:
// Shape: [14336, 3584, 64]
//
// Memory layout (row-major):
// [expert_0_data][expert_1_data][expert_2_data]...[expert_63_data]
```

When `ggml_mul_mat_id()` is called with `ids=[5, 12, 7, 19]`, it selects:
- Expert 5's slice at index 5
- Expert 12's slice at index 12
- etc.

### What's Needed for Selective Loading

#### Step 1: Modify GGUF Tensor Loading

**File**: `src/llama-model-loader.cpp`

Currently, the loader reads entire tensors. You need to add **partial tensor loading**:

```cpp
// New function to load only a slice of a tensor
struct ggml_tensor * llama_model_loader::create_tensor_slice(
    struct ggml_context * ctx,
    const std::string & name,
    const std::initializer_list<int64_t> & ne,
    const std::set<int> & expert_ids,  // NEW: which experts to load
    int total_experts                   // NEW: total expert count
) {
    // 1. Find tensor in GGUF file
    const struct ggml_tensor * cur = check_tensor_dims(name, ne, true);

    // 2. Calculate byte offsets for desired experts
    // Experts are in the last dimension
    size_t expert_size = ggml_nbytes(cur) / total_experts;
    std::vector<size_t> offsets;
    for (int expert_id : expert_ids) {
        offsets.push_back(expert_id * expert_size);
    }

    // 3. Create smaller tensor with new dimensions
    std::vector<int64_t> new_ne = ne;
    new_ne.back() = expert_ids.size();  // Reduce expert dimension
    struct ggml_tensor * tensor = ggml_new_tensor(ctx, cur->type, new_ne.size(), new_ne.data());

    // 4. Read only the required expert slices from file
    for (size_t i = 0; i < expert_ids.size(); ++i) {
        void * dst = (char*)tensor->data + i * expert_size;
        void * src = (char*)cur->data + offsets[i];
        memcpy(dst, src, expert_size);  // Or use mmap if available
    }

    return tensor;
}
```

#### Step 2: Add Expert ID Remapping

**Problem**: Worker has experts 16-31, but model code asks for expert IDs 0-63.

**Solution**: Add a remapping layer:

```cpp
// In llama_model or llama_context
struct expert_id_mapper {
    std::map<int, int> global_to_local;  // Maps global ID -> local index
    std::set<int> local_expert_ids;       // IDs this worker has

    // Initialize from expert range "16-31"
    void init(const std::string & range) {
        local_expert_ids = parse_expert_range(range);
        int local_idx = 0;
        for (int global_id : local_expert_ids) {
            global_to_local[global_id] = local_idx++;
        }
    }

    // Remap expert ID for tensor indexing
    int remap(int global_expert_id) const {
        auto it = global_to_local.find(global_expert_id);
        if (it == global_to_local.end()) {
            return -1;  // Expert not on this worker
        }
        return it->second;
    }
};
```

#### Step 3: Update Expert Selection in build_moe_ffn

```cpp
// In build_moe_ffn(), after expert selection
ggml_tensor * selected_experts = ggml_argsort_top_k(selection_probs, n_expert_used);

// NEW: Remap expert IDs if in worker mode
if (expert_worker_mode) {
    // selected_experts contains global IDs [5, 12, 7, 19]
    // Remap to local indices [5, 12, 7, 19] -> [0, 2, 1, 3] (for worker with experts 5-20)
    for (int i = 0; i < n_expert_used; ++i) {
        int global_id = selected_experts->data[i];
        int local_id = mapper.remap(global_id);
        if (local_id == -1) {
            // This expert is not on this worker - should be on a remote worker
            // Handle via RPC
        } else {
            selected_experts->data[i] = local_id;
        }
    }
}

// Now use remapped IDs with reduced expert tensors
ggml_tensor * up = build_lora_mm_id(up_exps, cur, selected_experts);
```

#### Step 4: Memory Savings Calculation

For Qwen3-30B-A3B:
- Total experts: 64
- Each expert ~450MB (weights only)
- Total expert weight: ~29GB

**With selective loading** (16 experts per worker):
- Worker loads: 16 × 450MB = ~7.2GB
- **Savings: 75% memory per worker**

### Implementation Checklist

- [ ] Add `create_tensor_slice()` to model loader
- [ ] Add expert ID remapping structure
- [ ] Modify tensor creation to use `create_tensor_slice()` when in worker mode
- [ ] Update `build_moe_ffn()` to remap IDs before `ggml_mul_mat_id()`
- [ ] Test that remapping works correctly
- [ ] Verify memory usage is reduced

---

## 2. Minimizing RPC Latency - Optimization Guide

Latency is **critical** for distributed MoE. Here's a complete optimization strategy:

### Latency Budget Analysis

For a single token generation with 2 experts selected:

```
Total latency = Network RTT + Serialization + Computation + Deserialization

Example breakdown:
- Network RTT:       5-10ms (local network) or 50-100ms (WAN)
- Serialization:     0.5-2ms (depends on size)
- Expert compute:    10-50ms (depends on hardware)
- Deserialization:   0.5-2ms

Total: 16-164ms per token
```

**Goal**: Keep network overhead < 10% of computation time.

### Strategy 1: Efficient Serialization

**Option A: Protocol Buffers** (Recommended)
```protobuf
// expert_rpc.proto
message ExpertRequest {
  uint32 layer_id = 1;
  repeated uint32 expert_ids = 2;  // Packed encoding
  uint32 n_tokens = 3;
  uint32 n_embd = 4;
  bytes input_data = 5;       // Raw float array
  bytes weights = 6;          // Raw float array
}

message ExpertResponse {
  uint32 status = 1;
  bytes output_data = 2;      // Pre-aggregated result
}
```

**Why**:
- Binary format (no text overhead)
- Automatic field numbering (version compatibility)
- Efficient varint encoding
- ~50-70% smaller than JSON

**Option B: Raw Binary** (Fastest)
```cpp
struct ExpertRequestHeader {
    uint32_t magic = 0x45585052;
    uint8_t  layer_id;
    uint8_t  n_experts;
    uint16_t n_tokens;
    uint32_t n_embd;
    // Immediately followed by raw float arrays
} __attribute__((packed));

// Send: header + expert_ids + weights + input_data
// All in one contiguous buffer, no copying
```

**Why**:
- Zero serialization overhead
- Directly memcpy from tensor data
- Fastest possible (but less flexible)

### Strategy 2: Connection Pooling

```cpp
class ExpertRPCClient {
    std::unordered_map<std::string, std::vector<Connection*>> connection_pools;

    Connection* get_connection(const std::string& endpoint) {
        auto& pool = connection_pools[endpoint];
        if (pool.empty()) {
            return new Connection(endpoint);  // Create new
        }
        Connection* conn = pool.back();
        pool.pop_back();
        return conn;
    }

    void return_connection(const std::string& endpoint, Connection* conn) {
        connection_pools[endpoint].push_back(conn);  // Reuse
    }
};
```

**Impact**: Eliminates TCP handshake overhead (~3-5ms per request).

### Strategy 3: Token Batching (Prompt Processing)

**Scenario**: During prompt processing, multiple tokens are evaluated at the same layer simultaneously.

**Solution**: Batch all tokens' expert requests together:

```cpp
// Processing 10 tokens at layer 0:
// Token 0: needs experts [5, 7]  from endpoint A
// Token 1: needs experts [5, 12] from endpoint A
// Token 2: needs experts [7, 19] from endpoint A

// Batch into single request:
Request to endpoint A: {
    layer: 0,
    experts: [5, 7, 12, 19],              // Union of all needed
    inputs: [token0, token1, token2],     // All token inputs
    weights: [weights for each token],
    n_tokens: 3
}

// vs sending 3 separate requests
```

**Impact**:
- Reduces network round trips from N → 1 (for N tokens)
- Only applicable during **prompt processing** (parallel tokens)
- **Not applicable** during autoregressive generation (one token at a time)

**Note**: Naturally happens when you group experts by endpoint - no special logic needed!

### Strategy 4: Asynchronous Dispatch

```cpp
// SEQUENTIAL (bad):
for (auto& [endpoint, experts] : remote_experts) {
    result = call_rpc_sync(endpoint, experts);  // BLOCKS
    results.push_back(result);
}
// Total time = sum of all RPC times

// PARALLEL (good):
std::vector<std::future<Tensor>> futures;
for (auto& [endpoint, experts] : remote_experts) {
    futures.push_back(std::async([endpoint, experts]() {
        return call_rpc_sync(endpoint, experts);
    }));
}
// Wait for all
for (auto& f : futures) {
    results.push_back(f.get());
}
// Total time = max of all RPC times
```

**Impact**: If you have 3 workers, reduces total time from 3× to 1× (assuming parallel capacity).

### Strategy 5: Zero-Copy Where Possible

```cpp
// BAD: Multiple copies
Tensor input = get_input();
std::vector<float> buffer(input.size());
memcpy(buffer.data(), input.data(), input.size() * sizeof(float));  // Copy 1
std::string serialized = serialize(buffer);                          // Copy 2
socket.send(serialized);                                             // Copy 3

// GOOD: Direct buffer send
Tensor input = get_input();
socket.send_raw(input.data(), input.size() * sizeof(float));        // No copies
```

**Impact**: Saves ~2-5ms per request for large tensors.

### Strategy 6: Compression (Conditional)

**When to use**: Network bandwidth is the bottleneck (e.g., WAN).

```cpp
// For activations (already in float16 or lower precision)
if (network_is_slow) {
    // Quantize to int8 before sending
    std::vector<int8_t> quantized = quantize_f32_to_i8(input_data);
    send(quantized);  // 4x smaller
    // Worker dequantizes on receive
}
```

**Trade-off**:
- Reduces network time by 4×
- Adds quantization overhead (~1-2ms)
- Small accuracy loss (usually negligible)

**When it helps**: Network latency > 20ms per transfer.

### Strategy 7: Result Caching (Usually Not Practical)

**Idea**: Cache expert outputs based on input tensor.

**Reality**: Input tensors are different every forward pass, so cache hits are rare.

```cpp
// Would need: hash(input_tensor) + layer + expert as key
// But input changes every token, so almost never cache hits
```

**The few cases it might help:**
- Speculative decoding (evaluating multiple draft tokens)
- Extremely repetitive prompts (rare)

**Verdict**: ❌ Not recommended for typical use cases. Skip this optimization.

### Strategy 8: Predictive Prefetching (Advanced - For Later)

**Idea**: Predict which experts will be selected and start fetching before router completes.

```cpp
// Timeline with prefetching:
// │─ Predict (1ms) ─│───── Router (5ms) ──────│
//                   ↓
//                   └──── RPC starts (10ms) ────│
//                                              ↑ Router done 5ms earlier
// Overlap RPC with router execution!

// Simple version: Use previous layer's selection
if (layer > 0) {
    predicted = previous_layer_experts;
    start_async_fetch(predicted);  // Speculative
}
selected = run_router(input);  // Actual selection
```

**Considerations**:
- Needs high accuracy (>70%) to be worthwhile
- Wastes bandwidth on mispredictions
- Adjacent layers often select similar experts (60-80% overlap)

**Verdict**: ⚠️ Save for later - only matters if network latency >50ms.

### Complete Optimized RPC Flow

```cpp
ggml_tensor* evaluate_remote_experts_optimized(
    const std::string& endpoint,
    uint8_t layer_id,
    const std::vector<int>& expert_ids,
    ggml_tensor* input,
    ggml_tensor* weights
) {
    // 1. Get persistent connection
    Connection* conn = pool.get_connection(endpoint);

    // 2. Zero-copy prepare
    ExpertRequestHeader header = {
        .magic = 0x45585052,
        .layer_id = layer_id,
        .n_experts = (uint8_t)expert_ids.size(),
        .n_tokens = (uint16_t)input->ne[1],
        .n_embd = (uint32_t)input->ne[0]
    };

    // 3. Single send with iovec (scatter-gather I/O)
    struct iovec iov[4] = {
        {&header, sizeof(header)},
        {expert_ids.data(), expert_ids.size()},
        {weights->data, ggml_nbytes(weights)},
        {input->data, ggml_nbytes(input)}
    };
    conn->sendv(iov, 4);  // Single syscall

    // 4. Receive directly into output tensor (zero-copy)
    ggml_tensor* output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    conn->recv_into(output->data, ggml_nbytes(output));

    // 5. Return connection to pool
    pool.return_connection(endpoint, conn);

    return output;
}
```

### Latency Optimization Checklist

**Tier 1: Essential** (20-30ms savings):
- [x] Use binary protocol (protobuf or raw)
- [x] Connection pooling
- [x] Async dispatch for multiple endpoints
- [x] Zero-copy I/O

**Tier 2: Significant for Batch Processing** (10-20ms savings):
- [ ] Token batching (only for prompt processing with multiple tokens)
- [ ] Compression (only if network bandwidth is bottleneck)
- [ ] FP16/BF16 tensors instead of FP32

**Tier 3: Advanced/Research** (conditional benefits):
- [ ] Predictive prefetching (only if network latency >50ms)
- [ ] ~~Result caching~~ (not practical - skip this)
- [ ] ~~Batching across layers~~ (not possible due to dependencies)

### Measurement & Profiling

```cpp
// Add timing to find bottlenecks
struct RPCMetrics {
    std::chrono::microseconds serialize_time;
    std::chrono::microseconds network_time;
    std::chrono::microseconds deserialize_time;
    std::chrono::microseconds compute_time;
};

// Profile each request
auto start = std::chrono::high_resolution_clock::now();
serialize(request);
auto t1 = std::chrono::high_resolution_clock::now();
send_and_receive(request);
auto t2 = std::chrono::high_resolution_clock::now();
deserialize(response);
auto t3 = std::chrono::high_resolution_clock::now();

metrics.serialize_time = t1 - start;
metrics.network_time = t2 - t1;
metrics.deserialize_time = t3 - t2;

// Log and analyze
if (metrics.network_time > 10ms) {
    LOG_WARN("High network latency: %dms", metrics.network_time.count());
}
```

### Expected Performance

**With all optimizations**:
- Local network (1Gbps): 2-5ms RPC overhead
- Fast network (10Gbps): 1-2ms RPC overhead
- Slow network (100Mbps): 10-20ms RPC overhead

**Without optimizations**:
- Can be 5-10× worse (50-100ms)

### Key Takeaways

**For distributed MoE, latency is critical**:

✅ **Do these** (proven wins):
1. Binary protocol (protobuf or raw binary)
2. Connection pooling (reuse TCP connections)
3. Async dispatch (parallel requests to multiple endpoints)
4. Zero-copy I/O (direct buffer sends)
5. Token batching (automatically works during prompt processing)

❌ **Skip these** (not practical):
- Result caching (inputs always change)
- Batching across layers (layers depend on each other sequentially)

⚠️ **Consider later** (advanced):
- Predictive prefetching (complex, only helps with high latency)
- Compression (only if network is the bottleneck)

**Target**: With Tier 1 optimizations, RPC overhead should be 2-10ms per request, making distributed evaluation viable even for latency-sensitive applications.
