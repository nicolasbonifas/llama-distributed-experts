# Distributed MoE Experts - Quick Start Guide

## What's Been Implemented

I've created the foundational infrastructure for distributed MoE expert evaluation:

### ✅ Complete:
1. **Command-line interface** - Arguments for worker and master modes
2. **Expert range parsing** - Utilities to parse "0-31" format
3. **RPC protocol header** - Data structures for expert RPC
4. **Expert worker server** - Executable that can start in worker mode
5. **Build system integration** - All files added to CMake

### 🚧 Stubbed (compiles but doesn't work yet):
- RPC protocol implementation (returns zeros)
- Network communication
- Expert evaluation on workers
- Master-side expert dispatching

## How to Build

```bash
cd /home/user/llama-distributed-experts
mkdir -p build && cd build
cmake .. -DGGML_RPC=ON
make -j$(nproc)
```

This will create:
- `./build/bin/expert-server` - Worker node
- `./build/bin/llama-cli` - Master node (with expert RPC support)

## Testing What Works

### Test 1: Verify Command-Line Arguments

```bash
# Should show the new arguments
./build/bin/llama-cli --help | grep expert

# Should see:
# --expert-worker-mode
# --expert-rpc-servers
```

### Test 2: Start Worker (Stub Mode)

```bash
./build/bin/expert-server \
    --model /path/to/qwen3-30b.gguf \
    --expert-worker-mode "0-15" \
    --port 50052

# Should print:
# "Expert worker initialized successfully"
# "Listening on 0.0.0.0:50052..."
```

### Test 3: Verify Expert Range Parsing

The parsing utilities are compiled into libcommon. You can test them by:
```cpp
#include "common.h"
auto ids = parse_expert_range("0-15,20,25-27");
// Should give: {0,1,2,...,15,20,25,26,27}
```

## What Doesn't Work Yet

### Critical Missing Pieces:

#### 1. **build_moe_ffn() Modification** (`src/llama-graph.cpp:936-1203`)

   **Current**: Evaluates all experts locally

   **Needed**:
   ```cpp
   // After expert selection (line ~1051)
   for (each selected expert) {
       if (expert is on remote node) {
           call ggml_rpc_expert_evaluate()
       } else {
           evaluate locally (existing code)
       }
   }
   merge local and remote results
   ```

#### 2. **Expert Routing** (`src/llama-model.h`)

   Add to `llama_model`:
   ```cpp
   struct {
       std::string endpoint;
       std::set<int> expert_ids;
   } expert_endpoints[];

   std::string find_expert_endpoint(int expert_id, int layer);
   ```

#### 3. **Actual RPC Communication** (`ggml/src/ggml-rpc/ggml-rpc-expert.cpp`)

   Replace stubs with:
   - Socket communication or use existing RPC framework
   - Serialize request/response structures
   - Send/receive over network

#### 4. **Worker-Side Expert Evaluation**

   When worker receives request:
   - Extract expert tensors for requested IDs
   - Run FFN computation
   - Apply weights and aggregate
   - Send result back

## Next Steps (In Order)

### Step 1: Test Compilation
```bash
cd build && make -j
# Should compile without errors
```

### Step 2: Add Expert Routing to llama_model

Edit `src/llama.cpp` in `llama_model_load()`:
```cpp
// After model loading, parse expert_rpc_servers
if (!params.expert_rpc_servers.empty()) {
    model.has_remote_experts = true;
    // Parse "host:port:range,..." format
    // Populate model.expert_endpoints
}
```

### Step 3: Modify build_moe_ffn()

This is the core change. You need to:
1. Check if `model.has_remote_experts`
2. For each selected expert, determine if local or remote
3. Dispatch RPC calls for remote experts
4. Merge results

See pseudocode in IMPLEMENTATION-STATUS.md

### Step 4: Implement Real RPC

Option A: Use existing ggml-rpc infrastructure
- Look at how `ggml/src/ggml-rpc/ggml-rpc.cpp` does it
- Reuse the socket communication layer

Option B: Simple TCP sockets
- Easier to understand
- More control
- More work

### Step 5: Worker-Side Evaluation

In `ggml-rpc-expert.cpp`:
```cpp
// Receive request from network
// Load model if not already loaded
// Extract expert slices from merged tensors
// Run expert FFN
// Send results back
```

## Testing Strategy

### Phase 1: Local Testing (Single Machine)
```bash
# Terminal 1: Start worker
./expert-server -m model.gguf --expert-worker-mode "0-3" -p 50052

# Terminal 2: Run master
./llama-cli -m model.gguf \
    --expert-rpc-servers "localhost:50052:0-3" \
    -p "Hello"
```

### Phase 2: Distributed Testing (Two Machines)
```bash
# Machine 1: Worker (experts 0-15)
./expert-server -m model.gguf --expert-worker-mode "0-15" -p 50052

# Machine 2: Worker (experts 16-31)
./expert-server -m model.gguf --expert-worker-mode "16-31" -p 50052

# Machine 3: Master
./llama-cli -m model.gguf \
    --expert-rpc-servers "192.168.1.10:50052:0-15,192.168.1.11:50052:16-31" \
    -p "Test prompt"
```

## Debugging Tips

### Enable Verbose Logging
```cpp
// In ggml-rpc-expert.cpp
#define EXPERT_RPC_DEBUG 1
fprintf(stderr, "DEBUG: ...");
```

### Check Expert Selection
```cpp
// In build_moe_ffn(), after line 1038
fprintf(stderr, "Selected experts for layer %d: ", il);
for (int i = 0; i < n_expert_used; ++i) {
    fprintf(stderr, "%d ", selected_experts->data[i]);
}
fprintf(stderr, "\n");
```

### Verify Network Communication
```bash
# Check if worker is listening
netstat -an | grep 50052

# Test basic connectivity
telnet localhost 50052
```

## Key Files Reference

| File | Purpose | Status |
|------|---------|--------|
| `common/common-expert.cpp` | Expert range parsing | ✅ Complete |
| `ggml/include/ggml-rpc-expert.h` | RPC protocol | ✅ Defined |
| `ggml/src/ggml-rpc/ggml-rpc-expert.cpp` | RPC implementation | ⚠️ Stub only |
| `tools/rpc/expert-server.cpp` | Worker executable | ⚠️ Stub only |
| `src/llama-graph.cpp` | MoE evaluation | ❌ Not modified |
| `src/llama-model.h` | Expert routing | ❌ Not added |

## Learning Resources

To understand the existing code better:

1. **MoE Implementation**: Read `src/models/qwen3moe.cpp` to see how MoE is used
2. **Existing RPC**: Look at `ggml/src/ggml-rpc/ggml-rpc.cpp` for network code
3. **Tensor Operations**: Study `src/llama-graph.cpp` for ggml tensor API

## Common Issues

### Issue: Compilation Errors
- **Solution**: Make sure `cmake .. -DGGML_RPC=ON` is set
- Check that all files are added to CMakeLists.txt

### Issue: Expert tensors are merged
- **Reality**: Current architecture stores all experts in one big tensor
- **Impact**: Can't selectively load individual experts
- **Workaround**: Load all, distribute at inference time

### Issue: Network protocol undefined
- **Status**: Intentionally left as TODO
- **Next**: Implement based on existing ggml-rpc pattern

## Questions to Explore

As you implement:
1. How are expert tensors laid out in memory?
2. Can we slice merged expert tensors efficiently?
3. What's the overhead of RPC calls vs local evaluation?
4. How to handle worker failures gracefully?
5. Is there a better way to organize expert storage for distribution?

## Success Criteria

You'll know it's working when:
1. ✅ Code compiles
2. ✅ Worker starts and shows assigned experts
3. ⏳ Master detects remote experts in config
4. ⏳ Master dispatches RPC calls to workers
5. ⏳ Workers evaluate experts and return results
6. ⏳ Model generates correct output with distributed experts

Good luck with the implementation! Start with Step 1 (test compilation) and work through each piece methodically.
