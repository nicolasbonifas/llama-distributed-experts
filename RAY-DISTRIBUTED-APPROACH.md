# Distributed MoE with Ray Framework

## Overview

Ray is a distributed computing framework that makes it easy to scale Python applications. For distributed MoE, we'd use Ray to orchestrate multiple llama.cpp worker processes across machines.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Master Process (Python)                                 │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Ray Driver                                         │ │
│  │  - Coordinates workers                              │ │
│  │  - Manages routing logic                            │ │
│  │  - Handles load balancing                           │ │
│  └────────────────────────────────────────────────────┘ │
│                          │                               │
│                          ↓                               │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Inference Engine                                   │ │
│  │  - Run router on CPU (lightweight)                  │ │
│  │  - Get expert IDs for each token                    │ │
│  │  - Dispatch to appropriate workers                  │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ↓                 ↓                 ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Ray Worker 1 │  │ Ray Worker 2 │  │ Ray Worker 3 │
│              │  │              │  │              │
│ llama.cpp    │  │ llama.cpp    │  │ llama.cpp    │
│ Experts 0-31 │  │ Experts32-63 │  │ Experts64-95 │
│ GPU 0        │  │ GPU 1        │  │ GPU 2        │
└──────────────┘  └──────────────┘  └──────────────┘
```

## Implementation Example

```python
import ray
import numpy as np
from llama_cpp import Llama
from typing import List, Dict

# Initialize Ray cluster
ray.init(address='auto')  # Connect to existing cluster or start new

@ray.remote(num_gpus=1)  # Each worker gets 1 GPU
class ExpertWorker:
    """Worker that loads and evaluates a subset of experts"""

    def __init__(self,
                 model_path: str,
                 expert_range: str,
                 gpu_id: int):
        """
        Args:
            model_path: Path to GGUF model file
            expert_range: "0-31" or "32-63" etc
            gpu_id: Which GPU to use
        """
        self.expert_range = expert_range
        self.gpu_id = gpu_id

        # Load llama.cpp model with only assigned experts
        # (Would need to add selective loading to llama.cpp)
        self.model = Llama(
            model_path=model_path,
            n_gpu_layers=-1,  # All on GPU
            main_gpu=gpu_id,
            # Hypothetical future parameter:
            expert_range=expert_range
        )

        self.expert_ids = self._parse_range(expert_range)
        print(f"Worker initialized with experts {self.expert_ids} on GPU {gpu_id}")

    def _parse_range(self, range_str: str) -> List[int]:
        """Parse '0-31' into [0,1,2,...,31]"""
        start, end = map(int, range_str.split('-'))
        return list(range(start, end + 1))

    def evaluate_experts(self,
                        layer_id: int,
                        expert_ids: List[int],
                        input_tensor: np.ndarray,
                        weights: np.ndarray) -> np.ndarray:
        """
        Evaluate assigned experts for given input

        Args:
            layer_id: Which layer (0-31 for 32-layer model)
            expert_ids: Which experts to evaluate (subset of self.expert_ids)
            input_tensor: [n_embd, n_tokens] activation tensor
            weights: [n_experts_requested, n_tokens] router weights

        Returns:
            weighted_output: [n_embd, n_tokens] aggregated expert outputs
        """
        # Filter to only experts we have
        local_expert_ids = [eid for eid in expert_ids if eid in self.expert_ids]

        if not local_expert_ids:
            # This worker doesn't have any of the requested experts
            return None

        # Call into llama.cpp to evaluate experts
        # (Would need to expose this API from llama.cpp)
        output = self.model.evaluate_layer_experts(
            layer_id=layer_id,
            expert_ids=local_expert_ids,
            input_tensor=input_tensor,
            weights=weights
        )

        return output

    def get_expert_ids(self) -> List[int]:
        """Return which experts this worker handles"""
        return self.expert_ids

    def health_check(self) -> bool:
        """Check if worker is healthy"""
        return True


class DistributedMoEEngine:
    """Master coordinator for distributed MoE inference"""

    def __init__(self,
                 model_path: str,
                 worker_configs: List[Dict]):
        """
        Args:
            model_path: Path to GGUF model
            worker_configs: List of {expert_range, gpu_id} dicts
                Example: [
                    {"expert_range": "0-31", "gpu_id": 0},
                    {"expert_range": "32-63", "gpu_id": 1},
                    {"expert_range": "64-95", "gpu_id": 2},
                ]
        """
        self.model_path = model_path

        # Create workers
        self.workers = [
            ExpertWorker.remote(
                model_path=model_path,
                expert_range=cfg["expert_range"],
                gpu_id=cfg["gpu_id"]
            )
            for cfg in worker_configs
        ]

        # Build expert-to-worker mapping
        self.expert_map = {}  # expert_id -> list of worker indices
        for worker_idx, worker in enumerate(self.workers):
            expert_ids = ray.get(worker.get_expert_ids.remote())
            for eid in expert_ids:
                if eid not in self.expert_map:
                    self.expert_map[eid] = []
                self.expert_map[eid].append(worker_idx)

        print(f"Initialized {len(self.workers)} workers")
        print(f"Expert map: {self.expert_map}")

        # Load router model (lightweight, CPU-only)
        self.router_model = Llama(
            model_path=model_path,
            n_gpu_layers=0,  # CPU only
            # Only load router weights, not experts (hypothetical)
            load_mode="router_only"
        )

    def generate(self, prompt: str, max_tokens: int = 100):
        """Generate text using distributed MoE"""

        # Tokenize
        tokens = self.router_model.tokenize(prompt)

        for _ in range(max_tokens):
            # Run forward pass with distributed experts
            logits = self._forward_pass(tokens)

            # Sample next token
            next_token = self._sample(logits)
            tokens.append(next_token)

            if next_token == self.router_model.token_eos():
                break

        return self.router_model.detokenize(tokens)

    def _forward_pass(self, tokens: List[int]) -> np.ndarray:
        """
        Run forward pass with distributed expert evaluation

        This is where the magic happens!
        """
        # Get embeddings and run attention layers (on CPU/local)
        hidden_states = self.router_model.get_embeddings(tokens)

        # For each layer
        for layer_id in range(self.router_model.n_layers()):

            if self.router_model.is_moe_layer(layer_id):
                # MoE layer - distribute expert evaluation
                hidden_states = self._evaluate_moe_layer(
                    layer_id,
                    hidden_states
                )
            else:
                # Regular layer - run locally
                hidden_states = self.router_model.evaluate_layer(
                    layer_id,
                    hidden_states
                )

        # Final layer norm and logits
        logits = self.router_model.get_logits(hidden_states)
        return logits

    def _evaluate_moe_layer(self,
                           layer_id: int,
                           input_tensor: np.ndarray) -> np.ndarray:
        """
        Evaluate MoE layer using distributed workers

        This is the key distributed MoE logic
        """
        n_tokens = input_tensor.shape[1]
        n_embd = input_tensor.shape[0]

        # 1. Run router to get expert selections
        router_logits = self.router_model.run_router(layer_id, input_tensor)

        # 2. Select top-k experts per token
        k = self.router_model.n_expert_used()
        selected_experts, weights = self._select_top_k(router_logits, k)
        # selected_experts: [k, n_tokens] - expert IDs
        # weights: [k, n_tokens] - normalized weights

        # 3. Group expert requests by worker
        worker_requests = self._group_by_worker(
            selected_experts,
            weights,
            n_tokens
        )

        # 4. Dispatch to workers IN PARALLEL (Ray magic!)
        futures = []
        for worker_idx, request in worker_requests.items():
            future = self.workers[worker_idx].evaluate_experts.remote(
                layer_id=layer_id,
                expert_ids=request['expert_ids'],
                input_tensor=input_tensor,
                weights=request['weights']
            )
            futures.append(future)

        # 5. Wait for all workers to complete and gather results
        results = ray.get(futures)  # Blocks until all complete

        # 6. Aggregate results from all workers
        output = np.sum([r for r in results if r is not None], axis=0)

        return output

    def _group_by_worker(self,
                        selected_experts: np.ndarray,
                        weights: np.ndarray,
                        n_tokens: int) -> Dict:
        """
        Group expert evaluation requests by worker

        Input:
            selected_experts: [k, n_tokens] - which experts selected per token
            weights: [k, n_tokens] - router weights

        Output:
            {worker_idx: {expert_ids: [...], weights: [...]}}
        """
        worker_requests = {}

        # For each token
        for token_idx in range(n_tokens):
            # For each selected expert for this token
            for k_idx in range(selected_experts.shape[0]):
                expert_id = selected_experts[k_idx, token_idx]
                weight = weights[k_idx, token_idx]

                # Find which worker has this expert
                worker_indices = self.expert_map.get(expert_id, [])

                if not worker_indices:
                    raise ValueError(f"No worker has expert {expert_id}")

                # Choose worker (simple: first one, could be load-balanced)
                worker_idx = worker_indices[0]

                # Add to request
                if worker_idx not in worker_requests:
                    worker_requests[worker_idx] = {
                        'expert_ids': [],
                        'weights': []
                    }

                worker_requests[worker_idx]['expert_ids'].append(expert_id)
                worker_requests[worker_idx]['weights'].append(weight)

        return worker_requests

    def _select_top_k(self, logits: np.ndarray, k: int):
        """Select top-k experts and compute softmax weights"""
        # logits: [n_experts, n_tokens]

        # Get top-k indices per token
        top_k_indices = np.argsort(logits, axis=0)[-k:]  # [k, n_tokens]

        # Get corresponding logits
        top_k_logits = np.take_along_axis(logits, top_k_indices, axis=0)

        # Softmax over top-k
        exp_logits = np.exp(top_k_logits - np.max(top_k_logits, axis=0))
        weights = exp_logits / np.sum(exp_logits, axis=0)

        return top_k_indices, weights

    def _sample(self, logits: np.ndarray) -> int:
        """Sample next token from logits"""
        # Simple argmax sampling
        return int(np.argmax(logits))


# Usage example
if __name__ == "__main__":
    # Define worker configuration
    # Let's split 128 experts across 4 workers
    worker_configs = [
        {"expert_range": "0-31", "gpu_id": 0},
        {"expert_range": "32-63", "gpu_id": 1},
        {"expert_range": "64-95", "gpu_id": 2},
        {"expert_range": "96-127", "gpu_id": 3},
    ]

    # Create distributed engine
    engine = DistributedMoEEngine(
        model_path="/path/to/qwen3-30b.gguf",
        worker_configs=worker_configs
    )

    # Generate text
    output = engine.generate("Once upon a time", max_tokens=100)
    print(output)
```

## Key Advantages of Ray Approach

### 1. **Automatic Parallelism**
```python
# These calls happen in parallel automatically!
futures = [worker.evaluate.remote(...) for worker in workers]
results = ray.get(futures)  # Wait for all to complete
```

Ray handles:
- Thread/process management
- Network communication
- Result gathering
- Error handling

### 2. **Fault Tolerance**
```python
@ray.remote(max_retries=3)  # Auto-retry on failure
class ExpertWorker:
    ...
```

If a worker dies:
- Ray detects failure
- Automatically retries on another node
- No manual error handling needed

### 3. **Dynamic Scaling**
```python
# Can add/remove workers at runtime
new_worker = ExpertWorker.remote(model_path, "128-159", gpu_id=4)
engine.workers.append(new_worker)
```

### 4. **Resource Management**
```python
@ray.remote(
    num_gpus=1,           # Reserve 1 GPU
    num_cpus=4,           # Reserve 4 CPU cores
    memory=16*1024**3,    # Reserve 16GB RAM
)
class ExpertWorker:
    ...
```

Ray ensures resources are available before scheduling.

### 5. **Built-in Monitoring**
```python
# Ray dashboard shows:
# - Worker CPU/GPU utilization
# - Network bandwidth
# - Task latency
# - Memory usage
```

Access at http://localhost:8265

## Comparison with Pure llama.cpp Approach

| Aspect | Ray Approach | Pure llama.cpp |
|--------|-------------|----------------|
| **Implementation** | Python wrapper | C++ modifications |
| **Complexity** | ~500 lines Python | ~1000+ lines C++ |
| **Networking** | Ray handles it | Manual TCP sockets |
| **Fault tolerance** | Built-in retries | Manual implementation |
| **Load balancing** | Ray scheduler | Manual balancing |
| **Monitoring** | Ray dashboard | Custom logging |
| **Latency** | +5-10ms Python overhead | Pure C++ (faster) |
| **Flexibility** | Easy to modify | Hard to modify |
| **Deployment** | Requires Ray cluster | Standalone binaries |

## Limitations of Ray Approach

### 1. **Python Overhead**
Every RPC call goes through Python:
```
Python → Ray → Network → Ray → Python → llama.cpp
```
Adds ~5-10ms latency per call.

### 2. **Serialization Overhead**
NumPy arrays must be serialized:
```python
# This copies data multiple times:
input_tensor = np.array(...)  # Copy 1: C++ to NumPy
ray.put(input_tensor)          # Copy 2: NumPy to Ray
# ... send over network ...     # Copy 3: Ray to network
ray.get(future)                # Copy 4: Network to Ray
result = np.array(...)         # Copy 5: Ray to NumPy
```

Pure C++ approach has zero-copy potential.

### 3. **Dependencies**
Requires:
- Python runtime
- Ray framework
- NumPy
- llama-cpp-python bindings

Pure llama.cpp is self-contained.

### 4. **Not CLI-Friendly**
Can't just run:
```bash
./llama-cli --model model.gguf --expert-rpc-servers ...
```

Must write Python orchestration code.

## When to Use Ray

**Use Ray if:**
- ✅ Building inference service (not CLI tool)
- ✅ Need fault tolerance and auto-scaling
- ✅ Python ecosystem is acceptable
- ✅ 10-20ms extra latency is OK
- ✅ Want rapid prototyping

**Use pure llama.cpp if:**
- ✅ Need lowest possible latency
- ✅ Building CLI tool or embedded system
- ✅ Want standalone binaries
- ✅ Python overhead unacceptable
- ✅ Contributing to llama.cpp project

## Hybrid Approach

Best of both worlds:

```python
# Use Ray for orchestration
@ray.remote
class ExpertWorker:
    def __init__(self):
        # Start llama.cpp expert-server as subprocess
        self.process = subprocess.Popen([
            './expert-server',
            '--model', 'model.gguf',
            '--expert-range', '0-31',
            '--port', '50052'
        ])

    def evaluate(self, ...):
        # Direct TCP call to expert-server (bypass Ray for data)
        return tcp_rpc_call('localhost:50052', ...)
```

This way:
- Ray handles scheduling/fault-tolerance
- Data flows through direct TCP (faster)
- Get benefits of both approaches

## Conclusion

Ray is excellent for:
- Research and prototyping
- Production inference services
- Complex orchestration needs

But for the llama.cpp project itself, the pure C++ approach is more appropriate because:
- Keeps llama.cpp self-contained
- Enables CLI usage
- Minimizes latency
- Fits project philosophy

The work we did (TCP RPC layer, expert routing) could be used by **either** approach!
