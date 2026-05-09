# Requirements

## Project Intent

Build a hands-on local continual-learning system for AMD Ryzen AI laptops that uses custom NPU execution for selected dense forward, backward, and update kernels. The project explicitly avoids ONNX Runtime so that kernel structure, data movement, buffering, tiling, and optimization choices remain visible and controllable.

The first complete learner is online logistic regression, followed by a single-hidden-layer MLP head. The final demo target is an adaptive local task or context ranker that learns from feedback while keeping orchestration, graph handling, replay management, and evaluation on the CPU.

## Functional Requirements

- Provide a CPU correctness path using NumPy or PyTorch for every learner, operator, and training step.
- Implement manual backpropagation for online logistic regression before adding an MLP head.
- Support replay-based continual learning with configurable replay policies and metrics for adaptation and forgetting.
- Implement or wrap custom IRON/MLIR-AIE/XRT NPU operators for selected dense kernels:
  - forward GEMM/GEMV: `Z = XW`
  - backward GEMM: `dW = X^T dZ`
  - AXPY/update: `W = W - lr * dW`
  - optional fused train-step variants after the individual kernels are correct
- Keep softmax, loss, simple reductions, replay-buffer management, labels, validation, sparse graph operations, and application orchestration on the CPU until measurements justify moving them.
- Add a benchmark and experiment harness that compares CPU-only and NPU-assisted training.
- Record enough experiment metadata to reproduce each claim: hardware, OS/kernel, driver/XRT/IRON versions, tensor shapes, precision, replay policy, dataflow variant, timings, transfer bytes, numerical error, and learning metrics.
- Add a user-facing adaptive task/context ranker only after the kernel and continual-learning experiments are stable.

## Research Requirements

- Determine which parts of a small online training loop are worth moving to the Ryzen AI NPU.
- Measure whether host/NPU data movement dominates training-step latency.
- Compare naive transfer-heavy NPU execution against optimized variants such as persistent weights, batched replay, fused update, pipelined chunks, sparse/partial update, and recompute-vs-store.
- Evaluate batch size, feature dimension, class count, replay buffer size, update sparsity, precision, and dataflow choices.
- Report learning quality and system cost together, not separately.
- Treat NPU slowdown relative to CPU as a valid research result if measurements are reproducible and explain where the overhead comes from.

## Nonfunctional Requirements

- Deterministic CPU reference tests must exist before each NPU operator is considered valid.
- NPU kernels must be validated against CPU references with absolute error, relative error, and downstream convergence checks.
- The system must remain runnable in CPU-only mode for development and baseline testing.
- Public or synthetic data must be used for research claims before any personal-productivity data is introduced.
- Benchmark results must separate kernel time, transfer time or bytes, and end-to-end training-step latency.
- Documentation must clearly separate implemented behavior from planned variants and speculative future work.

## Out Of Scope For Initial Implementation

- ONNX Runtime, Vitis AI EP, automatic graph partitioning, or full PyTorch autograd on the NPU.
- Full transformer, Gemma, GPT, or LoRA training.
- Large benchmark-suite reproduction.
- Federated learning or privacy-preserving cryptography.
- Full OS automation, invasive monitoring, or background collection of private content.
- Using cloud GPUs or dedicated training accelerators as the main target.

## Explicit Assumptions

- A Ryzen AI laptop with AMD NPU access is available for hardware milestones.
- The CPU remains the source of truth for correctness, orchestration, replay policy, feedback handling, sparse graph traversal, and fallback.
- NPU work starts with dense matrix kernels because the target hardware and IRON workflow are better suited to explicit tiled dense computation than irregular graph traversal.
- Online logistic regression is the first end-to-end training loop; the MLP and graph-aware ranker are later extensions.
- CPU float32 behavior is the initial numerical reference even if NPU kernels use bfloat16 or another supported lower precision.
- A CPU master copy of trainable weights may be kept during early experiments if needed for stability and validation.
- The initial NPU implementation may be slower than CPU because transfer overhead and launch overhead are part of the research question.
- Replay starts with compact feature vectors or latent representations, not raw private user content.
- The final task-ranker demo is a local proof of concept, not a full productivity agent.
- Exact package tooling, hardware versions, benchmark datasets, feedback schema, persistence model, and numeric performance targets are not specified in the source spec and must be decided before implementation.
