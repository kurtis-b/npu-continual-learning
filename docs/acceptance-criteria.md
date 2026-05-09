# Acceptance Criteria And Test Strategy

## Milestone Acceptance Criteria

### M0: Setup And Literature Grounding

- NPU toolchain setup is documented with exact hardware and software versions.
- At least one existing IRON/MLIR-AIE/XRT operator test runs successfully on the target machine.
- A CPU-only fallback development path is documented.
- Reference notes identify the design principles used by this project: explicit kernels, manual training graph, sparse update, latent replay, and IO-aware data movement.

### M1: CPU Continual-Learning Baseline

- Logistic regression trains end to end on a deterministic synthetic stream.
- Manual gradients pass finite-difference or independently computed gradient checks.
- Replay buffer behavior is deterministic under fixed seeds.
- Metrics include loss, current-task accuracy, old-task accuracy or forgetting, and replay memory.
- A reproducible CPU baseline run can be generated from a logged config.

### M2: NPU Forward GEMM

- NPU `Z = XW` matches the CPU reference within an explicitly documented tolerance.
- Tests cover small hand-checkable shapes and at least one realistic benchmark shape.
- Benchmark output separates host-to-NPU bytes, NPU-to-host bytes, kernel latency, and end-to-end step time.
- The implementation can fall back to CPU when NPU execution is unavailable.

### M3: NPU Backward GEMM

- NPU `dW = X^T dZ` matches CPU gradients within an explicitly documented tolerance.
- The NPU backward kernel is integrated into the logistic-regression training step.
- The full training loop still converges on the synthetic sanity dataset.
- Results compare CPU-only, forward-only NPU, and forward-plus-backward NPU configurations.

### M4: NPU Update And Fusion

- AXPY/update behavior matches the CPU update within the documented tolerance.
- End-to-end training remains numerically stable with the NPU update path enabled.
- Transfer accounting shows whether moving or fusing the update reduced round trips.
- Fused `dW + update` is attempted only if the separate kernels are correct and measurable.

### M5: Continual-Learning Evaluation

- Sweeps cover the agreed v1 values for batch size, feature dimension, class count, replay buffer size, precision, sparsity, and dataflow.
- Reports include accuracy, forgetting, adaptation speed, replay memory, transfer bytes, kernel latency, and end-to-end latency.
- Each plotted or summarized result is traceable to a config and raw log.
- The report identifies at least one regime where NPU assistance helps, or explains why transfer overhead dominates.

### M6: Small MLP Extension

- CPU MLP gradients pass deterministic gradient checks.
- MLP training converges on the sanity dataset before NPU integration.
- NPU-assisted MLP kernels match CPU references within documented tolerances.
- Store-vs-recompute measurements include memory and latency tradeoffs.

### M7: Graph-Aware Adaptive Ranker Demo

- A minimal ranking demo learns from explicit local feedback.
- Graph or structured context features are built on CPU.
- The demo can run with CPU-only and NPU-assisted dense training paths.
- Demo metrics include ranking quality or feedback acceptance plus training-step system metrics.
- No private background collection is required for the demo.

## Test Strategy

### CPU Unit Tests

- Test logistic-regression forward, softmax/loss, backward, update, and prediction behavior.
- Test MLP forward, activation backward, gradient paths, and update behavior after M6 begins.
- Test replay-buffer insertion, sampling, class balancing or reservoir behavior if selected, and deterministic seeding.
- Test feature builders on fixed input fixtures.
- Test metric computations, especially forgetting and adaptation-speed calculations.

### Gradient And Numerical Tests

- Use fixed seeds and small hand-checkable tensors before larger randomized shapes.
- Compare manual gradients against finite differences or an independent PyTorch/NumPy reference.
- Report absolute error, relative error, and downstream convergence impact.
- Track numerical drift by precision and shape.

### NPU Operator Tests

- Every NPU operator must have a CPU reference and golden tensor tests.
- Hardware-gated tests should skip cleanly when the NPU or toolchain is unavailable.
- Tests must validate both output values and shape/layout expectations.
- Initial operator tests should cover forward GEMM, backward GEMM, and AXPY independently before fused variants.

### Integration Tests

- Run the full CPU-only training loop on a tiny synthetic stream.
- Run NPU-assisted configurations incrementally: forward only, forward plus backward, then update.
- Verify the loss decreases or reaches a documented convergence threshold on the sanity dataset.
- Verify CPU fallback produces equivalent behavior when NPU execution is disabled.

### Benchmark And Reproducibility Tests

- Validate experiment config parsing and result-log schema.
- Ensure benchmark logs include hardware/software versions, tensor shapes, precision, replay policy, dataflow variant, timings, transfer bytes, and numerical error.
- Add smoke-sized benchmark tests that complete quickly and do not require large datasets.
- Keep large sweeps separate from routine unit tests.

## Underspecified Acceptance Inputs

- Exact numerical tolerances are not specified.
- Exact convergence thresholds are not specified.
- Exact benchmark shape lists are suggested by the source spec but still need v1 selection.
- Exact performance targets are not specified.
- Exact power or energy measurement method is not specified.
