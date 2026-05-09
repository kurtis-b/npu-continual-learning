# Roadmap

## M0: Setup And Literature Grounding

- Establish the development environment for IRON/MLIR-AIE/XRT.
- Reproduce at least one existing NPU operator smoke test, preferably AXPY or GEMM.
- Record hardware, OS/kernel, driver, XRT, IRON, and compiler versions.
- Summarize the relevant design principles from the source spec references.

## M1: CPU Continual-Learning Baseline

- Implement deterministic CPU-only online logistic regression.
- Add manual forward, backward, loss, update, metrics, and gradient checks.
- Add a replay buffer with the smallest selected v1 policy set.
- Create a synthetic or benchmark-backed stream with controlled drift.
- Produce the first reproducible CPU baseline report.

## M2: NPU Forward GEMM

- Implement or wrap the forward dense kernel for `Z = XW`.
- Add golden tensor tests against the CPU reference.
- Add benchmark instrumentation for kernel latency, transfer bytes, and end-to-end step time.
- Compare CPU-only inference/training-step forward time with the NPU-assisted path.

## M3: NPU Backward GEMM

- Implement or wrap the gradient dense kernel for `dW = X^T dZ`.
- Validate output against CPU gradients across small deterministic shapes.
- Integrate the operator into the logistic-regression training step.
- Produce the first NPU-assisted backprop benchmark.

## M4: NPU Update And Fusion

- Implement or wrap AXPY-style update for `W = W - lr * dW`.
- Compare CPU update, NPU update, and transfer costs.
- Attempt fused `dW + update` only after separate backward and update kernels are correct.
- Evaluate whether round-trip reduction improves end-to-end latency.

## M5: Continual-Learning Evaluation

- Run replay, batch-size, feature-dimension, class-count, precision, sparsity, and dataflow sweeps.
- Measure accuracy, forgetting, adaptation speed, replay memory, transfer bytes, and latency.
- Identify the shapes and replay regimes where NPU assistance is beneficial or clearly transfer-dominated.
- Produce a tradeoff report rather than optimizing for a single metric.

## M6: Small MLP Extension

- Add a single-hidden-layer CPU MLP reference.
- Add activation backward and the second gradient GEMM path.
- Validate convergence and gradient correctness on a small sanity dataset.
- Test activation store-vs-recompute behavior and multi-kernel scheduling costs.

## M7: Graph-Aware Adaptive Ranker Demo

- Define a minimal local task/context item schema.
- Add CPU-side graph or structured context features.
- Train the small head from explicit feedback.
- Keep dense learned components eligible for NPU acceleration.
- Demonstrate ranking adaptation under a controlled local feedback stream.

## Reviewable Task Sizing

- Each milestone should land as multiple small changes: one module or operator at a time, with tests and docs in the same review.
- NPU operators should be reviewed independently before being integrated into end-to-end training.
- Benchmark schema changes should be reviewed before large result files are generated.
- The application demo should not be started until the CPU and NPU training paths have stable acceptance criteria.
