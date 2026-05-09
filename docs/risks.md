# Risks

## Toolchain Friction

IRON/MLIR-AIE/XRT setup may be fragile and hardware-specific.

Mitigation:
- Start by reproducing existing AXPY or GEMM examples.
- Freeze and document hardware/software versions.
- Keep CPU-only tests runnable without the NPU.
- Avoid building project-specific complexity before the NPU smoke path works.

## NPU Path Slower Than CPU

Small online updates may be dominated by launch overhead and host/NPU transfers.

Mitigation:
- Treat slowdown as a valid research result if measured clearly.
- Separate kernel latency, transfer bytes, and end-to-end step latency.
- Sweep batch size, feature dimension, replay size, and dataflow variants.
- Try persistent weights, batched replay, fused update, and pipelining only after the naive baseline is correct.

## Unsupported Or Low-Value Operations

Softmax, loss, reductions, and simple elementwise operations may not be worth moving to the NPU initially.

Mitigation:
- Keep these operations on CPU until dense GEMM and update kernels are stable.
- Move or fuse them only if measurements show transfer or CPU overhead justifies it.
- Preserve a simple CPU reference path for every operation.

## Numerical Instability

Lower precision NPU kernels may change convergence behavior.

Mitigation:
- Use CPU float32 as the initial reference.
- Keep CPU master weights during early experiments if needed.
- Record absolute error, relative error, and convergence impact.
- Start with small learning rates and deterministic sanity datasets.

## Over-Scoping To Transformers

Full transformer or LLM fine-tuning would distract from the tractable dense-kernel research path.

Mitigation:
- Keep logistic regression and a small MLP as the primary implementation targets.
- Treat transformer or LoRA training as future work only after the small learners are proven.
- Make manual backprop and explicit kernel mapping the central deliverable.

## App Complexity

A full productivity agent could absorb effort without improving the kernel research.

Mitigation:
- Separate the benchmark/runtime work from the ranker demo.
- Use a synthetic or controlled stream before personal productivity data.
- Keep the final demo minimal: ranking plus explicit feedback.

## Privacy And Data Ambiguity

Real productivity data can create reproducibility, privacy, and retention problems.

Mitigation:
- Use public or synthetic data for research claims.
- Store compact features or latent representations rather than raw content where possible.
- Require an explicit data schema and retention policy before using real local data.

## Reproducibility Gaps

Performance and learning claims may be hard to validate without complete metadata.

Mitigation:
- Log configs, hardware/software versions, shapes, precision, replay policy, dataflow, timings, transfer bytes, and numerical error.
- Keep raw logs traceable to reports and plots.
- Use fixed seeds for correctness and small benchmark runs.

## Sparse Update Mapping Risk

Unstructured sparsity may not map efficiently onto the NPU.

Mitigation:
- Start with dense kernels and structured sparsity such as rows, columns, blocks, or low-rank deltas.
- Compare sparse update against dense update on both learning quality and data movement.
- Keep sparse graph traversal on CPU.

## Underspecified Decisions

Several implementation choices are not defined in the source spec, including package tooling, first dataset, feedback schema, result format, tolerances, and numeric success thresholds.

Mitigation:
- Track these in `docs/open-questions.md`.
- Do not encode major product behavior until these questions are answered.
- Prefer minimal, reversible defaults for early scaffolding.
