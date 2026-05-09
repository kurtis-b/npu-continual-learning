# Open Questions

These questions block implementation because the source spec intentionally defines project direction but leaves concrete engineering choices open.

## Hardware And Toolchain

- Which exact Ryzen AI NPU target should be supported first, and what are the required OS, kernel, XRT, IRON, MLIR-AIE, and compiler versions?
- Is there an existing local IRON/MLIR-AIE setup to reuse, or should the project include setup scripts from scratch?
- Which existing IRON operators or examples are the starting point for GEMM/GEMV and AXPY?
- Which NPU data types are available and stable for the target kernels?
- How should power or energy be measured: direct NPU telemetry, system-level power proxy, battery drain proxy, or omitted from v1?

## Repository And Runtime Choices

- Which Python packaging tool should be used: plain `venv`/`pip`, `uv`, Poetry, Conda, or an existing local convention?
- Which test runner and benchmark tooling should be standard: `pytest`, custom scripts, `asv`, or another harness?
- Should benchmark outputs be committed only as schemas/templates, or should small reference result files be committed too?
- What CI behavior is expected for hardware-gated NPU tests, since the NPU will not be available on most runners?

## Learning Workload

- What is the first benchmark stream: synthetic classification drift, CORe50 latent features, Avalanche scenario, or another controlled dataset?
- What are the initial tensor shapes for the first experiments: batch sizes, feature dimensions, and class counts?
- What feedback labels should the task ranker learn from: binary accept/reject, multiclass task class, pairwise preference, rating, or another signal?
- What replay policies are required in v1: FIFO, class-balanced, reservoir, hard-example, drift-triggered, or a smaller subset?
- What drift scenarios must be supported initially, and how should drift be detected or injected?

## Product Demo Scope

- What is the minimal task/context ranker input schema?
- What does a ranked item represent: task, document, project context, command suggestion, or another object?
- What local productivity graph data is allowed for the demo, and what must be synthetic?
- How should feedback be stored locally, and what retention or deletion behavior is required?
- What level of explanation should the ranker provide to users, if any?

## Metrics And Acceptance Targets

- What numerical tolerance is acceptable for each NPU operator compared with CPU reference?
- What minimum convergence behavior is required on the synthetic sanity dataset?
- What latency, memory, transfer-byte, or accuracy thresholds define success for each milestone?
- Should NPU-assisted training be required to outperform CPU for any shape, or is measured characterization sufficient?
- What result format is considered canonical: CSV, JSONL, SQLite, Parquet, or another format?

## Documentation And Release

- Should the project produce a technical report only, a runnable demo only, or both?
- Should public documentation include hardware setup instructions that may be specific to one laptop?
- What license and publication expectations apply if this becomes a public GitHub repository?
