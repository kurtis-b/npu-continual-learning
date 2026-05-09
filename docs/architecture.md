# Architecture

## Overview

The system is a small continual-learning runtime with a CPU correctness path and a custom NPU execution path. The CPU path owns correctness, orchestration, replay, sparse graph logic, validation, and fallback. The NPU path accelerates selected dense kernels through IRON/MLIR-AIE/XRT without ONNX Runtime.

```text
feedback or benchmark stream
  -> CPU feature builder
  -> replay buffer and mini-batch assembler
  -> CPU reference learner
  -> NPU operator backend
  -> evaluation and experiment harness
  -> optional adaptive task/context ranker demo
```

## Main Components

### CPU Feature Builder

- Converts benchmark examples or local demo events into fixed-size feature vectors.
- Supports text hashing, numeric features, latent features, and later graph-derived features.
- Keeps sparse graph traversal and irregular feature assembly on CPU.
- Produces mini-batches for current examples plus replay examples.

### Replay Buffer

- Stores compact continual-learning examples:
  - feature vector or latent representation
  - label or feedback signal
  - timestamp
  - task or stream identifier
  - optional metadata such as source or graph node id
- Starts with simple policies and expands only when needed for experiments.
- Tracks replay memory bytes and replay hit behavior.

### CPU Reference Learner

- Implements online logistic regression first.
- Adds a single-hidden-layer MLP after the linear learner is correct.
- Provides deterministic forward, backward, update, loss, metrics, and gradient checks.
- Serves as the correctness oracle for all NPU operators and training loops.

### NPU Operator Backend

- Uses IRON/MLIR-AIE/XRT for custom operator execution.
- Starts with separate operators:
  - forward GEMM/GEMV
  - backward GEMM
  - AXPY/update
- Later explores fused or persistent variants:
  - persistent weights
  - batched replay
  - fused `dW + update`
  - double-buffered or pipelined chunks
  - sparse or partial updates
  - recompute-vs-store for MLP activations
- Exposes transfer accounting and kernel timing to the benchmark harness.

### Experiment Harness

- Runs CPU-only and NPU-assisted configurations from logged experiment metadata.
- Separates CPU baseline latency, NPU kernel latency, transfer bytes, and end-to-end step latency.
- Reports learning quality, continual-learning behavior, numerical error, memory footprint, and system cost.
- Produces reproducible raw logs and summary plots or reports.

### Adaptive Ranker Demo

- Demonstrates local continual learning using a controlled task or context stream.
- Uses CPU-side features and graph context with dense learned scoring components.
- Collects explicit local feedback rather than invasive background monitoring.
- Remains downstream of the kernel research path and should not block earlier milestones.

## Initial Training Dataflow

Naive NPU-assisted logistic-regression step:

```text
CPU -> NPU: X, W
NPU:       Z = XW
NPU -> CPU: Z
CPU:       softmax, loss, dZ
CPU -> NPU: X, dZ
NPU:       dW = X^T dZ
NPU -> CPU: dW
CPU:       W = W - lr * dW
```

This baseline is expected to expose transfer overhead and may be slower than CPU. It exists to establish correctness and measurement discipline before optimized dataflows are attempted.

## Logistic Regression Kernel Mapping

```text
Inputs:
  X: [B, D]
  y: [B]
  W: [D, C]
  b: [C]

Forward:
  Z = XW + b
  P = softmax(Z)
  L = cross_entropy(P, y)

Backward:
  dZ = P - one_hot(y)
  dW = X^T dZ / B
  db = sum(dZ, axis=0) / B

Update:
  W = W - lr * dW
  b = b - lr * db
```

Initial target placement:

| Operation | Initial Placement | Reason |
| --- | --- | --- |
| `Z = XW` | NPU | First dense forward kernel target. |
| `Z += b` | CPU | Move later only if fusion is useful. |
| softmax/loss | CPU | Avoid early transfer/fusion complexity. |
| `dZ = P - Y` | CPU | Simple elementwise path; fusion candidate later. |
| `dW = X^T dZ` | NPU | Core backprop GEMM target. |
| `db = sum(dZ)` | CPU | Move only if fused. |
| `W -= lr * dW` | CPU first, NPU later | AXPY/update kernel target after GEMMs. |

## MLP Extension

After logistic regression is stable, add one hidden layer:

```text
H_pre = XW1 + b1
H = activation(H_pre)
Z = HW2 + b2
P = softmax(Z)
```

The MLP introduces activation backward, two gradient GEMMs, and an explicit decision between storing activations and recomputing them during backward. ReLU should be considered before GELU if implementation complexity blocks early progress.

## Underspecified Architecture Items

- Exact package layout, module names, package manager, and test runner are not specified.
- Exact NPU operator implementation style depends on the available IRON examples and target hardware.
- Exact benchmark stream, feedback schema, and app demo data model are not specified.
- Exact result storage format is not specified.
