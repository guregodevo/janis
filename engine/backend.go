// Package engine defines the backend-agnostic tensor interface that model code
// is written against. The concrete kernel library (MLX-C today) lives behind
// this interface as a swappable implementation, so replacing or adding a
// backend never touches the model forward passes.
//
// The model is graph-building, not eager: op methods RECORD nodes and return
// handles; nothing computes until Eval/Floats forces it. That keeps the
// interface indirection free of per-op cost — the heavy work batches at eval.
package engine

// DType is the element type of a Tensor.
type DType int

const (
	F32 DType = iota
	F16
	BF16
	I32
	U32
)

// Tensor is an opaque, lazily-evaluated handle into a backend.
type Tensor interface {
	Shape() []int
}

// AsyncEvaler is implemented by backends that can schedule evaluation without
// blocking the caller, so the decode loop can build the next step while the GPU
// works on the current one. Optional; callers fall back to Eval.
type AsyncEvaler interface {
	AsyncEval(ts ...Tensor)
}

// Sweeper is implemented by backends that need manual array lifecycle (e.g. a
// cgo backend over MLX). Backends with GC-managed memory (a pure-Go backend)
// simply don't implement it; callers use it opportunistically.
//
// Usage per decode step: Pin the survivors (weights are PinAll'd once; the new
// KV caches each step), Unpin the replaced ones, then Sweep frees the rest.
type Sweeper interface {
	// PinAll pins everything created so far (used once after weight load).
	PinAll()
	// Pin marks tensors to survive the next Sweep.
	Pin(ts ...Tensor)
	// Unpin clears the pin on tensors so the next Sweep frees them.
	Unpin(ts ...Tensor)
	// Sweep frees all tracked, unpinned tensors.
	Sweep()
}

// Freer is implemented by backends that can free specific tensors immediately,
// mid-forward, without a global Sweep. MoE expert offloading uses this to drop a
// layer's materialized expert stacks before building the next layer's, so peak
// memory stays at one layer instead of accumulating all of them until the
// per-step Sweep. Optional; callers fall back to letting Sweep reclaim them.
type Freer interface {
	// Free immediately releases the given tensors (no-op on pinned ones).
	Free(ts ...Tensor)
}

// RowScatterer is implemented by backends that can replace selected leading-
// axis rows of a stacked tensor in one fused op. MoE expert slot caches use it
// to land a token's missed experts into persistent [slots, ...] tensors — one
// scatter per tensor instead of per-row updates or concat chains, which is
// what keeps the op count flat as the cache fills. Optional; callers fall back
// to rebuilding transient stacks per token.
type RowScatterer interface {
	// ScatterRows returns dst with dst[indices[i]] = updates[i] along axis 0.
	// updates is [len(indices), ...] with trailing dims equal to dst's.
	ScatterRows(dst Tensor, indices []int32, updates Tensor) Tensor
}

// Backend records a tensor graph and evaluates it. Implementations wrap a
// kernel library; model code depends only on this interface.
//
// It starts minimal and grows one op at a time as the forward-pass ladder
// climbs (embedding -> RMSNorm -> RoPE -> attention -> MLP -> ...).
type Backend interface {
	// Version reports the underlying kernel library version (smoke signal).
	Version() string

	// FromFloats creates an f32 tensor from host data with the given shape.
	FromFloats(data []float32, shape ...int) Tensor

	// FromInt32 creates an i32 tensor (e.g. token ids / gather indices).
	FromInt32(data []int32, shape ...int) Tensor

	// FromRaw creates a tensor of the given dtype directly from packed bytes
	// (used to load weights from safetensors without a Go-side decode).
	FromRaw(dt DType, raw []byte, shape ...int) Tensor

	// Add records an elementwise sum and returns the result handle.
	Add(a, b Tensor) Tensor

	// Concat joins a and b along the given axis (KV-cache growth).
	Concat(a, b Tensor, axis int) Tensor

	// Slice keeps [start:end] along axis (KV-cache truncation, last-position
	// selection), full extent on the other dims.
	Slice(x Tensor, axis, start, end int) Tensor

	// Eval forces evaluation of the given tensors, materializing the graph so
	// far so it doesn't grow unbounded across decode steps.
	Eval(ts ...Tensor)

	// Mul records an elementwise product (with broadcasting).
	Mul(a, b Tensor) Tensor

	// SiLU applies the SiLU/swish activation x * sigmoid(x).
	SiLU(x Tensor) Tensor

	// RMSNorm applies x / sqrt(mean(x^2)+eps) * weight (fused kernel).
	RMSNorm(x, weight Tensor, eps float32) Tensor

	// LayerNorm applies standard layer normalization with weight + bias.
	LayerNorm(x, weight, bias Tensor, eps float32) Tensor

	// Erf is the elementwise error function (for exact GELU).
	Erf(x Tensor) Tensor

	// Mean averages over the given axis (axis removed).
	Mean(x Tensor, axis int) Tensor

	// RoPE applies rotary position embedding to the last `dims` features of x,
	// using position offset..offset+seq along the second-to-last axis.
	RoPE(x Tensor, dims int, traditional bool, base, scale float32, offset int) Tensor

	// RoPEFreqs is RoPE with explicit per-dim frequencies (base ignored) — used
	// for llama3 rope scaling.
	RoPEFreqs(x Tensor, dims int, traditional bool, scale float32, offset int, freqs Tensor) Tensor

	// Reshape returns x viewed with a new shape (same element count).
	Reshape(x Tensor, shape ...int) Tensor

	// Transpose permutes x's axes by the given order.
	Transpose(x Tensor, axes ...int) Tensor

	// SDPA is fused scaled-dot-product attention over [B, heads, L, dim]
	// tensors (GQA when q has more heads than k/v); causal applies a causal
	// mask.
	SDPA(q, k, v Tensor, scale float32, causal bool) Tensor

	// MatMul is batched matrix multiply with NumPy-style broadcasting.
	MatMul(a, b Tensor) Tensor

	// Tanh applies elementwise hyperbolic tangent.
	Tanh(x Tensor) Tensor

	// Softmax is a numerically-precise softmax along axis.
	Softmax(x Tensor, axis int) Tensor

	// ExpandDims inserts a size-1 axis at the given position.
	ExpandDims(x Tensor, axis int) Tensor

	// Gelu is the tanh-approximation GELU activation.
	Gelu(x Tensor) Tensor

	// ScalarMul / ScalarAdd apply x*v and x+v (broadcast scalar).
	ScalarMul(x Tensor, v float32) Tensor
	ScalarAdd(x Tensor, v float32) Tensor

	// Cast converts x to the given dtype.
	Cast(x Tensor, dt DType) Tensor

	// TakeAxis gathers slices of a along the given axis by integer indices
	// (row gather for embeddings when axis == 0).
	TakeAxis(a, indices Tensor, axis int) Tensor

	// Dequantize reconstructs a float tensor from MLX affine-quantized
	// (weight, scales, biases) with the given group size and bit width.
	Dequantize(w, scales, biases Tensor, groupSize, bits int) Tensor

	// QuantMatmul computes x @ w (w affine-quantized). With transpose=true it
	// treats w as [out, in] (the QuantizedLinear convention) -> [..., out].
	QuantMatmul(x, w, scales, biases Tensor, transpose bool, groupSize, bits int) Tensor

	// GatherQuantMatmul is the batched MoE expert matmul: w/scales/biases are
	// stacked [E, out, in] over experts; rhsIndices selects which expert applies
	// to each row. Output shape is rhsIndices.shape + [out]. This is the fused
	// op (MLX gather_qmm) that runs all of a layer's active experts in one call
	// instead of one QuantMatmul per expert.
	GatherQuantMatmul(x, w, scales, biases, rhsIndices Tensor, transpose bool, groupSize, bits int) Tensor

	// Floats evaluates t (forcing the graph) and copies its data out.
	Floats(t Tensor) []float32

	// Argmax returns the index of the max along axis (axis removed).
	Argmax(x Tensor, axis int) Tensor

	// Ints evaluates t and copies it out as int32 (small index tensors).
	Ints(t Tensor) []int32

	// Close releases backend-level resources (e.g. the default stream).
	Close()
}
