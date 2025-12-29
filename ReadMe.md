# WasmRust

**WasmRust** is a research-driven, production-oriented effort to make **Rust a WASM-native language**, not merely a language that *targets* WebAssembly.

WasmRust is not trying to “replace Rust”. It is asking a harder question:

> *What does it mean for a language to truly belong to WebAssembly?*

If Rust is to remain the foundation of the WASM ecosystem for the next decade, it must evolve **with WASM**, not around it. WasmRust exists to explore that evolution — openly, globally, and rigorously.

Rather than forking Rust or reinventing the ecosystem, WasmRust explores **minimal, evidence-based extensions** to Rust’s language, compiler, and tooling that close the real gaps in today’s Rust → WASM pipeline: binary size, compile time, component interoperability, and host friction.

> **Position**: Rust is a strong foundation for WASM — but not inherently optimal. WasmRust exists to close that gap.

---

## ✨ Motivation

Rust dominates the WASM ecosystem today (`wasmtime`, `wasmer`, `wasm-tools`, `wit-bindgen`), yet developers consistently encounter:

*   ❌ **Large binaries**: A "hello world" can be 35 KB, compared to 2 KB in C, due to monomorphization.
*   ❌ **Slow compile times**: The LLVM backend and borrow-checking lead to slow iteration cycles.
*   ❌ **JS interop friction**: The `wasm-bindgen` glue layer adds overhead and cognitive load.
*   ❌ **A steep learning curve**: Ownership and lifetimes can be a barrier for non-systems developers.
*   ❌ **Mismatches with the WASM Component Model**: Rust's semantics don't always map cleanly to WASM's emerging standards.

At the same time, alternative languages (Zig, AssemblyScript, Grain) demonstrate that **WASM can be smaller, faster, and simpler** — often at the cost of safety or ecosystem maturity.

**WasmRust asks a different question**:

> *What would Rust look like if WASM were a first-class execution model?*

---

## 🌍 Design Philosophy

WasmRust is guided by five core principles:

1.  **WASM-Native Semantics**: Model WASM concepts (resources, memories, components) *directly*, not via glue code.
2.  **Safety Without Runtime Bloat**: Preserve Rust’s memory safety while eliminating unnecessary abstraction overhead.
3.  **Incremental Adoption**: Interoperate with existing Rust, `wasm-bindgen`, and WASI code.
4.  **Global & Federated**: Avoid centralized registries and vendor lock-in; design for global accessibility.
5.  **Evidence Over Dogma**: Every feature must justify itself via benchmarks, size, or correctness.

---

## 🧱 Architecture Overview

WasmRust is structured as a **five-layer stack**, each independently useful and incrementally adoptable.

```
┌─────────────────────────────────────┐
│ Layer 5 — Tooling & Ecosystem        │
│ Federated registries, debugging      │
├─────────────────────────────────────┤
│ Layer 4 — Compiler                   │
│ WasmIR, Cranelift-first, PGO         │
├─────────────────────────────────────┤
│ Layer 3 — Runtime Semantics          │
│ Multi-memory, regions, streaming     │
├─────────────────────────────────────┤
│ Layer 2 — Component Model            │
│ WIT-native imports/exports           │
├─────────────────────────────────────┤
│ Layer 1 — Core Language Extensions   │
│ Linear types, effects, concurrency   │
└─────────────────────────────────────┘
```

---

### 🧠 Layer 1 — Core Language Extensions

#### 1. Linear Types for WASM Resources

WASM's [resource types](https://github.com/WebAssembly/component-model/blob/main/design/mvp/WIT.md#resources) require **affine types** (use-once semantics) to prevent resource leaks.

```rust
// Linear type enforced at compile-time
#[wasm::linear]
struct CanvasContext(wasm::Handle);

impl CanvasContext {
    fn draw(&mut self) { /* ... */ }

    // Consuming method (moves ownership)
    fn into_bitmap(self) -> ImageData { /* ... */ }
}

// ❌ Compile error: can't use after move
let ctx = acquire_canvas();
let img = ctx.into_bitmap();
ctx.draw(); // ERROR: value moved
```

#### 2. Structured Concurrency (WASM Threads)

Scoped concurrency with automatic joining and cancellation, matching patterns familiar to non-Rust developers.

```rust
use wasm::thread::scope;

#[wasm::export]
fn parallel_transform(data: SharedSlice<f32>) -> Result<(), Error> {
    scope(|s| {
        for chunk in data.chunks(1000) {
            s.spawn(|| process(chunk)); // Lifetime tied to scope
        }
        // ← All threads joined here automatically
    })?;
    Ok(())
}
```

#### 3. Effect System for Side Effects

Track I/O, JS calls, and atomics at the type level to enable powerful optimizations like tree-shaking dead effects.

```rust
// Effectful functions (explicit markers)
#[wasm::effect(js_call, atomic_read)]
fn fetch_and_cache(url: &str) -> Result<Vec<u8>, Error> {
    let data = js::fetch(url)?;  // js_call effect
    CACHE.store(url, data);       // atomic_write effect (inferred)
    Ok(data)
}
```

---

### 🔌 Layer 2 — Component Model Deep Dive

#### WIT Syntax as a First-Class Citizen

WasmRust treats [WIT IDL](https://component-model.bytecodealliance.org/design/wit.html) as a first-class interface language, enabling bidirectional code generation.

```rust
// Import definition (compiles to WIT)
#[wasm::wit]
interface crypto {
    use types.{bytes};

    resource key-pair {
        constructor(algorithm: string);
        sign: func(data: bytes) -> bytes;
    }
}

// Usage (type-safe, no glue)
use crypto::{KeyPair};

#[wasm::export]
fn sign_message(msg: &[u8]) -> Vec<u8> {
    let kp = KeyPair::new("ed25519");
    kp.sign(msg)
}
```

---

### ⚙️ Layer 3 — Runtime Semantics

#### Multi-Region Memory

First-class support for data residency and isolation, critical for GDPR and other compliance requirements.

```rust
#[wasm::memory(region = "eu-west-1", encryption = "AES256-GCM")]
static EU_DATA: wasm::Memory<8_000_000>; // 8 MB max

#[wasm::memory(region = "cn-north-1")]
static CN_DATA: wasm::Memory<8_000_000>;
```

#### Streaming Compilation Hints

Provide hints to browser engines to optimize layout and achieve 30-50% faster Time to Interactive.

```rust
#[wasm::compile_hints(
    tier = "baseline",  // Fast startup
    critical = ["render_frame", "handle_input"]
)]
mod ui;
```

---

### 🛠️ Layer 4 — Compiler Strategy

#### Cranelift-First Backend

Use [Cranelift](https://cranelift.dev/) for fast development builds and LLVM for optimized release builds.

*   **Dev builds**: Cranelift only (~2s compile for 10k LOC)
*   **Release builds**: LLVM + `wasm-opt` (~8s, 30% smaller)

#### Profile-Guided Optimization (PGO) via Instrumentation

Collect profiles in production to guide optimization decisions.

```bash
# Step 1: Build with instrumentation
cargo wasm build --profile=instrumented

# Step 2: Collect profiles in production
wasm-runner ./app.wasm --collect-profile=prod.prof

# Step 3: Rebuild with profile data
cargo wasm build --release --pgo=prod.prof
```
---

### 🌐 Layer 5 — Tooling & Ecosystem

#### Federated Registries

Avoid centralized points of failure and geopolitical restrictions by using federated registries.

```bash
# Add multiple registries
cargo wasm registry add apac https://wasm.asia/registry
cargo wasm add crypto@1.2 --registry=apac,bytecode-alliance
```

#### WASM-Aware Debugging

Build native tooling for better memory inspection and visualization in browser DevTools.

```bash
# Attach debugger with memory inspection
wasm-gdb ./app.wasm --port 9229

# Visualize memory layout
(gdb) wasm mem visualize
```

---

## 📊 Comparative Snapshot

| Metric          | WasmRust | Rust+bindgen | Zig   | AssemblyScript |
| --------------- | -------- | ------------ | ----- | -------------- |
| Binary size     | **~2 KB**    | ~35 KB       | **~1 KB** | ~8 KB          |
| Compile time    | **~3s**      | ~12s         | **~2s**   | ~4s            |
| Memory safety   | ✅        | ✅            | ⚠️    | ⚠️             |
| Component Model | ✅ Native | ❌            | ⚠️    | ❌              |
| JS Interop      | **0% overhead** | 5-10%       | 3-5%  | 1-3%           |
| Threads Safety  | ✅ Compile-time | ⚠️ Unsafe   | ⚠️ Unsafe | ⚠️ Unsafe   |

---

## 🚀 Roadmap

### Phase 1: Proof of Concept (3 months)
1.  **`wasm` crate**: `externref<T>`, `SharedSlice<T>`, `#[wasm::export]` macro
2.  **Cranelift backend**: Fork `rustc_codegen_cranelift`, add WASM target
3.  **Benchmark**: Compare vs Rust, AS, Zig on Mandelbrot/N-body

### Phase 2: Component Model (6 months)
4.  Bidirectional WIT ↔ Rust codegen
5.  `cargo-wasm` with federated registry support
6.  Browser DevTools integration (memory visualizer)

### Phase 3: Standardization (12 months)
7.  RFC to Rust project (Layer 1 features)
8.  Bytecode Alliance collaboration (WASI-P2 integration)
9.  W3C WebAssembly CG presentation

---

## 🎯 Critical Success Factors

1.  **Incremental adoption**: Must interop with existing `wasm-bindgen` code.
2.  **Binary size obsession**: Every byte matters for mobile/edge.
3.  **China/India developer experience**: Documentation in Mandarin, Hindi, Spanish.
4.  **Avoid vendor lock-in**: No Anthropic/OpenAI APIs in toolchain (preserve sovereignty).

---

## 🚧 Project Status

**Early research / prototype phase.**

*   APIs are unstable.
*   Ideas are experimental.
*   Benchmarks and measurements drive decisions.

---

## 🤝 Contributing

WasmRust is **research-first and community-driven**. We welcome:

*   Benchmarks
*   Compiler experiments
*   Design critiques
*   Documentation & localization

See `CONTRIBUTING.md` for details.

---
