# WasmRust — Rust-to-WebAssembly Compiler

**WasmRust** is a research-driven, production-oriented effort to make **Rust a WASM-native language**, not merely a language that *targets* WebAssembly.

WasmRust is not trying to “replace Rust”. It is asking a harder question:

> *What does it mean for a language to truly belong to WebAssembly?*

If Rust is to remain the foundation of the WASM ecosystem for the next decade, it must evolve **with WASM**, not around it. WasmRust exists to explore that evolution — openly, globally, and rigorously.

Rather than forking Rust or reinventing the ecosystem, WasmRust explores **minimal, evidence-based extensions** to Rust’s language, compiler, and tooling that close the real gaps in today’s Rust → WASM pipeline: binary size, compile time, component interoperability, and host friction.

> **Position**
> Rust is a strong foundation for WASM — but not inherently optimal.
> WasmRust exists to close that gap.

---

## Is Rust the Best Language for WASM? A Critical Analysis

### Where Rust Excels
- **Memory safety without GC**: Critical for WASM's no-runtime philosophy
- **Zero-cost abstractions**: Maps cleanly to WASM's stack machine
- **Predictable performance**: No hidden allocations or runtime surprises
- **Ecosystem maturity**: `wasmtime`, `wasmer`, `wasm-tools` heavily Rust-based

### Structural Limitations
| Challenge | Root Cause | Impact on WASM |
|-----------|-----------|----------------|
| **Large binaries** | Monomorphization explosion | 35 KB "hello world" vs 2 KB in C |
| **Compile times** | LLVM backend, borrow-checking | Slow iteration for web dev |
| **JS interop friction** | `wasm-bindgen` glue layer | 5-10% overhead, cognitive load |
| **Learning curve** | Lifetimes, ownership | Barrier vs TypeScript/AssemblyScript |

### Alternative Paradigms
- **Zig**: Manual memory management, comptime metaprogramming → ~1 KB binaries
- **Nim**: Python-like syntax, compiled to C → predictable WASM output
- **OCaml/ReScript**: Strong type inference, GC-aware WASM backend
- **Idris2**: Dependent types → provably correct WASM modules
- **Gleam**: Erlang VM alternative targeting WASM via Rust backend

**Verdict**: Rust is a **strong foundation**, but **not inherently optimal**. The key is designing a **WASM-native dialect** that removes impedance mismatches.

---

## Design Philosophy

WasmRust is guided by five core principles:

1. **WASM-Native Semantics**: Model WASM concepts (resources, memories, components) *directly*, not via glue code.
2. **Safety Without Runtime Bloat**: Preserve Rust’s memory safety while eliminating unnecessary abstraction overhead.
3. **Incremental Adoption**: Interoperate with existing Rust, `wasm-bindgen`, and WASI code.
4. **Global & Federated**: Avoid centralized registries and vendor lock-in; design for global accessibility.
5. **Evidence Over Dogma**: Every feature must justify itself via benchmarks, size, or correctness.

---

## Architecture Overview

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

## Layer 1: Core Language — Critical Additions

#### 1. Linear Types for WASM Resources
WASM's [resource types](https://github.com/WebAssembly/component-model/blob/main/design/mvp/WIT.md#resources) require **affine types** (use-once semantics):

```rust
// Linear type enforced at compile-time
#[wasm::linear]
struct CanvasContext(wasm::Handle);

impl CanvasContext {
    fn draw(&mut self) { /* ... */ }
    
    // Consuming method (moves ownership)
    fn into_bitmap(self) -> ImageData { /* ... */ }
}
```
**Why**: Prevents resource leaks in Component Model integrations (e.g., Web APIs, WASI sockets).

#### 2. Structured Concurrency (WASM Threads)
Provides robust concurrency with compile-time checked **cancellation** and **scoped lifetimes**:

```rust
use wasm::thread::scope;

#[wasm::export]
fn parallel_transform(data: SharedSlice<f32>) -> Result<(), Error> {
    scope(|s| {
        for chunk in data.chunks(1000) {
            s.spawn(|| process(chunk)); // Lifetime tied to scope
        }
    })?;
    Ok(())
}
```
**Benefit**: Matches Trio/Kotlin coroutines patterns familiar to non-Rust devs.

#### 3. Effect System for Side Effects
Track I/O, JS calls, and atomics at type level:

```rust
// Pure functions (no side effects)
fn fibonacci(n: u32) -> u32 { /* ... */ }

// Effectful functions (explicit markers)
#[wasm::effect(js_call, atomic_read)]
fn fetch_and_cache(url: &str) -> Result<Vec<u8>, Error> {
    let data = js::fetch(url)?;
    CACHE.store(url, data);
    Ok(data)
}
```
**Why**: Enables **tree-shaking dead effects** (e.g., remove all `js_call` code for server-side WASI builds).

---

## Layer 2: Language Extensions — Component Model Deep Dive

#### WIT Syntax as First-Class Citizen
Match [WIT IDL](https://component-model.bytecodealliance.org/design/wit.html) directly:

```rust
#[wasm::wit]
interface crypto {
    use types.{bytes};
    
    resource key-pair {
        constructor(algorithm: string);
        sign: func(data: bytes) -> bytes;
    }
    
    hash-sha256: func(data: bytes) -> bytes;
}
```
**Critical**: WIT → Rust codegen should be **bidirectional**.

#### Variance-Aware Generics
WASM Component Model requires [**subtyping**](https://github.com/WebAssembly/component-model/blob/main/design/mvp/Subtyping.md):

```rust
// Covariant in return type
trait Serializer {
    type Output: wasm::Exportable;
    fn encode(&self) -> Self::Output;
}
```
**Why**: Allows **safe component substitution**.

---

## Layer 3: Runtime — Global Constraints

#### Multi-Region Memory (China/EU Isolation)
For GDPR/data residency:

```rust
#[wasm::memory(region = "eu-west-1", encryption = "AES256-GCM")]
static EU_DATA: wasm::Memory<8_000_000>;

#[wasm::memory(region = "cn-north-1")]
static CN_DATA: wasm::Memory<8_000_000>;
```
**Implementation**: Compiler generates **multiple memory instances** per Component Model spec.

#### Streaming Compilation Hints
Optimize layout for browser engines:

```rust
#[wasm::compile_hints(
    tier = "baseline",
    critical = ["render_frame", "handle_input"]
)]
mod ui;

#[wasm::compile_hints(tier = "optimized")]
mod background_tasks;
```
**Benefit**: 30-50% faster **Time to Interactive** on mobile devices.

---

## Layer 4: Compiler — Architecture Shift

#### Cranelift-First Backend
Use [Cranelift](https://cranelift.dev/) for 10x faster compile times:

**Strategy**:
- **Dev builds**: Cranelift only (~2s compile for 10k LOC)
- **Release builds**: LLVM + `wasm-opt` (~8s, 30% smaller)

#### Profile-Guided Optimization (PGO) via Instrumentation
```bash
cargo wasm build --profile=instrumented
wasm-runner ./app.wasm --collect-profile=prod.prof
cargo wasm build --release --pgo=prod.prof
```
---

## Layer 5: Tooling — Decentralized Ecosystem

#### Component Registry — Avoid Centralization
Use **federated registries**:

```bash
cargo wasm registry add apac https://wasm.asia/registry
cargo wasm add crypto@1.2 --registry=apac,bytecode-alliance
```
**Why**: Resilience against geopolitical restrictions.

#### WASM-Aware Debugging
Build **native tooling** for better memory inspection:

```bash
wasm-gdb ./app.wasm --port 9229
```
---

## Revised Comparison: WasmRust vs Alternatives

| Metric | **WasmRust** | Rust + wasm-bindgen | AssemblyScript | **Zig** | **Grain** |
|---|---|---|---|---|---|
| **Binary Size** | **~2 KB** | ~35 KB | ~8 KB | **~1 KB** | ~6 KB |
| **Compile Time**| **~3s (Cranelift)**| ~12s (LLVM)| ~4s| **~2s**| ~3s |
| **Memory Safety**| ✅ Borrow-checked | ✅ Borrow-checked | ⚠️ Manual| ⚠️ Manual | ✅ Type-safe |
| **Component Model**| ✅ Native| ❌ Glue layer| ❌ None| ⚠️ Partial| ⚠️ Planned |
| **JS Interop**| **0% overhead** | 5-10%| 1-3%| 3-5%| 1-2% |
| **Learning Curve**| **Gentle** (Polonius)| Steep| Easy| Moderate | Moderate |
| **Threads Safety**| ✅ Compile-time | ⚠️ Unsafe| ⚠️ Unsafe| ⚠️ Unsafe| ⚠️ Unsafe |
| **Ecosystem**| 🌱 Bootstrap| 🌳 Mature| 🌿 Growing| 🌿 Growing| 🌱 Early |

---

## Roadmap

### Phase 1: Proof of Concept (3 months)
1. **`wasm` crate**: `externref<T>`, `SharedSlice<T>`, `#[wasm::export]` macro
2. **Cranelift backend**: Fork `rustc_codegen_cranelift`, add WASM target
3. **Benchmark**: Compare vs Rust, AS, Zig on Mandelbrot/N-body

### Phase 2: Component Model (6 months)
4. Bidirectional WIT ↔ Rust codegen
5. `cargo-wasm` with federated registry support
6. Browser DevTools integration (memory visualizer)

### Phase 3: Standardization (12 months)
7. RFC to Rust project (Layer 1 features)
8. Bytecode Alliance collaboration (WASI-P2 integration)
9. W3C WebAssembly CG presentation

---

## Critical Success Factors

1. **Incremental adoption**: Must interop with existing `wasm-bindgen` code
2. **Binary size obsession**: Every byte matters for mobile/edge
3. **China/India developer experience**: Documentation in Mandarin, Hindi, Spanish
4. **Avoid vendor lock-in**: No Anthropic/OpenAI APIs in toolchain (preserve sovereignty)

---

## Contributing

WasmRust is **research-first and community-driven**. We welcome:
* Benchmarks
* Compiler experiments
- Design critiques
- Documentation & localization

See `CONTRIBUTING.md` for details.

---

## Project Status

**Early research / prototype phase.**
APIs are unstable. Ideas are experimental. Evidence matters.

