# WasmRust

**WasmRust** is a research-driven, production-oriented effort to make **Rust a WASM-native language**, not merely a language that *targets* WebAssembly.

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

**WasmRust asks a different question**:

> *What would Rust look like if WASM were a first-class execution model?*

---

## What Is WasmRust?

WasmRust is a **specialized Rust toolchain** that keeps the Rust frontend unchanged (parser, HIR, MIR, borrow checker) and swaps or augments code generation for WASM. It provides library-level primitives that map directly to WASM concepts.

```text
┌─────────────────────────────────────────────┐
│                 rustc frontend              │
│   (parsing, HIR, MIR, borrow checking)       │
│                 UNCHANGED                   │
└───────────────────┬─────────────────────────┘
                    ▼
┌─────────────────────────────────────────────┐
│           WASM-specialized codegen           │
│   ┌────────────────┬────────────────────┐   │
│   │ Cranelift WASM │ LLVM WASM           │   │
│   │ (dev builds)   │ (release builds)    │   │
│   └────────────────┴────────────────────┘   │
└─────────────────────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────┐
│         crates/wasm (zero-cost APIs)         │
│   externref, threads, components, memory    │
└─────────────────────────────────────────────┘
```

---

## 🌍 Design Philosophy

1.  **WASM-Native Semantics**: Model WASM concepts (resources, memories, components) *directly*, not via glue code.
2.  **Safety Without Runtime Bloat**: Preserve Rust’s memory safety while eliminating unnecessary abstraction overhead.
3.  **Incremental Adoption**: Interoperate with existing Rust, `wasm-bindgen`, and WASI code.
4.  **Global & Federated**: Avoid centralized registries and vendor lock-in; design for global accessibility.
5.  **Evidence Over Dogma**: Every feature must justify itself via benchmarks, size, or correctness.

---

## 🧱 Architecture

WasmRust is structured as a **five-layer stack**, each independently useful and incrementally adoptable.

```mermaid
graph TB
    subgraph "Layer 5: Tooling & Distribution"
        A[cargo-wasm CLI [planned]]
        B[Registry Federation]
        C[Debug Tools]
        D[Profiler]
    end

    subgraph "Layer 4: Compiler Backend"
        E[Cranelift Backend]
        F[LLVM Backend]
        G[Profile-Guided Optimization]
        H[Verifier Pass [planned]]
        I[wasm-recognition Lints [planned]]
    end

    subgraph "Layer 3: Runtime Services"
        J[Memory Management]
        K[Threading Runtime]
        L[Component Linking]
    end

    subgraph "Layer 2: Language Extensions"
        M[Component Model Macros]
        N[WIT Integration]
        O[Capability Annotations]
    end

    subgraph "Layer 1: Core Language"
        P[WASM Native Types]
        Q[Linear Types]
        R[Safe Abstractions]
    end

    A --> E
    A --> F
    E --> J
    F --> J
    H --> E
    H --> F
    I --> H
    M --> P
    N --> Q
    J --> R
```

---

## The 5 Layers in Detail

### 🧠 Layer 1: Core Language Extensions & `crates/wasm`

The `wasm` crate is the foundation: `no_std`, dependency-free, runtime-free, and compiler-agnostic, providing zero-cost abstractions over WASM primitives.

*   **Linear Types for WASM Resources**: Prevents resource leaks with use-once semantics.
    ```rust
    #[wasm::linear]
    struct CanvasContext(wasm::Handle);

    impl CanvasContext {
        fn draw(&mut self) { /* ... */ }
        fn into_bitmap(self) -> ImageData { /* ... */ } // Consumes self
    }
    ```
*   **Structured Concurrency**: Scoped concurrency with automatic joining.
    ```rust
    use wasm::thread::scope;
    scope(|s| {
        for chunk in data.chunks(1000) {
            s.spawn(|| process(chunk)); // Lifetime tied to scope
        }
    });
    ```
*   **Effect System for Side Effects**: Tracks I/O and JS calls at the type level for optimization.
    ```rust
    #[wasm::effect(js_call)]
    fn fetch_data(url: &str) -> Result<Vec<u8>, Error> { /* ... */ }
    ```

### 🔌 Layer 2: Component Model Deep Dive

Treats WIT as a first-class interface language, enabling bidirectional code generation without glue code.

```rust
#[wasm::wit]
interface crypto {
    resource key-pair {
        constructor(algorithm: string);
        sign: func(data: bytes) -> bytes;
    }
}
```

### ⚙️ Layer 3: Runtime Semantics

*   **Multi-Region Memory**: First-class support for data residency and isolation (e.g., for GDPR).
    ```rust
    #[wasm::memory(region = "eu-west-1")]
    static EU_DATA: wasm::Memory<8_000_000>;
    ```
*   **Streaming Compilation Hints**: Optimize startup latency in browsers.
    ```rust
    #[wasm::compile_hints(tier = "baseline", critical = ["render_frame"])]
    mod ui;
    ```

### 🛠️ Layer 4: Compiler Strategy

*   **Cranelift-First Backend**: Fast dev builds (~2s) via Cranelift, optimized release builds via LLVM (~8s, 30% smaller).
*   **Compilation Pipeline**:
    ```mermaid
    graph LR
        A[Rust Source] --> B[HIR/MIR]
        B --> C[WasmIR - Stable Boundary]
        C --> D{Build Profile}
        D -->|Development| E[Cranelift Backend]
        D -->|Release| F[LLVM Backend]
        E --> G[Fast WASM + Debug Info]
        F --> H[Optimized WASM]
        H --> I[wasm-opt]
        I --> J[Component Model Wrapper]
    ```

### 🌐 Layer 5: Tooling & Ecosystem

*   **Federated Registries**: Avoids centralized points of failure.
    ```bash
    cargo wasm registry add apac https://wasm.asia/registry
    cargo wasm add crypto --registry=apac,bytecode-alliance
    ```
*   **WASM-Aware Debugging**: Native tooling for memory inspection.
    ```bash
    wasm-gdb ./app.wasm --port 9229
    (gdb) wasm mem visualize
    ```
---

## Repository Structure

```
wasmrust/
├── compiler/                # rustc extensions & backends
│   ├── codegen-cranelift/   # WASM-tuned Cranelift backend
│   └── codegen-llvm/        # WASM-optimized LLVM backend
│
├── crates/
│   ├── wasm/                # Core zero-cost WASM abstractions
│   └── wasm-macros/         # Proc macros for Component Model / WIT [planned]
│
├── tooling/
│   └── cargo-wasm/          # WASM-aware Cargo frontend [planned]
│
├── docs/
│   ├── SAFETY.md            # Unsafe invariants per type / crate
│   ├── compiler-contract.md # Formal compiler ↔ crate contracts
│   ├── RFCs/
│   └── architecture/
│
└── ReadMe.md
```

---

## Incremental Adoption

#### What Works Without WasmRust?
Everything in `crates/wasm`: it compiles on **stable Rust**, produces valid WASM, and has no dependency on a custom compiler. WasmRust **enhances**, but does not gate, functionality.

#### What Requires the WasmRust Compiler?
Native Component Model emission, Cranelift-accelerated builds, and advanced optimizations like PGO and thin monomorphization.

---

## Contracts & Governance

*   **Language Surface Contract**: Core (80%): Standard Rust; Extensions (15%): `wasm` crate; Plugins (4%): `-Z` flags; Hard Fork (<1%): Minimal changes if required.
*   **Compiler ↔ Crate Contract**: The compiler assumes invariants for types like `ExternRef` and `SharedSlice` which are documented in `SAFETY.md` and checked by compiler passes.
*   **Governance & Direction**: Upstream-friendly, library-first stabilization, and RFC-driven evolution.

---

## Notes on SAFETY.md

* Contains **formal unsafe invariants** per type.
* Used by the compiler **verifier pass** and **lint group**.
* Serves as authoritative documentation for both crate users and compiler developers.

---

## Host Profile Support

| Host Profile | Threading                     | JS Interop      | Component Model | Memory Regions |
| ------------ | ----------------------------- | --------------- | --------------- | -------------- |
| Browser      | SharedArrayBuffer + COOP/COEP | Direct calls    | Partial         | No             |
| Node.js      | Worker threads                | Native bindings | Polyfill        | No             |
| Wasmtime     | wasi-threads                  | Host functions  | Full            | Configurable   |
| Embedded     | No                            | No              | Partial         | No             |

---

## Testing and Verification

*   Property-Based Testing: binary size, ownership, threading safety.
*   Cross-Language ABI Testing: Zig, C, and other WASM components.
*   Reproducible Builds and Performance Benchmarks.

---

## Non-Goals

WasmRust is **not** a Rust fork, a new language, a replacement for `wasm-bindgen` (initially), or a JS framework.

---

## 📊 Comparative Snapshot

| Metric          | WasmRust | Rust+bindgen | Zig   | AssemblyScript |
| --------------- | -------- | ------------ | ----- | -------------- |
| Binary size     | **~2 KB**    | ~35 KB       | **~1 KB** | ~8 KB          |
| Compile time    | **~3s**      | ~12s         | **~2s**   | ~4s            |
| Memory safety   | ✅        | ✅            | ⚠️    | ⚠️             |
| Component Model | ✅ Native | ❌            | ⚠️    | ❌              |

---

## 🚀 Roadmap

*   **Phase 1 (3 mo)**: `wasm` crate PoC, Cranelift backend, Benchmarks.
*   **Phase 2 (6 mo)**: Bidirectional WIT codegen, `cargo-wasm`, Debugging tools.
*   **Phase 3 (12 mo)**: Rust RFCs, Bytecode Alliance collaboration, Wasm CG presentation.

---

## 🤝 Contributing

WasmRust is research-first and community-driven. We welcome benchmarks, compiler experiments, design critiques, and documentation.

#### Where to Start
*   📦 Use `crates/wasm` for low-level WASM code today.
*   📖 Read `docs/RFCs/0001-wasmrust-architecture.md`.
*   🧪 Experiment with Cranelift WASM builds (nightly).
*   🛠️ Contribute to core abstractions before compiler work.

---

## 🧭 Final Word

WasmRust is not trying to “replace Rust”. It is asking a harder question:

> *What does it mean for a language to truly belong to WebAssembly?*

WasmRust exists to explore that evolution — openly, globally, and rigorously.
