---
Status: Draft
Version: 0.1.0
Date: 2025-12-29
Type: Product Requirements Document
---

# WasmRust

## Introduction

This document specifies the requirements for WasmRust, an optimized Rust-to-WebAssembly compilation system that addresses the current limitations of the standard Rust WASM toolchain. The system aims to provide minimal binary sizes, fast compilation times, seamless Component Model integration, and zero-overhead JavaScript interoperability while maintaining Rust's memory safety guarantees.

## Glossary

- **WasmRust**: The optimized Rust-to-WASM compilation system being specified
- **Component_Model**: WebAssembly Component Model specification for composable WASM modules
- **WIT**: WebAssembly Interface Types - the IDL for Component Model interfaces
- **Linear_Memory**: WebAssembly's contiguous memory space accessible to WASM modules
- **ExternRef**: WebAssembly reference type for opaque host objects
- **FuncRef**: WebAssembly reference type for function pointers
- **Cranelift**: Fast code generator used as alternative to LLVM
- **PGO**: Profile-Guided Optimization using runtime profiling data
- **SharedSlice**: Safe abstraction for shared memory access across WASM threads
- **WasmGC**: WebAssembly Garbage Collection proposal for native GC support
- **GcArray**: Garbage-collected array type for WasmGC environments
- **GcString**: Garbage-collected string type for WasmGC environments

## Requirements

### Requirement 1: Binary Size Optimization

**User Story:** As a web developer, I want minimal WASM binary sizes, so that my applications load quickly on mobile devices and low-bandwidth connections.

#### Acceptance Criteria

1. WHEN compiling a "hello world" program with freestanding profile, THE WasmRust_Compiler SHALL generate binaries under 2 KB (improved from 15 KB to match MoonBit performance)
2. WHEN compiling applications with standard library features, THE WasmRust_Compiler SHALL generate binaries at most 3x larger than equivalent C programs compiled with similar feature sets
3. WHEN using generic functions, THE WasmRust_Compiler SHALL apply thin monomorphization to reduce code duplication by at least 30% compared to current rustc
4. WHEN dead code exists, THE WasmRust_Compiler SHALL eliminate unused functions and data through tree-shaking
5. THE WasmRust_Compiler SHALL provide size analysis tools showing per-function binary contributions with byte-level attribution

### Requirement 2: Fast Compilation Performance

**User Story:** As a developer, I want fast compilation times during development, so that I can iterate quickly on WASM applications.

#### Acceptance Criteria

1. WHEN compiling 10,000 lines of code in development mode, THE WasmRust_Compiler SHALL complete within 2 seconds (improved from 5 seconds to match MoonBit performance)
2. WHEN using Cranelift backend, THE WasmRust_Compiler SHALL compile at least 5x faster than LLVM backend
3. WHEN incremental compilation is enabled, THE WasmRust_Compiler SHALL recompile only changed modules
4. THE WasmRust_Compiler SHALL provide separate development and release build profiles
5. WHEN using release mode, THE WasmRust_Compiler SHALL apply LLVM optimizations for maximum performance

### Requirement 3: Memory Safety and Type System

**User Story:** As a systems programmer, I want memory safety guarantees in WASM, so that I can write secure applications without runtime crashes.

#### Acceptance Criteria

1. THE WasmRust_Compiler SHALL enforce Rust's ownership and borrowing rules at compile time within the WasmRust dialect constraints
2. WHEN accessing shared memory, THE SharedSlice_Type SHALL prevent data races through type system constraints limited to Pod types
3. WHEN using linear types, THE WasmRust_Compiler SHALL enforce use-once semantics for WASM resources through compiler extensions
4. THE WasmRust_Compiler SHALL provide safe abstractions for ExternRef and FuncRef types with managed reference tables
5. WHEN capability annotations are used, THE WasmRust_Compiler SHALL track capabilities for optimization hints without type-level effect enforcement

### Requirement 4: JavaScript Interoperability

**User Story:** As a web developer, I want efficient JavaScript integration with predictable performance characteristics, so that I can call JS APIs without complex bindings or unpredictable overhead.

#### Acceptance Criteria

1. WHEN calling JavaScript functions in supported host profiles, THE WasmRust_Runtime SHALL provide zero-copy data transfer and predictable boundary costs under 100 nanoseconds per call
2. WHEN passing Pod data between WASM and JS, THE WasmRust_Runtime SHALL avoid serialization through direct memory access
3. THE ExternRef_Type SHALL provide type-safe access to JavaScript objects with compile-time interface validation and runtime error handling
4. WHEN importing JS functions, THE WasmRust_Compiler SHALL generate direct WASM import declarations through managed reference tables
5. THE WasmRust_Runtime SHALL support bidirectional function calls with explicit ownership semantics and host-profile-specific error handling

### Requirement 5: Component Model Integration

**User Story:** As a system architect, I want to compose WASM modules from different languages, so that I can use the best tool for each component.

#### Acceptance Criteria

1. THE WasmRust_Compiler SHALL generate Component Model compatible WASM modules
2. WHEN defining interfaces, THE WasmRust_Compiler SHALL support bidirectional WIT code generation
3. WHEN importing components, THE WasmRust_Compiler SHALL provide type-safe bindings for external modules
4. THE WasmRust_Compiler SHALL support variance-aware generics for component substitution
5. WHEN linking components, THE WasmRust_Runtime SHALL enable zero-copy data sharing between modules

### Requirement 6: Threading and Concurrency

**User Story:** As a performance-conscious developer, I want safe concurrent programming in WASM environments that support it, so that I can utilize multiple cores when available.

#### Acceptance Criteria

1. WHEN spawning threads in environments with SharedArrayBuffer support, THE WasmRust_Runtime SHALL provide structured concurrency with automatic cleanup
2. THE SharedSlice_Type SHALL enable safe shared memory access across WASM threads with compile-time data race prevention
3. WHEN using atomic operations, THE WasmRust_Compiler SHALL generate efficient WASM atomic instructions where supported
4. THE WasmRust_Runtime SHALL detect threading capability and provide fallback single-threaded execution when threads are unavailable
5. WHEN threads complete, THE WasmRust_Runtime SHALL automatically join all spawned threads within scoped lifetimes

### Requirement 7: Development Tooling

**User Story:** As a developer, I want comprehensive development tools, so that I can debug and optimize WASM applications effectively.

#### Acceptance Criteria

1. THE WasmRust_Toolchain SHALL provide a cargo-wasm command-line tool for project management
2. WHEN debugging applications, THE WasmRust_Debugger SHALL visualize linear memory layout and usage
3. THE WasmRust_Profiler SHALL collect runtime performance data for profile-guided optimization
4. WHEN analyzing binaries, THE WasmRust_Analyzer SHALL show size breakdowns by function and module
5. THE WasmRust_Toolchain SHALL integrate with existing Rust development environments

### Requirement 8: Multi-Language Component Support

**User Story:** As a system architect, I want to combine WasmRust with high-performance modules in other languages, so that I can optimize critical paths while maintaining safety.

#### Acceptance Criteria

1. WHEN importing Zig components in supported host profiles, THE WasmRust_Runtime SHALL provide type-safe bindings through Component Model with WIT interface validation
2. WHEN importing C components, THE WasmRust_Runtime SHALL handle memory ownership semantics correctly through explicit borrow-checking at component boundaries
3. THE WasmRust_Compiler SHALL validate component interfaces at compile time and reject incompatible ABI signatures
4. WHEN linking multi-language components, THE WasmRust_Runtime SHALL enable zero-copy data sharing for Pod types and frozen buffers only
5. THE WasmRust_Toolchain SHALL support hybrid project builds with multiple source languages through unified build orchestration

### Requirement 9: Global Registry and Distribution

**User Story:** As a developer in any region, I want reliable access to WASM components, so that I can build applications without geographic restrictions.

#### Acceptance Criteria

1. THE WasmRust_Registry SHALL support federated component registries across multiple regions
2. WHEN a primary registry is unavailable, THE WasmRust_Toolchain SHALL automatically fallback to mirror registries
3. THE WasmRust_Registry SHALL support self-hosted private registries for enterprise use
4. WHEN publishing components, THE WasmRust_Toolchain SHALL support multiple registry targets
5. THE WasmRust_Registry SHALL provide cryptographic verification of component integrity

### Requirement 10: Profile-Guided Optimization

**User Story:** As a performance engineer, I want to optimize WASM binaries based on production usage patterns, so that I can achieve maximum runtime performance.

#### Acceptance Criteria

1. WHEN building with instrumentation, THE WasmRust_Compiler SHALL embed profiling hooks in the generated WASM with deterministic profile collection and toolchain version tracking
2. WHEN collecting profiles, THE WasmRust_Runtime SHALL record function call frequencies and memory access patterns with provenance tracking and normalization
3. WHEN rebuilding with profile data, THE WasmRust_Compiler SHALL optimize hot paths and inline frequently called functions while maintaining reproducible builds with identical toolchain versions
4. THE WasmRust_Compiler SHALL support lazy loading of cold code paths based on profile data through Component Model dynamic linking
5. THE WasmRust_Profiler SHALL provide visualization of performance bottlenecks and optimization opportunities with actionable recommendations

### Requirement 11: Host Profile Compatibility

**User Story:** As a deployment engineer, I want WasmRust applications to work across different execution environments, so that I can deploy the same code to browsers, servers, and edge computing platforms.

#### Acceptance Criteria

1. THE WasmRust_Runtime SHALL detect host profile capabilities at load time and adapt execution accordingly
2. WHEN threading is unavailable, THE WasmRust_Runtime SHALL provide single-threaded fallback execution without code changes
3. WHEN Component Model is unsupported, THE WasmRust_Runtime SHALL provide polyfill implementations for basic component functionality according to the Component Model Support Matrix
4. THE WasmRust_Compiler SHALL generate host-profile-specific optimizations based on target environment declarations
5. WHEN memory region intents are unsupported, THE WasmRust_Runtime SHALL fail gracefully at load time with clear error messages

### Requirement 12: Compiler Architecture

**User Story:** As a Rust developer, I want to understand WasmRust's relationship to stable Rust, so that I can assess migration costs and compatibility.

#### Acceptance Criteria

1. THE WasmRust_Compiler SHALL be implemented as a rustc extension with custom codegen backend, where 80% of features are library-based, 15% use unstable compiler flags, and less than 5% require incompatible changes
2. WHEN compiling standard Rust code, THE WasmRust_Compiler SHALL produce functionally equivalent output to rustc with LLVM backend
3. THE WasmRust_Compiler SHALL document all deviations from Rust language specification in a compatibility matrix
4. WHEN linear types are used, THE WasmRust_Compiler SHALL provide clear migration path if upstream Rust adopts different syntax
5. THE WasmRust_Compiler SHALL maintain compatibility with the stable Rust ecosystem including crates.io dependencies

### Requirement 13: Compiler-Crate Contract

**User Story:** As a compiler engineer, I want explicit contracts between the compiler and wasm crate, so that I can implement sound optimizations without ecosystem lock-in.

#### Acceptance Criteria

1. THE WasmRust_Compiler SHALL only assume documented unsafe invariants from the wasm crate as specified in the formal Compiler ↔ Crate Contract
2. WHEN optimizing wasm crate usage, THE WasmRust_Compiler SHALL only pattern-match explicitly whitelisted MIR shapes
3. THE WasmRust_Compiler SHALL provide mechanical verification that all optimizations reference documented invariants
4. WHEN the wasm crate is replaced or removed, THE WasmRust_Compiler SHALL produce semantically equivalent output to stable rustc
5. THE WasmRust_Compiler SHALL implement the wasm-recognition lint group to prevent unsound assumptions

### Requirement 14: Dual-Mode Compilation System

**User Story:** As a web developer, I want to choose between ownership-based and GC-based memory management for different parts of my application, so that I can optimize for performance or simplicity as needed.

#### Acceptance Criteria

1. THE WasmRust_Compiler SHALL support dual compilation modes: ownership mode (default) and GC mode via #[wasm::gc] attribute
2. WHEN compiling in ownership mode, THE WasmRust_Compiler SHALL use Rust's standard ownership and borrowing semantics
3. WHEN compiling with #[wasm::gc] attribute, THE WasmRust_Compiler SHALL generate WasmGC-compatible code using garbage collection
4. THE WasmRust_Compiler SHALL allow mixing ownership and GC modes within the same codebase through explicit boundaries
5. WHEN targeting WASM environments without GC support, THE WasmRust_Compiler SHALL provide polyfill implementations or compile-time errors

### Requirement 15: GC-Native Type System

**User Story:** As a developer targeting modern WASM runtimes, I want native garbage-collected types that integrate seamlessly with WasmGC, so that I can write simpler code without manual memory management.

#### Acceptance Criteria

1. THE WasmRust_Compiler SHALL provide GC-native types: GcArray<T>, GcString, GcBox<T>, and GcRef<T>
2. WHEN using GC types, THE WasmRust_Compiler SHALL generate native WasmGC instructions without polyfills
3. THE WasmRust_Compiler SHALL enforce GC type safety at compile time preventing mixing of GC and non-GC references
4. WHEN interfacing between GC and ownership modes, THE WasmRust_Compiler SHALL require explicit conversion through safe APIs
5. THE WasmRust_Runtime SHALL provide automatic memory management for GC types with deterministic collection in supported environments

## Non-Goals

The following are explicitly outside the scope of WasmRust:

- **Browser Runtime Replacement**: WasmRust will not replace or modify browser WASM engines
- **Rust Language Fork**: Core Rust language semantics will not be changed; extensions will be library-based where possible, with minimal compiler extensions for WASM-specific features
- **Universal Threading**: Full threading support in all environments; single-threaded fallbacks are acceptable and expected
- **Universal Garbage Collection**: GC support is opt-in via #[wasm::gc] attribute; ownership mode remains the default
- **Backward Compatibility**: Breaking changes from existing wasm-bindgen workflows are acceptable for significant improvements
- **Universal Host Support**: Performance and capability guarantees apply only to explicitly supported host profiles

## Build Profiles

WasmRust SHALL support the following build profiles as contracts:

### Freestanding Profile
- No standard library
- No panic handling
- No allocator
- Custom entry points only
- Cranelift backend only
- Target: <2 KB binaries (improved from 15 KB to match MoonBit)

### Development Profile  
- Cranelift backend
- Fast compilation (<2s for 10k LOC, improved from 5s)
- Debug symbols included
- Incremental compilation enabled
- Deterministic output for reproducibility

### Release Profile
- LLVM backend with full optimizations
- Profile-guided optimization when available
- Maximum size and speed optimizations
- Reproducible builds with manifest hashing
- Toolchain version embedding

### GC Profile
- WasmGC-native compilation mode
- Garbage collection for marked types
- Simplified memory management
- Compatible with modern WASM runtimes

## Host Profile Support

WasmRust explicitly supports the following execution environments:

### Browser Profile
- **Threading**: SharedArrayBuffer + COOP/COEP headers required
- **JS Interop**: Direct calls with managed reference tables
- **Component Model**: Partial support via polyfills
- **Memory Regions**: Not supported
- **WasmGC**: Native support in modern browsers
- **Performance Target**: <100ns JS call overhead

### Node.js Profile  
- **Threading**: Worker threads
- **JS Interop**: Native bindings
- **Component Model**: Via polyfill
- **Memory Regions**: Not supported
- **WasmGC**: Via V8 engine support
- **Performance Target**: <50ns JS call overhead

### Wasmtime Profile
- **Threading**: wasi-threads
- **JS Interop**: Host functions
- **Component Model**: Full native support
- **Memory Regions**: Configurable by host
- **WasmGC**: Native support
- **Performance Target**: <25ns host call overhead

### Embedded Profile
- **Threading**: Not supported
- **JS Interop**: Not supported  
- **Component Model**: Partial (static linking only)
- **Memory Regions**: Not supported
- **WasmGC**: Not supported (ownership mode only)
- **Performance Target**: Minimal runtime overhead

## Security and Trust Model

### Compiler Security
- THE WasmRust_Compiler SHALL validate all input sources and reject malicious code patterns
- THE WasmRust_Compiler SHALL provide cryptographic signatures for all generated artifacts
- THE WasmRust_Compiler SHALL maintain isolation between compilation units

### Runtime Security  
- THE WasmRust_Runtime SHALL enforce Component Model security boundaries
- THE WasmRust_Runtime SHALL validate all cross-component calls at runtime
- THE WasmRust_Runtime SHALL prevent unauthorized memory access between components

### Registry Security
- THE WasmRust_Registry SHALL require cryptographic signatures for all published components
- THE WasmRust_Registry SHALL provide audit trails for all component downloads and updates
- THE WasmRust_Registry SHALL support revocation of compromised components

## Appendix A: Baseline Definitions

### C Baseline for Size Comparison (Requirement 1.2)

**Compiler**: Clang 18.0 with wasm32-wasi target
**Flags**: `-Oz -flto=full -Wl,--gc-sections`
**Features**:
- Allocator: dlmalloc (same as wasm crate default)
- No exceptions (equivalent to panic=abort)
- No C++ stdlib (equivalent to no_std)

### Benchmark Applications

1. **Hello World**: Print "Hello, World!" to console, exit
2. **JSON Parser**: Parse 10KB JSON document using standard library
3. **Image Filter**: Apply 3x3 convolution to 512x512 RGBA image
4. **Crypto Hash**: SHA-256 over 1MB buffer

### Size Measurement

- **Tool**: wasm-opt --strip-debug --strip-producers
- **Metric**: Final .wasm file size after all optimizations
- **Threshold**: WasmRust output ≤ 3.0x C baseline for equivalent functionality

## Appendix B: Component Model Support Matrix

| Feature | Browser | Node.js | Wasmtime | Embedded |
|---------|---------|---------|----------|----------|
| Import/Export Functions | ✅ Native | ✅ Native | ✅ Native | ✅ Static |
| Resources (handles) | ⚠️ Polyfill | ⚠️ Polyfill | ✅ Native | ❌ |
| Canonical ABI | ✅ JS impl | ✅ Native | ✅ Native | ⚠️ Subset |
| Dynamic Linking | ❌ | ⚠️ Via loader | ✅ Native | ❌ |
| Futures/Streams | ❌ | ❌ | ✅ Preview 2 | ❌ |
| WasmGC Support | ✅ Native | ✅ Native | ✅ Native | ❌ |

**Legend**: 
- ✅ Full native support
- ⚠️ Partial/polyfill implementation  
- ❌ Not supported

### Polyfill Scope

**Browser/Node.js Polyfills Include**:
- Component imports/exports via JS wrapper functions
- Resource handles via WeakMap-based lifetime management
- Basic Canonical ABI for primitive types

**Browser/Node.js Polyfills Exclude**:
- Native resource destructors (manual cleanup required)
- Async/streaming interfaces (callback-based alternatives)
- Cross-component memory sharing (serialization required)

## Appendix C: Security Threat Model

### Threats Addressed

1. **Supply Chain Attack (High Priority)**
   - **Mitigation**: Cryptographic signatures on all registry components
   - **Detection**: Audit logs + transparency logs (like Certificate Transparency)

2. **Memory Safety Violations (Critical)**
   - **Mitigation**: Rust type system + linear types for WASM resources
   - **Detection**: Property-based testing with invalid memory access patterns

3. **Component Isolation Bypass (High Priority)**
   - **Mitigation**: Component Model security boundaries
   - **Detection**: Fuzz testing cross-component calls

### Threats NOT Addressed

1. **Side-Channel Attacks (e.g., Spectre)**
   - **Rationale**: Requires browser mitigations, not compiler-level

2. **Denial of Service via Resource Exhaustion**
   - **Rationale**: Host responsibility to set limits

3. **Timing Attacks on Cryptographic Code**
   - **Rationale**: Use constant-time crypto libraries (not compiler's job)

## Appendix D: Compiler-Crate Contract Specification

### Purpose

This contract defines the semantic boundary between the WasmRust compiler extension and the `wasm` crate to:
- Prevent unsound compiler assumptions
- Enable aggressive WASM-specific optimization safely
- Preserve library-first evolution
- Allow `wasm` crate to work on stable rustc
- Make upstreaming to rustc possible

### Fundamental Principles

1. **Zero-Cost Invariant**: All public types in `wasm` crate are `#[repr(transparent)]` or `#[repr(C)]`, layout-compatible with WASM counterparts, and free of hidden allocations
2. **No Semantic Magic**: The `wasm` crate provides no behavior that requires compiler support
3. **Escape Hatch Rule**: Everything the compiler assumes must be reproducible by a pure library implementation

### Type-Level Contracts

#### ExternRef<T>
```rust
#[repr(transparent)]
pub struct ExternRef<T> {
    handle: u32,
    _marker: PhantomData<T>,
}
```

**Compiler MAY assume**:
- Maps 1:1 to WASM externref
- Is opaque and non-dereferenceable
- Does not alias Rust memory
- Has no Rust-visible interior mutability

**Compiler MUST NOT assume**:
- Any lifetime or ownership beyond Rust typing
- GC behavior or host identity stability
- That equal handles represent equal objects

#### SharedSlice<'a, T: Pod>
```rust
pub struct SharedSlice<'a, T: Pod> {
    ptr: NonNull<T>,
    len: usize,
    _lifetime: PhantomData<&'a [T]>,
}
```

**Compiler MAY assume**:
- `T: Pod` implies no pointers, no drop glue, bitwise movable
- Backed by linear memory, safe for concurrent reads
- Writes governed by Rust aliasing rules

**Compiler MUST NOT assume**:
- Atomicity unless explicitly requested
- That threads exist (may lower to single-threaded)
- Memory is shared across components unless proven

#### Pod Trait
```rust
unsafe trait Pod: Copy + 'static {}
```

**Compiler MAY assume**:
- Trivially copyable with no invalid bit patterns
- Safe for zero-copy serialization

**Compiler MUST NOT assume**:
- Endianness normalization or stable ABI across targets
- That all `Copy` types are `Pod`

### MIR Pattern Matching Rules

The compiler is only allowed to recognize and optimize specific MIR patterns:

1. **ExternRef Pass-Through Pattern**: `_1 = ExternRef::new(_2); _3 = call foo(_1)`
2. **SharedSlice Load Pattern**: `_elt = (*(_slice.ptr + idx))` where `T: Pod`
3. **Pod Copy Pattern**: `_2 = _1` where `_1: T, T: Pod`
4. **Component Boundary Call Pattern**: `_0 = call component::import_X(_1, _2)`

### Optimization Safety Rules

**Allowed Optimizations**:
- Inline through wasm wrappers
- Merge monomorphizations when proven safe
- Remove unused exports
- Replace library calls with intrinsics

**Forbidden Optimizations**:
- Change observable behavior
- Introduce UB if wasm crate is replaced
- Assume unsafe blocks are safe
- Break Rust aliasing or lifetime rules

### Verification Requirements

All optimizations relying on this contract MUST:
- Reference the specific invariant section relied upon
- Be testable by removing the optimization and compiling with stable rustc
- Observe identical semantics in both cases