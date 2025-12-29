# SAFETY.md

**WasmRust – `crates/wasm` Core Safety & Invariants**

This document defines **soundness, safety invariants, and compiler contracts** for the `wasm` crate.

It is **normative**:
If behavior contradicts this document, the implementation is unsound.

---

## Scope

This document covers all `unsafe` code and all types whose correctness depends on **external invariants**, including:

* `ExternRef<T>`
* `FuncRef<Args, Ret>`
* `SharedSlice<'a, T>`
* `Pod`
* Any marker traits or attributes affecting aliasing, ownership, or concurrency

The `wasm` crate is designed to be:

* `no_std` compatible
* Runtime-free where possible
* Compiler-agnostic
* Safe to use on **stable Rust**

---

## Trust Model

### Trusted

* Rust type system and borrow checker
* Rust aliasing and lifetime rules
* WASM execution model (linear memory, reference types)
* Host environment correctness *only when explicitly capability-checked*

### Untrusted

* JavaScript hosts
* Foreign WASM components
* Raw pointers crossing ABI boundaries
* Host-provided memory or references

---

## General Safety Principles

1. **No hidden runtime checks**

   * Safety is enforced by types and invariants, not dynamic guards.

2. **Unsafe code is localized**

   * Every `unsafe` block must be justified by a documented invariant.

3. **Zero-cost means zero semantic cost**

   * Abstractions may not introduce hidden allocations, refcounting, or locks.

4. **Compiler-visible semantics**

   * Types are intentionally simple so compilers may pattern-match them.

---

## Type-Level Safety Contracts

---

## `ExternRef<T>`

```rust
#[repr(transparent)]
pub struct ExternRef<T: ?Sized> {
    handle: u32,
    _phantom: PhantomData<T>,
}
```

### Purpose

Represents a **typed handle** to a host-managed object (e.g., JavaScript object).

---

### Safety Invariants

The following **must always hold**:

1. `handle` refers to a valid entry in host reference table **or is null-equivalent**
2. The host guarantees:

   * Stable identity for lifetime of handle
   * No reuse of handle value for a different object
3. `T` encodes *logical type*, not memory layout
4. Dropping `ExternRef<T>` does **not** free host memory unless explicitly documented

---

### User Obligations

* Must not fabricate `ExternRef<T>` from arbitrary integers
* Must not transmute between `ExternRef<U>` and `ExternRef<V>`
* Must respect lifetime and ownership rules imposed by APIs

---

### Undefined Behavior

Any of the following is **UB**:

* Using a handle after the host invalidates it
* Passing an `ExternRef<T>` to a host expecting a different logical type
* Forging handles via `mem::transmute` or raw casts

---

### Compiler Contract

The compiler **may assume**:

* `ExternRef<T>` is a plain integer handle
* Cloning is cheap and does not allocate
* Equality comparisons are constant-time
* No aliasing with linear memory

The compiler **must not assume**:

* That `ExternRef<T>` points to linear memory
* That it can dereference or inspect host data

---

## `FuncRef<Args, Ret>`

```rust
#[repr(transparent)]
pub struct FuncRef<Args, Ret> {
    index: u32,
    _phantom: PhantomData<(Args, Ret)>,
}
```

### Purpose

Represents a **callable host or WASM function reference**.

---

### Safety Invariants

* `index` refers to a valid function entry
* Signature compatibility is enforced at creation time
* Calling a `FuncRef` uses the correct ABI

---

### User Obligations

* Must ensure function signature matches call site
* Must not use after function is invalidated
* Must respect ABI calling conventions

---

### Undefined Behavior

* Calling a `FuncRef` with an incompatible signature
* Using a function handle after it is invalidated
* Calling with wrong argument types or count

---

### Compiler Contract

The compiler may:

* Treat `FuncRef` as opaque and non-aliasing
* Inline or optimize calls if signature is statically known
* Represent as a raw `i32` in WASM function tables

The compiler must not:

* Assume function contents are safe
* Skip ABI validation
* Treat as data pointer

---

## `Pod` Trait

```rust
pub unsafe trait Pod: Copy + 'static {
    /// Returns true if the type is valid for zero-copy sharing
    fn is_valid_for_sharing() -> bool {
        true
    }
}
```

### Purpose

Marks types that are safe for **bitwise copying and shared access**.

---

### Safety Invariants

For any `T: Pod`, all must hold:

1. No padding bytes with semantic meaning
2. No pointers or references
3. No interior mutability
4. No drop glue
5. Valid for all bit patterns

---

### User Obligations

* Only implement `Pod` for types meeting **all invariants**
* Prefer derive macros or sealed implementations where possible
* Must not implement for types with references or interior mutability

---

### Undefined Behavior

* Implementing `Pod` for a type with references or interior mutability
* Sharing non-`Pod` data across threads or components
* Assuming all bit patterns are valid for non-`Pod` types

---

### Compiler Contract

The compiler may:

* Assume `T: Pod` can be freely copied
* Assume no data races occur for shared reads
* Optimize memory operations aggressively
* Treat as bitwise movable

The compiler must not:

* Assume zeroed memory is valid unless proven
* Insert hidden validation or runtime checks

---

## `SharedSlice<'a, T>`

```rust
pub struct SharedSlice<'a, T: Pod> {
    ptr: NonNull<T>,
    len: usize,
    _phantom: PhantomData<&'a [T]>,
}
```

### Purpose

Provides **read-only shared access** to linear memory across threads or components.

---

### Safety Invariants

1. `ptr` points to valid linear memory for `len` elements
2. Memory is immutable for lifetime `'a`
3. `T: Pod` guarantees race-free shared access
4. No aliasing mutable access exists during `'a`

---

### User Obligations

* Must not mutate memory backing a `SharedSlice`
* Must not extend lifetime beyond memory validity
* Must respect host threading capability checks
* Must ensure `T: Pod` when creating shared slices

---

### Undefined Behavior

* Mutating shared memory while a `SharedSlice` exists
* Constructing from invalid or dangling pointers
* Using `SharedSlice` in unsupported threading environments
* Creating with non-`Pod` types

---

### Compiler Contract

The compiler may assume:

* `SharedSlice<'a, T>` behaves like `&'a [T]`
* No writes occur through any alias during `'a`
* Reads are race-free due to `T: Pod`
* Memory layout is contiguous and predictable

This enables:

* Vectorization
* Load hoisting
* Bounds-check elimination
* Cross-thread read reordering

The compiler must not:

* Assume mutable access is safe
* Skip bounds checking based on trust
* Assume memory remains valid beyond `'a`

---

## Threading & Host Capabilities

Threading-dependent abstractions **must**:

* Be feature-gated
* Perform explicit host capability detection
* Fail safely (compile-time or runtime) when unsupported

It is **unsound** to assume threading support implicitly.

### Capability Detection Rules

* `get_host_capabilities()` must be called before threading operations
* `HostCapabilities::threading` must be checked before mutable access
* Unsupported environments must return errors, not panic

---

## Unsafe Code Policy

Every `unsafe` block must:

* Reference the specific invariant it relies on
* Be minimal and localized
* Be auditable without reading external code

Example of acceptable unsafe block:

```rust
// SAFETY: T: Pod guarantees no interior mutability or drop glue.
// The pointer is valid for 'len elements as guaranteed by caller.
unsafe { 
    slice::from_raw_parts(ptr, len) 
}
```

Example of unacceptable unsafe block:

```rust
unsafe {
    // No justification - this would be rejected
    some_operation()
}
```

---

## Error Handling Policy

All error types must implement:

* `Debug` for debugging
* `Clone` for error propagation
* `PartialEq` for testing
* `Display` for user-facing errors

Error conversions must be explicit:

* `From<ErrorType>` implementations must be sound
* No implicit error conversions
* Error variants must represent distinct failure modes

---

## Testing Requirements

Safety is validated via:

* Property-based tests (adversarial inputs)
* Cross-thread stress tests (when enabled)
* Bounds-checking validation
* ABI conformance tests
* MIR inspection for compiler patterns

Passing tests **do not weaken** invariants.

---

## Change Policy

Any change that affects:

* Layout of any public type
* Safety invariants
* Compiler assumptions
* Error handling behavior

**must** update this document and bump the crate version.

---

## Summary

This crate is safe **only because**:

* Invariants are explicit and documented
* Unsafe code is justified and minimal
* The compiler contract is narrow and precise
* Property tests validate all critical invariants

If you cannot explain why an unsafe block is sound **using only this document**, it is a bug.