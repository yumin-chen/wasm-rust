# WasmIR Specification v1.0

## Overview

WasmIR (WasmRust Intermediate Representation) is a stable intermediate representation between the rustc frontend and WasmRust backends. It serves as the boundary layer that encodes WASM-specific optimizations, ownership annotations, and capability hints while maintaining compatibility with WebAssembly's execution model.

WasmIR is designed to be the stable contract between the Rust frontend and multiple backend implementations (Cranelift for development, LLVM for release), enabling backend-agnostic optimizations and ensuring consistent semantics across compilation profiles.

## Design Goals

1. **Stability**: WasmIR provides a stable boundary between frontend and backends, allowing backend evolution without frontend changes
2. **Optimization**: Enables WASM-specific optimizations not available in standard Rust MIR, including thin monomorphization and streaming layout
3. **Safety**: Encodes ownership and type safety information for WebAssembly, preserving Rust's memory safety guarantees
4. **Capability**: Supports capability annotations for host profile optimization and graceful degradation
5. **Performance**: Enables efficient code generation for WebAssembly targets with predictable performance characteristics
6. **Debuggability**: Preserves source location information and variable names for debugging support

## Type System

### Core Types

WasmIR's type system is designed to map efficiently to WebAssembly while preserving Rust's type safety guarantees.

#### Value Types
```
i32    // 32-bit signed integer (maps to WASM i32)
i64    // 64-bit signed integer (maps to WASM i64)
f32    // 32-bit floating point (maps to WASM f32)
f64    // 64-bit floating point (maps to WASM f64)
```

#### Reference Types
```
externref<T>    // JavaScript object reference (type-safe, maps to WASM externref)
funcref         // Function reference (maps to WASM funcref)
```

#### Composite Types
```
array<T, N>     // Fixed-size array with compile-time known size
struct<T> {     // Structure with named fields
    field1: T1,
    field2: T2,
}
```

#### Linear Types (Experimental)
```
linear<T>       // Use-once semantics (consumed after use)
                // Enables zero-cost resource management
```

#### Capability-Annotated Types
```
capability<T, C>  // Type T with capability C required for access
                  // Used for host profile optimization
```

### Type Safety Rules

1. **Value Types**: All arithmetic operations are type-checked at compile time
2. **Reference Types**: Cannot be dereferenced directly; must use specific operations
3. **Linear Types**: Must be consumed exactly once; compiler enforces use-once semantics
4. **Capability Types**: Access requires runtime capability validation

### Memory Model

#### Linear Memory
- WebAssembly linear memory with bounds checking in development builds
- Bounds checking can be eliminated in release builds through static analysis
- Supports direct memory access through WASM load/store instructions
- Memory layout is deterministic and matches C ABI for interoperability

#### Shared Memory (Threading Profile)
- Thread-safe shared memory access when SharedArrayBuffer is available
- Atomic operations for synchronization
- Compile-time data race prevention through type system
- Graceful fallback to single-threaded execution when threading unavailable

#### Memory Regions (Host Profile Dependent)
- Geographic memory regions for data residency compliance
- Encryption annotations for sensitive data
- Host validates region capabilities at component load time
- Unsupported regions cause load-time failures, not runtime violations

## Instruction Set

WasmIR instructions are designed to map efficiently to WebAssembly while preserving high-level semantic information for optimization.

### Local Variable Operations
```
local.get N        // Get local variable N (maps to WASM local.get)
local.set N V      // Set local variable N to value V (maps to WASM local.set)
local.tee N V      // Set N to V and return V (maps to WASM local.tee)
```

### Constant Operations
```
i32.const V       // 32-bit integer constant
i64.const V       // 64-bit integer constant
f32.const V       // 32-bit float constant
f64.const V       // 64-bit float constant
```

### Binary Operations
```
i32.add L R       // Integer addition
i32.sub L R       // Integer subtraction
i32.mul L R       // Integer multiplication
i32.div_s L R     // Signed integer division
i32.div_u L R     // Unsigned integer division
i32.rem_s L R     // Signed remainder
i32.rem_u L R     // Unsigned remainder
i32.and L R       // Bitwise AND
i32.or L R        // Bitwise OR
i32.xor L R       // Bitwise XOR
i32.shl L R       // Left shift
i32.shr_s L R     // Right shift (signed)
i32.shr_u L R     // Right shift (unsigned)
i32.rotl L R      // Rotate left
i32.rotr L R      // Rotate right
i32.eq L R        // Equal
i32.ne L R        // Not equal
i32.lt_s L R      // Signed less than
i32.lt_u L R      // Unsigned less than
i32.le_s L R      // Signed less than or equal
i32.le_u L R      // Unsigned less than or equal
i32.gt_s L R      // Signed greater than
i32.gt_u L R      // Unsigned greater than
i32.ge_s L R      // Signed greater than or equal
i32.ge_u L R      // Unsigned greater than or equal
```

### Unary Operations
```
i32.clz V         // Count leading zeros
i32.ctz V         // Count trailing zeros
i32.popcnt V      // Population count (number of 1 bits)
i32.eqz V         // Equal to zero (returns i32: 1 if V == 0, 0 otherwise)
```

### Memory Operations
```
i32.load align=A offset=O        // Load 32-bit value from memory
i64.load align=A offset=O        // Load 64-bit value from memory
f32.load align=A offset=O        // Load 32-bit float from memory
f64.load align=A offset=O        // Load 64-bit float from memory
i32.load8_s align=A offset=O     // Load 8-bit signed, extend to 32-bit
i32.load8_u align=A offset=O     // Load 8-bit unsigned, extend to 32-bit
i32.load16_s align=A offset=O    // Load 16-bit signed, extend to 32-bit
i32.load16_u align=A offset=O    // Load 16-bit unsigned, extend to 32-bit
i64.load8_s align=A offset=O     // Load 8-bit signed, extend to 64-bit
i64.load8_u align=A offset=O     // Load 8-bit unsigned, extend to 64-bit
i64.load16_s align=A offset=O    // Load 16-bit signed, extend to 64-bit
i64.load16_u align=A offset=O    // Load 16-bit unsigned, extend to 64-bit
i64.load32_s align=A offset=O    // Load 32-bit signed, extend to 64-bit
i64.load32_u align=A offset=O    // Load 32-bit unsigned, extend to 64-bit

i32.store align=A offset=O       // Store 32-bit value to memory
i64.store align=A offset=O       // Store 64-bit value to memory
f32.store align=A offset=O       // Store 32-bit float to memory
f64.store align=A offset=O       // Store 64-bit float to memory
i32.store8 align=A offset=O      // Store lower 8 bits of 32-bit value
i32.store16 align=A offset=O     // Store lower 16 bits of 32-bit value
i64.store8 align=A offset=O      // Store lower 8 bits of 64-bit value
i64.store16 align=A offset=O     // Store lower 16 bits of 64-bit value
i64.store32 align=A offset=O     // Store lower 32 bits of 64-bit value

memory.size                      // Get current memory size in pages
memory.grow                      // Grow memory by specified pages
```

### Control Flow
```
br label                         // Unconditional branch to label
br_if cond label                 // Conditional branch if cond != 0
br_table index label0...labelN default  // Branch table (switch statement)
return                           // Return from function (no value)
return V                         // Return value V from function
unreachable                      // Unreachable instruction (trap)
```

### Function Operations
```
call func_index args...          // Direct function call
call_indirect type_index args... // Indirect function call through table
```

### Reference Type Operations
```
ref.null type                    // Create null reference of given type
ref.is_null ref                  // Test if reference is null
ref.func func_index              // Create function reference

externref.new value              // Create external reference from value
externref.get ref field          // Get field from external reference
externref.set ref field value    // Set field on external reference
externref.call ref method args   // Call method on external reference

funcref.new func_index           // Create function reference
funcref.call ref args            // Call function through reference
```

### Component Model Operations (WasmRust Extension)
```
component.import name type       // Import from component
component.export name value      // Export to component
component.instantiate module     // Instantiate component
component.call instance func args // Call component function
```

### Atomic Operations (Threading Profile)
```
i32.atomic.load align=A offset=O     // Atomic load
i64.atomic.load align=A offset=O     // Atomic load
i32.atomic.store align=A offset=O    // Atomic store
i64.atomic.store align=A offset=O    // Atomic store

i32.atomic.rmw.add align=A offset=O  // Atomic read-modify-write add
i32.atomic.rmw.sub align=A offset=O  // Atomic read-modify-write subtract
i32.atomic.rmw.and align=A offset=O  // Atomic read-modify-write AND
i32.atomic.rmw.or align=A offset=O   // Atomic read-modify-write OR
i32.atomic.rmw.xor align=A offset=O  // Atomic read-modify-write XOR
i32.atomic.rmw.xchg align=A offset=O // Atomic exchange

i32.atomic.cmpxchg align=A offset=O  // Atomic compare and exchange
i64.atomic.cmpxchg align=A offset=O  // Atomic compare and exchange

memory.atomic.notify offset=O        // Notify waiting threads
memory.atomic.wait32 offset=O        // Wait for notification (32-bit)
memory.atomic.wait64 offset=O        // Wait for notification (64-bit)
```

## Ownership Annotations

WasmIR extends WebAssembly with ownership annotations to preserve Rust's memory safety guarantees and enable zero-cost abstractions.

### Linear Types (Experimental)
```
linear.consume V           // Consume linear value (use exactly once)
linear.move V              // Move linear value (transfer ownership)
linear.clone V             // Clone linear value (if type supports cloning)
linear.drop V              // Explicitly drop linear value
linear.borrow V lifetime   // Borrow linear value for specified lifetime
```

### Ownership States
```
owned(V)                   // Value V is owned by current context
moved(V)                   // Value V has been moved (cannot be used)
borrowed(V, lifetime)      // Value V is borrowed for lifetime
consumed(V)                // Value V has been consumed (linear types)
```

### Lifetime Annotations
```
lifetime.begin 'a          // Begin lifetime 'a
lifetime.end 'a            // End lifetime 'a
lifetime.extend 'a 'b      // Extend lifetime 'a to 'b
```

## Capabilities

WasmIR includes a capability system for host profile optimization and graceful degradation.

### Core Capabilities
```
capability.threading         // Threading and parallel execution
capability.atomic           // Atomic memory operations
capability.shared_memory     // Shared memory access (SharedArrayBuffer)
capability.simd             // SIMD vector operations
```

### JavaScript Interop Capabilities
```
capability.js_interop       // JavaScript interoperability
capability.externref         // External reference support
capability.funcref          // Function reference support
capability.dom_access       // DOM manipulation capabilities
capability.web_apis         // Web API access (fetch, etc.)
```

### Memory Region Capabilities
```
capability.memory_region "eu-west-1"    // Geographic memory region
capability.memory_encryption "AES256-GCM" // Memory encryption
capability.memory_isolation             // Memory isolation between components
```

### Component Model Capabilities
```
capability.component_model   // Component Model support
capability.wit_bindgen      // WIT interface generation
capability.component_linking // Dynamic component linking
```

### Host Profile Capabilities
```
capability.browser_profile   // Browser execution environment
capability.nodejs_profile    // Node.js execution environment
capability.wasmtime_profile  // Wasmtime runtime environment
capability.embedded_profile  // Embedded/constrained environment
```

### Capability Operations
```
capability.check cap         // Check if capability is available
capability.require cap       // Require capability (fail if unavailable)
capability.optional cap      // Mark capability as optional
capability.fallback cap alt  // Provide fallback if capability unavailable
```

## Optimization Hints

WasmIR includes optimization hints to guide backend code generation and improve performance.

### Function-Level Hints
```
@inline(threshold=N)         // Inline function if size <= N instructions
@noinline                    // Never inline this function
@hot                         // Function is frequently called (optimize for speed)
@cold                        // Function is rarely called (optimize for size)
@pure                        // Function has no side effects (enable CSE)
@const                       // Function result depends only on arguments
```

### Call Convention Hints
```
call.fast func args          // Use fast calling convention
call.tail func args          // Tail call optimization
call.inline func args        // Inline call site
call.indirect_hint type func // Hint for indirect call optimization
```

### Memory Access Pattern Hints
```
memory.access_pattern streaming     // Sequential streaming access
memory.access_pattern random       // Random access pattern
memory.access_pattern temporal     // Temporal locality (cache-friendly)
memory.prefetch addr               // Prefetch memory location
```

### Loop Optimization Hints
```
loop.unroll factor=N         // Unroll loop N times
loop.vectorize width=N       // Vectorize loop with width N
loop.parallel               // Loop can be parallelized
loop.bound N                // Loop has known iteration count N
```

### Branch Prediction Hints
```
br_if.likely cond label     // Branch is likely taken
br_if.unlikely cond label   // Branch is unlikely taken
```

### Data Layout Hints
```
@align(N)                   // Align data to N-byte boundary
@packed                     // Pack struct fields tightly
@hot_data                   // Frequently accessed data
@cold_data                  // Rarely accessed data
```

## Validation Rules

WasmIR validation ensures type safety, memory safety, and correct control flow.

### Type Checking Rules

1. **Value Type Consistency**: All operations must be performed on compatible types
   - Arithmetic operations require numeric types
   - Comparison operations require compatible types
   - Memory operations require pointer or integer types

2. **Reference Type Safety**: Reference types must be handled correctly
   - ExternRef cannot be dereferenced directly
   - FuncRef must match expected signature when called
   - Null references must be checked before use

3. **Linear Type Enforcement**: Linear types must follow use-once semantics
   - Linear values must be consumed exactly once
   - Moved values cannot be accessed
   - Borrowed values must respect lifetime constraints

4. **Capability Validation**: Capability requirements must be satisfied
   - Required capabilities must be available in target host profile
   - Optional capabilities provide graceful degradation
   - Capability violations are compile-time errors

### Control Flow Validation

1. **Basic Block Structure**: All basic blocks must be well-formed
   - Each basic block ends with exactly one terminator
   - All branches target valid basic blocks
   - No unreachable code after terminators

2. **Function Structure**: Functions must have valid entry and exit points
   - Functions must have exactly one entry block
   - All execution paths must end with return or unreachable
   - Return types must match function signature

3. **Loop Structure**: Loops must be properly nested and bounded
   - Loop headers must dominate all loop blocks
   - Loop exits must be properly structured
   - Infinite loops must be explicitly marked

### Memory Safety Validation

1. **Bounds Checking**: Memory accesses must be within bounds
   - Array accesses must be within array bounds
   - Pointer arithmetic must not overflow
   - Stack accesses must be within stack frame

2. **Alignment Requirements**: Memory accesses must be properly aligned
   - Load/store operations must respect alignment constraints
   - Atomic operations require natural alignment
   - Unaligned accesses must be explicitly marked

3. **Lifetime Validation**: References must not outlive their referents
   - Borrowed references must not escape their lifetime
   - Dangling pointers are compile-time errors
   - Use-after-free is prevented by ownership system

### Component Model Validation

1. **Interface Compatibility**: Component interfaces must match
   - Import/export signatures must be compatible
   - Type definitions must be consistent
   - Version compatibility must be verified

2. **Resource Management**: Component resources must be properly managed
   - Resources must be explicitly closed
   - Resource leaks are compile-time errors
   - Cross-component resource sharing is validated

3. **Security Boundaries**: Component isolation must be maintained
   - Direct memory access between components is forbidden
   - All inter-component communication goes through interfaces
   - Capability boundaries are enforced

## Implementation Guidelines

### Backend Mapping Strategy

WasmIR is designed to map efficiently to WebAssembly while preserving optimization opportunities:

1. **Direct Instruction Mapping**: Most WasmIR instructions map 1:1 to WebAssembly opcodes
2. **Type Preservation**: WasmIR types map directly to WebAssembly value types and reference types
3. **Memory Layout Compatibility**: Memory layout matches WebAssembly linear memory model
4. **Function Signature Compatibility**: Function signatures match WebAssembly calling conventions

### Optimization Strategy

#### Instruction Selection
- Choose optimal WebAssembly instruction sequences based on target capabilities
- Exploit WebAssembly-specific features (SIMD, atomic operations, reference types)
- Apply peephole optimizations during instruction selection

#### Register Allocation
- Optimize for WebAssembly's stack-based execution model
- Minimize local variable usage through smart register allocation
- Exploit WebAssembly's unlimited local variables for spilling

#### Code Layout Optimization
- Arrange functions for optimal streaming compilation and instantiation
- Group hot functions together for better cache locality
- Separate cold code paths to reduce binary size

#### Dead Code Elimination
- Remove unused functions and data at the WasmIR level
- Eliminate unreachable basic blocks
- Optimize away unused local variables

### Error Handling Strategy

#### Compilation Errors
All compilation errors must be precise, actionable, and include source location information:

```rust
// Example error message format
error[E0001]: Invalid memory access
  --> src/main.rs:15:5
   |
15 |     *ptr = value;
   |     ^^^^ memory access may be out of bounds
   |
   = note: consider using bounds checking or safe memory access patterns
   = help: see https://wasmrust.dev/book/memory-safety for more information
```

#### Runtime Error Recovery
WasmIR supports graceful degradation for environment limitations:

1. **Capability Detection**: Runtime checks for host profile capabilities
2. **Fallback Mechanisms**: Automatic fallback to simpler implementations
3. **Error Propagation**: Clear error messages for unsupported operations
4. **Resource Cleanup**: Automatic cleanup on error conditions

### Performance Considerations

#### Code Size Optimization
- Prefer 32-bit operations over 64-bit when possible (smaller encoding)
- Use immediate values efficiently (avoid unnecessary constants)
- Optimize for binary size through function deduplication
- Eliminate unused code aggressively

#### Execution Speed Optimization
- Optimize hot paths identified through profiling
- Use efficient instruction sequences for common patterns
- Minimize memory traffic through smart data layout
- Exploit WebAssembly parallelism where available

#### Memory Usage Optimization
- Optimize stack allocation patterns
- Manage register pressure effectively
- Optimize memory layout for cache efficiency
- Avoid unnecessary memory allocations

### Security Considerations

#### Type Safety Enforcement
- Strong type checking at compile time
- Runtime type validation for reference types
- Memory bounds checking in development builds
- Component boundary enforcement

#### Memory Safety Guarantees
- Prevent buffer overflows through bounds checking
- Eliminate use-after-free through ownership system
- Prevent data races through type system constraints
- Validate all memory accesses

#### Component Isolation
- Enforce security boundaries between components
- Validate all inter-component communication
- Prevent direct memory access between components
- Audit component interface usage

## Migration Strategy

### From Rust MIR to WasmIR

The migration from Rust MIR to WasmIR follows a systematic transformation process:

1. **Function Signature Conversion**: Map Rust function signatures to WasmIR signatures
2. **Type System Mapping**: Convert Rust types to WasmIR types with capability annotations
3. **Basic Block Transformation**: Preserve control flow structure while adding WasmIR-specific annotations
4. **Instruction Lowering**: Convert MIR instructions to WasmIR instructions with optimization hints
5. **Debug Information Preservation**: Maintain source location and variable name information

### From WasmIR to WebAssembly

The compilation from WasmIR to WebAssembly is designed for efficiency and correctness:

1. **Instruction Selection**: Choose optimal WebAssembly instruction sequences
2. **Register Allocation**: Optimize local variable usage for WebAssembly's execution model
3. **Code Generation**: Generate efficient WebAssembly bytecode
4. **Optimization Passes**: Apply WebAssembly-specific optimizations
5. **Binary Generation**: Produce optimized WebAssembly modules

### Versioning and Compatibility

#### Version Management
- WasmIR version is tied to WasmRust compiler version
- Semantic versioning for breaking changes
- Backward compatibility guaranteed within major versions
- Clear migration path for breaking changes

#### Tool Support
- Automated migration tools for version upgrades
- Validation tools for compatibility checking
- Documentation for manual migration steps
- Community support for migration issues

## Tooling Support

### Development Tools

#### WasmIR Validator
- Comprehensive type checking and validation
- Control flow analysis and verification
- Memory safety validation
- Component model compliance checking
- Performance analysis and optimization suggestions

#### WasmIR Optimizer
- Instruction-level optimizations
- Control flow optimizations
- Dead code elimination
- Function inlining and specialization
- Profile-guided optimization support

#### WasmIR Debugger
- Source-level debugging support
- Variable inspection and modification
- Breakpoint support at instruction level
- Call stack visualization
- Memory layout inspection

### Analysis Tools

#### Performance Profiler
- Instruction-level performance analysis
- Hot path identification
- Memory access pattern analysis
- Cache performance analysis
- Optimization opportunity identification

#### Size Analyzer
- Binary size breakdown by function
- Code size attribution and analysis
- Optimization impact measurement
- Size regression detection
- Comparison with baseline implementations

#### Security Analyzer
- Memory safety violation detection
- Component boundary validation
- Capability usage analysis
- Security vulnerability scanning
- Compliance checking for security standards

## Conclusion

WasmIR represents a significant advancement in WebAssembly compilation technology, providing a stable, efficient, and safe intermediate representation that bridges the gap between high-level Rust code and optimized WebAssembly output. By incorporating ownership semantics, capability annotations, and comprehensive optimization hints, WasmIR enables the WasmRust compiler to generate high-performance WebAssembly code while maintaining Rust's safety guarantees.

The specification is designed to be implementable, testable, and extensible, providing a solid foundation for the WasmRust compiler ecosystem. Through careful attention to performance, safety, and developer experience, WasmIR enables the next generation of WebAssembly applications with predictable performance characteristics and robust security properties.

## Appendix A: Instruction Reference

### Complete Instruction Set

This appendix provides a complete reference for all WasmIR instructions, including their semantics, WebAssembly mapping, and usage examples.

[Detailed instruction reference would continue here with comprehensive documentation for each instruction...]

## Appendix B: Type System Reference

### Complete Type System

This appendix provides a complete reference for the WasmIR type system, including type rules, conversion semantics, and safety properties.

[Detailed type system reference would continue here...]

## Appendix C: Validation Rules Reference

### Complete Validation Rules

This appendix provides a complete reference for WasmIR validation rules, including formal specifications and implementation guidelines.

[Detailed validation rules would continue here...]

## Appendix D: Performance Benchmarks

### Benchmark Results

This appendix provides performance benchmark results comparing WasmIR-based compilation with other approaches.

[Detailed benchmark results would continue here...]

## Examples

This section provides comprehensive examples of WasmIR usage for common programming patterns.

### Example 1: Simple Arithmetic Function

```rust
// Rust source: fn add(a: i32, b: i32) -> i32 { a + b }

// WasmIR representation:
function add(i32, i32) -> i32 {
    // Function signature: (i32, i32) -> i32
    // Local variables: %0 = param0, %1 = param1, %2 = result
    
    block_0:
        %2 = i32.add %0, %1          // Add parameters
        return %2                     // Return result
}
```

**Rust Implementation:**
```rust
let signature = Signature {
    params: vec![Type::I32, Type::I32],
    returns: Some(Type::I32),
};

let mut func = WasmIR::new("add".to_string(), signature);
let result_local = func.add_local(Type::I32);

let instructions = vec![
    Instruction::BinaryOp {
        op: BinaryOp::Add,
        left: Operand::Local(0),   // First parameter
        right: Operand::Local(1),  // Second parameter
    },
    Instruction::LocalSet {
        index: result_local,
        value: Operand::Local(0),  // Result of addition
    },
];

let terminator = Terminator::Return {
    value: Some(Operand::Local(result_local)),
};

func.add_basic_block(instructions, terminator);
```

### Example 2: Memory Access Function

```rust
// Rust source: fn load_and_increment(ptr: *mut i32) -> i32 {
//     let value = unsafe { *ptr };
//     unsafe { *ptr = value + 1 };
//     value
// }

// WasmIR representation:
function load_and_increment(i32) -> i32 {
    // %0 = ptr (parameter), %1 = loaded_value, %2 = incremented
    
    block_0:
        %1 = i32.load %0 align=4 offset=0    // Load value from memory
        %2 = i32.add %1, i32.const(1)        // Increment value
        i32.store %0, %2 align=4 offset=0    // Store back to memory
        return %1                             // Return original value
}
```

**Rust Implementation:**
```rust
let signature = Signature {
    params: vec![Type::I32],  // ptr
    returns: Some(Type::I32),
};

let mut func = WasmIR::new("load_and_increment".to_string(), signature);
let loaded_local = func.add_local(Type::I32);
let incremented_local = func.add_local(Type::I32);

let instructions = vec![
    Instruction::MemoryLoad {
        address: Operand::Local(0),  // ptr parameter
        ty: Type::I32,
        align: Some(4),
        offset: 0,
    },
    Instruction::LocalSet {
        index: loaded_local,
        value: Operand::Local(0),  // Result of load
    },
    Instruction::BinaryOp {
        op: BinaryOp::Add,
        left: Operand::Local(loaded_local),
        right: Operand::Constant(Constant::I32(1)),
    },
    Instruction::LocalSet {
        index: incremented_local,
        value: Operand::Local(0),  // Result of add
    },
    Instruction::MemoryStore {
        address: Operand::Local(0),  // ptr parameter
        value: Operand::Local(incremented_local),
        ty: Type::I32,
        align: Some(4),
        offset: 0,
    },
];

let terminator = Terminator::Return {
    value: Some(Operand::Local(loaded_local)),
};

func.add_basic_block(instructions, terminator);
```

### Example 3: Control Flow with Conditional

```rust
// Rust source: fn max(a: i32, b: i32) -> i32 {
//     if a > b { a } else { b }
// }

// WasmIR representation:
function max(i32, i32) -> i32 {
    // %0 = a, %1 = b, %2 = condition, %3 = result
    
    block_0:
        %2 = i32.gt_s %0, %1              // Compare a > b
        br_if %2, block_1, block_2        // Branch based on condition
    
    block_1:  // a > b is true
        %3 = %0                           // result = a
        br block_3                        // Jump to return
    
    block_2:  // a > b is false
        %3 = %1                           // result = b
        br block_3                        // Jump to return
    
    block_3:  // return block
        return %3
}
```

**Rust Implementation:**
```rust
let signature = Signature {
    params: vec![Type::I32, Type::I32],
    returns: Some(Type::I32),
};

let mut func = WasmIR::new("max".to_string(), signature);
let condition_local = func.add_local(Type::I32);
let result_local = func.add_local(Type::I32);

// Block 0: Comparison and branch
let block0_instructions = vec![
    Instruction::BinaryOp {
        op: BinaryOp::Gt,
        left: Operand::Local(0),   // a
        right: Operand::Local(1),  // b
    },
    Instruction::LocalSet {
        index: condition_local,
        value: Operand::Local(0),  // Result of comparison
    },
];
let block0_terminator = Terminator::Branch {
    condition: Operand::Local(condition_local),
    then_block: BlockId(1),
    else_block: BlockId(2),
};
func.add_basic_block(block0_instructions, block0_terminator);

// Block 1: a > b is true
let block1_instructions = vec![
    Instruction::LocalSet {
        index: result_local,
        value: Operand::Local(0),  // result = a
    },
];
let block1_terminator = Terminator::Jump { target: BlockId(3) };
func.add_basic_block(block1_instructions, block1_terminator);

// Block 2: a > b is false
let block2_instructions = vec![
    Instruction::LocalSet {
        index: result_local,
        value: Operand::Local(1),  // result = b
    },
];
let block2_terminator = Terminator::Jump { target: BlockId(3) };
func.add_basic_block(block2_instructions, block2_terminator);

// Block 3: Return
let block3_instructions = vec![];
let block3_terminator = Terminator::Return {
    value: Some(Operand::Local(result_local)),
};
func.add_basic_block(block3_instructions, block3_terminator);
```

### Example 4: JavaScript Interop with ExternRef

```rust
// Rust source: 
// #[wasm::export]
// fn process_js_object(obj: ExternRef<JsObject>) -> i32 {
//     obj.call("getValue", ()).unwrap_or(0)
// }

// WasmIR representation:
@capability(js_interop)
function process_js_object(externref<JsObject>) -> i32 {
    // %0 = obj (ExternRef parameter), %1 = result
    
    block_0:
        capability.check js_interop       // Verify JS interop capability
        %1 = externref.call %0, "getValue", []  // Call JS method
        br_if externref.is_null(%1), block_error, block_success
    
    block_success:
        %2 = externref.cast %1, i32       // Cast result to i32
        return %2
    
    block_error:
        return i32.const(0)               // Return default value
}
```

**Rust Implementation:**
```rust
let signature = Signature {
    params: vec![Type::ExternRef("JsObject".to_string())],
    returns: Some(Type::I32),
};

let mut func = WasmIR::new("process_js_object".to_string(), signature);
func.add_capability(Capability::JsInterop);

let result_local = func.add_local(Type::I32);
let js_result_local = func.add_local(Type::ExternRef("any".to_string()));

// Block 0: Capability check and JS method call
let block0_instructions = vec![
    Instruction::CapabilityCheck {
        capability: Capability::JsInterop,
    },
    Instruction::JSMethodCall {
        object: Operand::Local(0),  // obj parameter
        method: "getValue".to_string(),
        args: vec![],
        return_type: Some(Type::ExternRef("any".to_string())),
    },
    Instruction::LocalSet {
        index: js_result_local,
        value: Operand::Local(0),  // Result of JS call
    },
    Instruction::ExternRefIsNull {
        externref: Operand::Local(js_result_local),
    },
];
let block0_terminator = Terminator::Branch {
    condition: Operand::Local(0),  // Result of is_null check
    then_block: BlockId(2),        // error block
    else_block: BlockId(1),        // success block
};
func.add_basic_block(block0_instructions, block0_terminator);

// Block 1: Success - cast and return
let block1_instructions = vec![
    Instruction::ExternRefCast {
        externref: Operand::Local(js_result_local),
        target_type: Type::I32,
    },
    Instruction::LocalSet {
        index: result_local,
        value: Operand::Local(0),  // Result of cast
    },
];
let block1_terminator = Terminator::Return {
    value: Some(Operand::Local(result_local)),
};
func.add_basic_block(block1_instructions, block1_terminator);

// Block 2: Error - return default
let block2_instructions = vec![];
let block2_terminator = Terminator::Return {
    value: Some(Operand::Constant(Constant::I32(0))),
};
func.add_basic_block(block2_instructions, block2_terminator);
```

### Example 5: Threading with Atomic Operations

```rust
// Rust source:
// fn atomic_increment(counter: &AtomicI32) -> i32 {
//     counter.fetch_add(1, Ordering::SeqCst)
// }

// WasmIR representation:
@capability(threading, atomic_memory)
function atomic_increment(i32) -> i32 {
    // %0 = counter_ptr, %1 = old_value
    
    block_0:
        capability.check atomic_memory    // Verify atomic capability
        %1 = i32.atomic.rmw.add %0, i32.const(1), seq_cst
        return %1                         // Return old value
}
```

**Rust Implementation:**
```rust
let signature = Signature {
    params: vec![Type::I32],  // counter pointer
    returns: Some(Type::I32),
};

let mut func = WasmIR::new("atomic_increment".to_string(), signature);
func.add_capability(Capability::Threading);
func.add_capability(Capability::AtomicMemory);

let old_value_local = func.add_local(Type::I32);

let instructions = vec![
    Instruction::CapabilityCheck {
        capability: Capability::AtomicMemory,
    },
    Instruction::AtomicOp {
        op: AtomicOp::Add,
        address: Operand::Local(0),  // counter_ptr
        value: Operand::Constant(Constant::I32(1)),
        order: MemoryOrder::SeqCst,
    },
    Instruction::LocalSet {
        index: old_value_local,
        value: Operand::Local(0),  // Result of atomic add
    },
];

let terminator = Terminator::Return {
    value: Some(Operand::Local(old_value_local)),
};

func.add_basic_block(instructions, terminator);
```

### Example 6: Component Model Export

```rust
// Rust source:
// #[wasm::component_export]
// fn compute_hash(data: &[u8]) -> [u8; 32] {
//     // Hash computation logic
// }

// WasmIR representation:
@capability(component_model)
@export("compute_hash")
function compute_hash(i32, i32) -> i32 {
    // %0 = data_ptr, %1 = data_len, %2 = result_ptr
    
    block_0:
        capability.check component_model  // Verify component capability
        %2 = call hash_sha256, %0, %1     // Call internal hash function
        component.export "compute_hash", %2  // Export result
        return %2
}
```

**Rust Implementation:**
```rust
let signature = Signature {
    params: vec![Type::I32, Type::I32],  // data_ptr, data_len
    returns: Some(Type::I32),            // result_ptr
};

let mut func = WasmIR::new("compute_hash".to_string(), signature);
func.add_capability(Capability::ComponentModel);

let result_local = func.add_local(Type::I32);

let instructions = vec![
    Instruction::CapabilityCheck {
        capability: Capability::ComponentModel,
    },
    Instruction::Call {
        func_ref: 1,  // hash_sha256 function index
        args: vec![
            Operand::Local(0),  // data_ptr
            Operand::Local(1),  // data_len
        ],
    },
    Instruction::LocalSet {
        index: result_local,
        value: Operand::Local(0),  // Result of hash call
    },
];

let terminator = Terminator::Return {
    value: Some(Operand::Local(result_local)),
};

func.add_basic_block(instructions, terminator);
```

### Example 7: Linear Types for Resource Management

```rust
// Rust source:
// fn use_file_handle(handle: LinearFileHandle) -> Result<String, Error> {
//     let content = handle.read_all()?;  // Consumes handle
//     Ok(content)
// }

// WasmIR representation:
function use_file_handle(linear<FileHandle>) -> i32 {
    // %0 = handle (linear), %1 = content_ptr, %2 = result
    
    block_0:
        ownership.check owned(%0)         // Verify handle is owned
        %1 = call file_read_all, %0       // Call read (consumes handle)
        linear.consume %0                 // Mark handle as consumed
        ownership.set consumed(%0)        // Update ownership state
        br_if call_succeeded(%1), block_success, block_error
    
    block_success:
        %2 = i32.const(0)                 // Success code
        return %2
    
    block_error:
        %2 = i32.const(-1)                // Error code
        return %2
}
```

**Rust Implementation:**
```rust
let signature = Signature {
    params: vec![Type::Linear {
        inner_type: Box::new(Type::ExternRef("FileHandle".to_string())),
    }],
    returns: Some(Type::I32),
};

let mut func = WasmIR::new("use_file_handle".to_string(), signature);

let content_local = func.add_local(Type::I32);
let result_local = func.add_local(Type::I32);

// Add ownership annotation
func.add_ownership_annotation(OwnershipAnnotation {
    variable: 0,  // handle parameter
    state: OwnershipState::Owned,
    source_location: SourceLocation {
        file: "example.rs".to_string(),
        line: 1,
        column: 1,
    },
});

let block0_instructions = vec![
    Instruction::Call {
        func_ref: 2,  // file_read_all function
        args: vec![Operand::Local(0)],  // handle
    },
    Instruction::LocalSet {
        index: content_local,
        value: Operand::Local(0),  // Result of read
    },
    Instruction::LinearOp {
        op: LinearOp::Consume,
        value: Operand::Local(0),  // Consume handle
    },
];

// Add branch based on call success (simplified)
let block0_terminator = Terminator::Branch {
    condition: Operand::Local(content_local),
    then_block: BlockId(1),  // success
    else_block: BlockId(2),  // error
};

func.add_basic_block(block0_instructions, block0_terminator);

// Success block
let block1_instructions = vec![
    Instruction::LocalSet {
        index: result_local,
        value: Operand::Constant(Constant::I32(0)),
    },
];
let block1_terminator = Terminator::Return {
    value: Some(Operand::Local(result_local)),
};
func.add_basic_block(block1_instructions, block1_terminator);

// Error block
let block2_instructions = vec![
    Instruction::LocalSet {
        index: result_local,
        value: Operand::Constant(Constant::I32(-1)),
    },
];
let block2_terminator = Terminator::Return {
    value: Some(Operand::Local(result_local)),
};
func.add_basic_block(block2_instructions, block2_terminator);
```

## Migration Strategy

### From Rust MIR
1. Convert function signatures and types
2. Map basic blocks and control flow
3. Preserve debug information and source locations
4. Add capability annotations from attributes

### To WebAssembly
1. Direct instruction mapping where possible
2. Optimize for WebAssembly execution model
3. Apply WASM-specific optimizations
4. Generate efficient code layout

### Versioning
1. WasmIR version is tied to WasmRust compiler version
2. Backward compatibility guaranteed within major versions
3. Migration path provided for breaking changes
4. Tool support for automated migration

## Tooling Support

### Validation
- WasmIR validator for type checking
- Control flow analysis tools
- Memory safety verification
- Component model compliance checking

### Optimization
- Instruction selector for different targets
- Register allocator for WebAssembly
- Code layout optimizer
- Dead code eliminator

### Debugging
- Source location preservation
- Variable naming preservation
- Basic block visualization
- Instruction-level debugging

## Performance Considerations

### Code Size
- Prefer 32-bit operations where possible
- Use immediate values efficiently
- Optimize for binary size
- Eliminate unused code aggressively

### Execution Speed
- Optimize hot paths aggressively
- Use efficient instruction sequences
- Minimize memory traffic
- Exploit WebAssembly parallelism

### Memory Usage
- Stack allocation optimization
- Register pressure management
- Memory layout optimization
- Garbage collection avoidance

## Security Considerations

### Type Safety
- Strong type enforcement
- Memory bounds checking
- Reference type validation
- Component boundary enforcement

### Code Injection
- Instruction validation
- Control flow verification
- Memory access validation
- Component isolation

### Side Channels
- Speculative execution prevention
- Constant-time operations
- Memory access pattern randomization
- Component isolation enforcement
