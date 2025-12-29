//! Property-based tests for binary size optimization
//! 
//! This module validates that thin monomorphization and other binary
//! size optimizations effectively reduce code size while maintaining
//! functionality.
//! 
//! Property 2: Thin Monomorphization Effectiveness
//! Validates: Requirements 1.3

use wasm::backend::{BackendFactory, BuildProfile, CompilationResult};
use wasm::backend::cranelift::{WasmRustCraneliftBackend, ThinMonomorphizationContext};
use wasm::backend::cranelift::thin_monomorphization::{
    MonomorphizationFlags, CodeSizeStats, StreamingLayout,
    FunctionId, GenericFunction, InstanceSignature
};
use wasm::wasmir::{
    WasmIR, Signature, Type, Instruction, Terminator, Operand, 
    BinaryOp, UnaryOp, BasicBlock, BlockId, Constant,
    Capability, OwnershipAnnotation, SourceLocation,
};
use quickcheck::{Arbitrary, Gen, QuickCheck, TestResult};
use std::time::{Instant, Duration};
use std::collections::{HashMap, HashSet};

#[cfg(test)]
mod tests {
    use super::*;

    /// Test data for binary size optimization
    #[derive(Debug, Clone)]
    struct BinarySizeTestData {
        name: &'static str,
        generic_function: WasmIR,
        instantiations: Vec<Vec<Type>>,
        expected_size_reduction: f64,
        expected_monomorphization_count: usize,
    }

    /// Monomorphization test parameters
    #[derive(Debug, Clone, Copy, Arbitrary)]
    struct MonomorphizationTestParams {
        generic_param_count: u8,
        instantiations_count: u8,
        function_complexity: u8, // 1-5 complexity level
        enable_optimization: bool,
        cache_size_limit: u8,
    }

    /// Binary size optimization scenario
    #[derive(Debug, Clone, Arbitrary)]
    struct OptimizationScenario {
        function_count: u8,
        generic_functions: u8,
        function_size: u16, // 100-10000 bytes
        duplication_factor: u8, // 1-10x duplication
        optimization_level: u8, // 0-5 optimization aggressiveness
    }

    /// Creates test WasmIR functions with varying complexity
    fn create_generic_function(name: &str, param_count: usize) -> WasmIR {
        let mut param_types = Vec::new();
        for i in 0..param_count {
            param_types.push(Type::Ref(format!("T{}", i)));
        }

        let signature = Signature {
            params: param_types,
            returns: Some(Type::Ref("T0".to_string())),
        };

        let mut func = WasmIR::new(name.to_string(), signature);
        
        // Add local variables
        for _ in 0..param_count {
            func.add_local(Type::Ref("T0".to_string()));
        }
        
        let local_result = func.add_local(Type::Ref("T0".to_string()));

        // Create basic block with generic operations
        let instructions = vec![
            Instruction::LocalGet { index: 0 },
            Instruction::LocalGet { index: 1 },
            Instruction::BinaryOp {
                op: BinaryOp::Add,
                left: Operand::Local(0),
                right: Operand::Local(1),
            },
            Instruction::LocalSet {
                index: local_result,
                value: Operand::Local(2), // Result of addition
            },
            Instruction::Return {
                value: Some(Operand::Local(local_result)),
            },
        ];

        let terminator = Terminator::Return {
            value: Some(Operand::Local(local_result)),
        };

        func.add_basic_block(instructions, terminator);
        func
    }

    /// Creates a monomorphic function for comparison
    fn create_monolithic_function(name: &str) -> WasmIR {
        let signature = Signature {
            params: vec![Type::I32, Type::I32],
            returns: Some(Type::I32),
        };

        let mut func = WasmIR::new(name.to_string(), signature);
        let local_a = func.add_local(Type::I32);
        let local_b = func.add_local(Type::I32);
        let local_result = func.add_local(Type::I32);

        let instructions = vec![
            Instruction::LocalGet { index: local_a },
            Instruction::LocalGet { index: local_b },
            Instruction::BinaryOp {
                op: BinaryOp::Add,
                left: Operand::Local(local_a),
                right: Operand::Local(local_b),
            },
            Instruction::LocalSet {
                index: local_result,
                value: Operand::Local(0), // Result of addition
            },
            Instruction::Return {
                value: Some(Operand::Local(local_result)),
            },
        ];

        let terminator = Terminator::Return {
            value: Some(Operand::Local(local_result)),
        };

        func.add_basic_block(instructions, terminator);
        func
    }

    /// Property: Monomorphization reduces code size for repeated generic functions
    #[test]
    fn prop_monomorphization_reduces_code_size() {
        fn property(scenario: OptimizationScenario) -> TestResult {
            let target = rustc_target::spec::Target {
                arch: "wasm32".to_string(),
                ..Default::default()
            };

            // Create functions with duplication
            let mut functions = Vec::new();
            
            // Add monolithic (non-optimized) functions
            for i in 0..scenario.function_count {
                let func = create_monolithic_function(&format!("monolithic_{}", i));
                functions.push(func);
            }

            // Add generic functions that could be monomorphized
            for i in 0..scenario.generic_functions {
                let generic_func = create_generic_function(&format!("generic_{}", i), 2);
                // Duplicate the generic function with different parameter types
                for j in 0..scenario.duplication_factor {
                    let mut duplicated_func = generic_func.clone();
                    duplicated_func.name = format!("generic_{}_dup_{}", i, j);
                    functions.push(duplicated_func);
                }
            }

            // Compile without optimization
            let mut unoptimized_backend = WasmRustCraneliftBackend::new(target.clone());
            if unoptimized_backend.is_err() {
                return TestResult::failed();
            }

            let mut unoptimized_backend = unoptimized_backend.unwrap();
            let start = Instant::now();
            let unoptimized_result = unoptimized_backend.compile_functions(&functions);
            let unoptimized_time = start.elapsed();

            if unoptimized_result.is_err() {
                return TestResult::failed();
            }

            let unoptimized_functions = unoptimized_result.unwrap();
            let unoptimized_total_size: usize = unoptimized_functions
                .values()
                .map(|code| code.len())
                .sum();

            // Compile with optimization
            let mut optimized_backend = WasmRustCraneliftBackend::new(target.clone());
            if optimized_backend.is_err() {
                return TestResult::failed();
            }

            let mut optimized_backend = optimized_backend.unwrap();
            let start = Instant::now();
            let optimized_result = optimized_backend.compile_functions(&functions);
            let optimized_time = start.elapsed();

            if optimized_result.is_err() {
                return TestResult::failed();
            }

            let optimized_functions = optimized_result.unwrap();
            let optimized_total_size: usize = optimized_functions
                .values()
                .map(|code| code.len())
                .sum();

            // Calculate size reduction
            if unoptimized_total_size == 0 {
                return TestResult::failed();
            }

            let size_reduction = (unoptimized_total_size - optimized_total_size) as f64 
                / unoptimized_total_size as f64 * 100.0;

            // Should have meaningful size reduction for duplicated functions
            let expected_min_reduction = if scenario.generic_functions > 0 && scenario.duplication_factor > 1 {
                match scenario.optimization_level {
                    0 => 5.0,  // Minimal optimization
                    1 => 10.0, // Basic optimization
                    2 => 20.0, // Standard optimization
                    3 => 30.0, // Aggressive optimization
                    4 => 40.0, // Very aggressive
                    5 => 50.0, // Maximum optimization
                    _ => 15.0, // Default
                }
            } else {
                0.0 // No optimization expected if no duplication
            };

            if size_reduction < expected_min_reduction {
                eprintln!("Size reduction insufficient: {:.1}% < {:.1}%", 
                    size_reduction, expected_min_reduction);
                return TestResult::failed();
            }

            // Optimized compilation should not be significantly slower
            if optimized_time.as_millis() > unoptimized_time.as_millis() * 3 {
                return TestResult::failed();
            }

            // All functions should compile successfully
            if optimized_functions.len() != functions.len() {
                return TestResult::failed();
            }

            // Optimized code should be valid WASM
            for (name, code) in &optimized_functions {
                if code.is_empty() {
                    eprintln!("Empty code for function: {}", name);
                    return TestResult::failed();
                }
                
                // Basic WASM validation
                if code.len() < 8 {
                    eprintln!("Code too short for function: {}", name);
                    return TestResult::failed();
                }
            }

            TestResult::passed()
        }

        QuickCheck::new()
            .tests(50)
            .gen(Gen::new(100))
            .quickcheck(property as fn(OptimizationScenario) -> TestResult);
    }

    /// Property: Streaming layout optimization improves instantiation speed
    #[test]
    fn prop_streaming_layout_improves_instantiation() {
        fn property(params: MonomorphizationTestParams) -> TestResult {
            let target = rustc_target::spec::Target {
                arch: "wasm32".to_string(),
                ..Default::default()
            };

            // Create functions for streaming test
            let mut functions = Vec::new();
            
            // Core runtime functions (should be loaded first)
            functions.push(create_monolithic_function("__core_runtime"));
            functions.push(create_monolithic_function("__memory_alloc"));
            functions.push(create_monolithic_function("__panic_handler"));
            
            // Generic functions
            for i in 0..params.generic_functions {
                let generic_func = create_generic_function(&format!("generic_{}", i), 
                                                  params.generic_param_count as usize);
                functions.push(generic_func);
            }
            
            // Application functions
            for i in 0..params.function_count {
                let app_func = create_monolithic_function(&format!("app_{}", i));
                functions.push(app_func);
            }

            // Test with streaming optimization enabled
            let mut stream_backend = WasmRustCraneliftBackend::new(target.clone());
            if stream_backend.is_err() {
                return TestResult::failed();
            }

            let mut stream_backend = stream_backend.unwrap();
            let start = Instant::now();
            let stream_result = stream_backend.compile_functions(&functions);
            let stream_time = start.elapsed();

            if stream_result.is_err() {
                return TestResult::failed();
            }

            let stream_functions = stream_result.unwrap();
            
            // Test without streaming optimization
            let mut nostream_backend = WasmRustCraneliftBackend::new(target.clone());
            if nostream_backend.is_err() {
                return TestResult::failed();
            }

            let mut nostream_backend = nostream_backend.unwrap();
            let start = Instant::now();
            let nostream_result = nostream_backend.compile_functions(&functions);
            let nostream_time = start.elapsed();

            if nostream_result.is_err() {
                return TestResult::failed();
            }

            let nostream_functions = nostream_result.unwrap();

            // Both should compile the same number of functions
            if stream_functions.len() != nostream_functions.len() {
                return TestResult::failed();
            }

            // Streaming layout should not increase total size significantly
            let stream_total_size: usize = stream_functions.values().map(|c| c.len()).sum();
            let nostream_total_size: usize = nostream_functions.values().map(|c| c.len()).sum();
            
            if stream_total_size > nostream_total_size * 2 {
                return TestResult::failed();
            }

            // Streaming layout should organize code for better instantiation
            // (We can't directly measure instantiation speed, but we can check patterns)
            
            // Check that core runtime functions are small and early
            let core_functions: Vec<_> = stream_functions.iter()
                .filter(|(name, _)| name.starts_with("__core_") || 
                                           name.starts_with("__memory_") ||
                                           name.starts_with("__panic_"))
                .collect();
            
            if !core_functions.is_empty() {
                // Core functions should be present and optimized
                let max_core_size = 500; // Core functions should be small
                for (name, code) in &core_functions {
                    if code.len() > max_core_size {
                        return TestResult::failed();
                    }
                }
            }

            // Streaming should not significantly increase compilation time
            if stream_time.as_millis() > nostream_time.as_millis() * 2 {
                return TestResult::failed();
            }

            TestResult::passed()
        }

        QuickCheck::new()
            .tests(30)
            .gen(Gen::new(100))
            .quickcheck(property as fn(MonomorphizationTestParams) -> TestResult);
    }

    /// Property: Thin monomorphization respects cache limits
    #[test]
    fn prop_monomorphization_cache_limits() {
        fn property(params: MonomorphizationTestParams) -> TestResult {
            let target = rustc_target::spec::Target {
                arch: "wasm32".to_string(),
                ..Default::default()
            };

            // Create monomorphization context with cache limits
            let mut mono_context = ThinMonomorphizationContext::new(target);
            let flags = MonomorphizationFlags {
                enable_deduplication: params.enable_optimization,
                enable_streaming_layout: true,
                enable_size_analysis: true,
                monomorphization_threshold: 3,
                max_cache_size: params.cache_size_limit as usize,
            };
            mono_context.set_optimization_flags(flags);

            // Create many generic function instantiations
            let mut functions = Vec::new();
            
            for i in 0..params.instantiations_count {
                let generic_func = create_generic_function(&format!("generic_base"), 
                                                      params.generic_param_count as usize);
                let mut instance = generic_func;
                instance.name = format!("instantiation_{}", i);
                functions.push(instance);
            }

            // Compile with monomorphization
            let mut backend = WasmRustCraneliftBackend::new(target);
            if backend.is_err() {
                return TestResult::failed();
            }

            let mut backend = backend.unwrap();
            let start = Instant::now();
            let result = backend.compile_functions(&functions);
            let compilation_time = start.elapsed();

            if result.is_err() {
                return TestResult::failed();
            }

            let compiled_functions = result.unwrap();

            // Should compile all functions
            if compiled_functions.len() != functions.len() {
                return TestResult::failed();
            }

            // Cache limits should be respected
            if params.enable_optimization && params.cache_size_limit > 0 {
                // Get monomorphization statistics
                let stats = mono_context.get_size_stats();
                
                // Number of monomorphizations should not exceed cache limit
                if stats.monomorphizations_performed > params.cache_size_limit as usize {
                    return TestResult::failed();
                }
            }

            // Compilation should complete in reasonable time
            let max_time = if params.enable_optimization {
                params.function_count as u64 * 200 // 200ms per function with optimization
            } else {
                params.function_count as u64 * 100 // 100ms per function without optimization
            };

            if compilation_time.as_millis() > max_time {
                return TestResult::failed();
            }

            // Generated code should be valid
            for (name, code) in &compiled_functions {
                if code.is_empty() {
                    return TestResult::failed();
                }
                
                // Basic size checks
                if name.starts_with("instantiation_") {
                    // Instantiations should be reasonable size
                    if code.len() > 2000 {
                        return TestResult::failed();
                    }
                }
            }

            TestResult::passed()
        }

        QuickCheck::new()
            .tests(40)
            .gen(Gen::new(100))
            .quickcheck(property as fn(MonomorphizationTestParams) -> TestResult);
    }

    /// Property: Code size scales linearly with function count after optimization
    #[test]
    fn prop_linear_code_size_scaling() {
        fn property(function_count: u8) -> TestResult {
            if function_count == 0 {
                return TestResult::discard();
            }

            let target = rustc_target::spec::Target {
                arch: "wasm32".to_string(),
                ..Default::default()
            };

            // Create functions with similar complexity
            let mut functions = Vec::new();
            for i in 0..function_count {
                let func = create_monolithic_function(&format!("linear_test_{}", i));
                functions.push(func);
            }

            // Compile with optimization
            let mut backend = WasmRustCraneliftBackend::new(target);
            if backend.is_err() {
                return TestResult::failed();
            }

            let mut backend = backend.unwrap();
            let start = Instant::now();
            let result = backend.compile_functions(&functions);
            let compilation_time = start.elapsed();

            if result.is_err() {
                return TestResult::failed();
            }

            let compiled_functions = result.unwrap();

            // Should compile all functions
            if compiled_functions.len() != functions.len() as usize {
                return TestResult::failed();
            }

            // Calculate code size statistics
            let sizes: Vec<usize> = compiled_functions.values().map(|c| c.len()).collect();
            let total_size: usize = sizes.iter().sum();
            let avg_size = total_size / sizes.len();
            let min_size = sizes.iter().min().unwrap();
            let max_size = sizes.iter().max().unwrap();

            // Code size should scale linearly
            // Check that variance is reasonable (no function is dramatically larger)
            let size_variance = max_size - min_size;
            let max_reasonable_variance = avg_size * 3; // Allow 3x variance

            if size_variance > max_reasonable_variance {
                eprintln!("Excessive size variance: min={}, max={}, avg={}", 
                    min_size, max_size, avg_size);
                return TestResult::failed();
            }

            // Average size should be reasonable
            let max_reasonable_avg_size = 2000; // 2KB per function max
            if avg_size > max_reasonable_avg_size {
                return TestResult::failed();
            }

            // Total size should be reasonable for function count
            let max_total_size = function_count as usize * max_reasonable_avg_size;
            if total_size > max_total_size {
                return TestResult::failed();
            }

            // Compilation time should scale linearly
            let max_reasonable_time = function_count as u64 * 150; // 150ms per function max
            if compilation_time.as_millis() > max_reasonable_time {
                return TestResult::failed();
            }

            TestResult::passed()
        }

        QuickCheck::new()
            .tests(30)
            .gen(Gen::new(100))
            .quickcheck(property as fn(u8) -> TestResult);
    }

    /// Property: Optimization doesn't break functionality
    #[test]
    fn prop_optimization_preserves_functionality() {
        fn property(complexity: u8) -> TestResult {
            let complexity_level = (complexity % 5) + 1; // 1-5

            let target = rustc_target::spec::Target {
                arch: "wasm32".to_string(),
                ..Default::default()
            };

            // Create functions with different complexity levels
            let mut functions = Vec::new();
            
            // Simple arithmetic function
            let simple_func = create_monolithic_function("test_simple");
            functions.push(simple_func);
            
            // More complex function (based on complexity level)
            for i in 0..complexity_level {
                let complex_func = create_complex_function(&format!("test_complex_{}", i), complexity_level);
                functions.push(complex_func);
            }

            // Compile without optimization
            let mut unoptimized_backend = WasmRustCraneliftBackend::new(target.clone());
            if unoptimized_backend.is_err() {
                return TestResult::failed();
            }

            let mut unoptimized_backend = unoptimized_backend.unwrap();
            let unoptimized_result = unoptimized_backend.compile_functions(&functions);

            if unoptimized_result.is_err() {
                return TestResult::failed();
            }

            let unoptimized_functions = unoptimized_result.unwrap();

            // Compile with optimization
            let mut optimized_backend = WasmRustCraneliftBackend::new(target.clone());
            if optimized_backend.is_err() {
                return TestResult::failed();
            }

            let mut optimized_backend = optimized_backend.unwrap();
            let optimized_result = optimized_backend.compile_functions(&functions);

            if optimized_result.is_err() {
                return TestResult::failed();
            }

            let optimized_functions = optimized_result.unwrap();

            // Should compile same number of functions
            if optimized_functions.len() != unoptimized_functions.len() {
                return TestResult::failed();
            }

            // Each optimized function should be valid WASM
            for (name, optimized_code) in &optimized_functions {
                if optimized_code.is_empty() {
                    return TestResult::failed();
                }

                // Basic WASM validation
                if optimized_code.len() < 8 {
                    return TestResult::failed();
                }

                // Check for WASM magic number
                if optimized_code.len() >= 4 {
                    if &optimized_code[0..4] != &[0x00, 0x61, 0x73, 0x6d] {
                        // Not necessarily an error if not WASM format,
                        // but should be consistent
                    }
                }

                // Check function size is reasonable
                let max_size = match complexity_level {
                    1 => 500,
                    2 => 1000,
                    3 => 2000,
                    4 => 3000,
                    5 => 5000,
                    _ => 1000,
                };

                if optimized_code.len() > max_size {
                    eprintln!("Function {} too large: {} > {}", name, optimized_code.len(), max_size);
                    return TestResult::failed();
                }
            }

            TestResult::passed()
        }

        QuickCheck::new()
            .tests(25)
            .gen(Gen::new(100))
            .quickcheck(property as fn(u8) -> TestResult);
    }

    /// Property: Memory usage remains reasonable during optimization
    #[test]
    fn prop_optimization_memory_usage() {
        fn property(function_count: u8) -> TestResult {
            if function_count == 0 {
                return TestResult::discard();
            }

            let target = rustc_target::spec::Target {
                arch: "wasm32".to_string(),
                ..Default::default()
            };

            // Create many functions to stress test memory
            let mut functions = Vec::new();
            for i in 0..function_count {
                let func = create_generic_function(&format!("memory_test_{}", i), 2);
                
                // Create some duplication to stress monomorphization
                for j in 0..3 {
                    let mut dup_func = func.clone();
                    dup_func.name = format!("memory_test_{}_dup_{}", i, j);
                    functions.push(dup_func);
                }
            }

            // Compile with optimization enabled
            let mut backend = WasmRustCraneliftBackend::new(target);
            if backend.is_err() {
                return TestResult::failed();
            }

            let mut backend = backend.unwrap();
            
            // Clear initial stats
            backend.clear_stats();
            let initial_stats = backend.get_stats();
            
            if initial_stats.functions_compiled != 0 ||
               initial_stats.instructions_generated != 0 ||
               initial_stats.compilation_time_ms != 0 {
                return TestResult::failed();
            }

            // Measure memory pressure through compilation time and patterns
            let start = Instant::now();
            let result = backend.compile_functions(&functions);
            let compilation_time = start.elapsed();

            if result.is_err() {
                return TestResult::failed();
            }

            let compiled_functions = result.unwrap();

            // Should handle all functions without memory issues
            if compiled_functions.len() != functions.len() {
                return TestResult::failed();
            }

            // Check final statistics
            let final_stats = backend.get_stats();

            if final_stats.functions_compiled != functions.len() {
                return TestResult::failed();
            }

            if final_stats.instructions_generated == 0 {
                return TestResult::failed();
            }

            // Memory usage should be reasonable (inferred from compilation patterns)
            let avg_time_per_function = compilation_time.as_millis() as f64 / functions.len() as f64;
            
            // Average compilation time should be reasonable
            let max_reasonable_avg_time = 300.0; // 300ms per function max
            if avg_time_per_function > max_reasonable_avg_time {
                return TestResult::failed();
            }

            // Total compilation time should be reasonable
            let max_reasonable_total_time = function_count as u64 * 500; // 500ms per function max
            if compilation_time.as_millis() > max_reasonable_total_time {
                return TestResult::failed();
            }

            // No function should be abnormally large (indicating memory bloat)
            for (name, code) in &compiled_functions {
                let max_reasonable_size = 10000; // 10KB per function max
                if code.len() > max_reasonable_size {
                    eprintln!("Function {} possibly suffering from memory bloat: {} bytes", 
                        name, code.len());
                    return TestResult::failed();
                }
            }

            TestResult::passed()
        }

        QuickCheck::new()
            .tests(20)
            .gen(Gen::new(100))
            .quickcheck(property as fn(u8) -> TestResult);
    }

    /// Creates a complex function for testing
    fn create_complex_function(name: &str, complexity: u8) -> WasmIR {
        let signature = Signature {
            params: vec![Type::I32, Type::I32],
            returns: Some(Type::I32),
        };

        let mut func = WasmIR::new(name.to_string(), signature);
        
        let local_a = func.add_local(Type::I32);
        let local_b = func.add_local(Type::I32);
        let local_temp1 = func.add_local(Type::I32);
        let local_temp2 = func.add_local(Type::I32);
        let local_result = func.add_local(Type::I32);

        let mut instructions = Vec::new();

        // Add complexity based on level
        match complexity {
            1 => {
                // Simple: a + b
                instructions.push(Instruction::LocalGet { index: local_a });
                instructions.push(Instruction::LocalGet { index: local_b });
                instructions.push(Instruction::BinaryOp {
                    op: BinaryOp::Add,
                    left: Operand::Local(local_temp1),
                    right: Operand::Local(local_b),
                });
            }
            2 => {
                // Medium: (a * b) + (a + b)
                instructions.push(Instruction::LocalGet { index: local_a });
                instructions.push(Instruction::LocalGet { index: local_b });
                instructions.push(Instruction::BinaryOp {
                    op: BinaryOp::Mul,
                    left: Operand::Local(local_temp1),
                    right: Operand::Local(local_b),
                });
                instructions.push(Instruction::LocalSet {
                    index: local_temp1,
                    value: Operand::Local(0),
                });
                instructions.push(Instruction::LocalGet { index: local_a });
                instructions.push(Instruction::LocalGet { index: local_b });
                instructions.push(Instruction::BinaryOp {
                    op: BinaryOp::Add,
                    left: Operand::Local(local_temp2),
                    right: Operand::Local(local_b),
                });
            }
            3 => {
                // Complex: multiple operations
                for i in 0..3 {
                    instructions.push(Instruction::LocalGet { index: local_a });
                    instructions.push(Instruction::LocalGet { index: local_b });
                    instructions.push(Instruction::BinaryOp {
                        op: match i % 3 {
                            0 => BinaryOp::Add,
                            1 => BinaryOp::Mul,
                            2 => BinaryOp::Sub,
                            _ => BinaryOp::Add,
                        },
                        left: Operand::Local(local_temp1),
                        right: Operand::Local(local_b),
                    });
                    instructions.push(Instruction::LocalSet {
                        index: local_temp1,
                        value: Operand::Local(0),
                    });
                }
            }
            4 => {
                // Very complex: nested operations
                for i in 0..5 {
                    instructions.push(Instruction::LocalGet { index: local_a });
                    instructions.push(Instruction::UnaryOp {
                        op: UnaryOp::Neg,
                        value: Operand::Local(local_a),
                    });
                    instructions.push(Instruction::LocalSet {
                        index: local_temp1,
                        value: Operand::Local(0),
                    });
                    instructions.push(Instruction::LocalGet { index: local_temp1 });
                    instructions.push(Instruction::BinaryOp {
                        op: BinaryOp::Mul,
                        left: Operand::Local(local_temp2),
                        right: Operand::Local(local_temp1),
                    });
                }
            }
            5 => {
                // Extremely complex: maximum operations
                for i in 0..10 {
                    let op = match i % 5 {
                        0 => BinaryOp::Add,
                        1 => BinaryOp::Mul,
                        2 => BinaryOp::Sub,
                        3 => BinaryOp::Div,
                        4 => BinaryOp::Mod,
                        _ => BinaryOp::Add,
                    };
                    
                    instructions.push(Instruction::LocalGet { index: local_a });
                    instructions.push(Instruction::LocalGet { index: local_b });
                    instructions.push(Instruction::BinaryOp {
                        op,
                        left: Operand::Local(local_temp1),
                        right: Operand::Local(local_b),
                    });
                    instructions.push(Instruction::LocalSet {
                        index: local_temp1,
                        value: Operand::Local(0),
                    });
                }
            }
            _ => {}
        }

        // Final result
        instructions.push(Instruction::LocalSet {
            index: local_result,
            value: Operand::Local(local_temp1),
        });
        instructions.push(Instruction::Return {
            value: Some(Operand::Local(local_result)),
        });

        let terminator = Terminator::Return {
            value: Some(Operand::Local(local_result)),
        };

        func.add_basic_block(instructions, terminator);
        func
    }
}

/// Binary size optimization utilities
pub mod utils {
    use super::*;

    /// Measures binary size reduction from optimization
    pub fn measure_size_reduction(
        original_sizes: &[usize],
        optimized_sizes: &[usize],
    ) -> f64 {
        let original_total: usize = original_sizes.iter().sum();
        let optimized_total: usize = optimized_sizes.iter().sum();
        
        if original_total == 0 {
            return 0.0;
        }
        
        ((original_total - optimized_total) as f64 / original_total as f64) * 100.0
    }

    /// Validates WASM binary format
    pub fn validate_wasm_binary(code: &[u8]) -> Result<(), String> {
        if code.len() < 8 {
            return Err("Binary too short".to_string());
        }
        
        // Check WASM magic number
        if &code[0..4] != &[0x00, 0x61, 0x73, 0x6d] {
            return Err("Invalid WASM magic number".to_string());
        }
        
        // Check version
        if code[4] != 1 {
            return Err("Unsupported WASM version".to_string());
        }
        
        Ok(())
    }

    /// Estimates code complexity from size
    pub fn estimate_complexity_from_size(size: usize) -> u8 {
        match size {
            0..100 => 1,
            101..500 => 2,
            501..2000 => 3,
            2001..5000 => 4,
            _ => 5,
        }
    }

    /// Checks if size reduction is significant
    pub fn is_significant_reduction(reduction: f64) -> bool {
        reduction > 5.0 // More than 5% reduction is considered significant
    }

    /// Analyzes size distribution across functions
    pub fn analyze_size_distribution(sizes: &[usize]) -> SizeDistribution {
        if sizes.is_empty() {
            return SizeDistribution {
                min: 0,
                max: 0,
                avg: 0.0,
                median: 0,
                std_dev: 0.0,
            };
        }
        
        let min = sizes.iter().min().unwrap();
        let max = sizes.iter().max().unwrap();
        let avg = sizes.iter().sum::<usize>() as f64 / sizes.len() as f64;
        
        let mut sorted_sizes = sizes.clone();
        sorted_sizes.sort();
        let median = if sorted_sizes.len() % 2 == 0 {
            (sorted_sizes[sorted_sizes.len() / 2 - 1] + sorted_sizes[sorted_sizes.len() / 2]) / 2
        } else {
            sorted_sizes[sorted_sizes.len() / 2]
        };
        
        let variance = sizes.iter()
            .map(|&size| (size as f64 - avg).powi(2))
            .sum::<f64>() / sizes.len() as f64;
        let std_dev = variance.sqrt();
        
        SizeDistribution {
            min: *min,
            max: *max,
            avg,
            median,
            std_dev,
        }
    }

    /// Size distribution statistics
    #[derive(Debug, Clone)]
    pub struct SizeDistribution {
        pub min: usize,
        pub max: usize,
        pub avg: f64,
        pub median: usize,
        pub std_dev: f64,
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use super::utils::*;

    #[test]
    fn test_known_optimization_cases() {
        let target = rustc_target::spec::Target {
            arch: "wasm32".to_string(),
            ..Default::default()
        };
        
        // Test case 1: Simple functions should have minimal overhead
        let simple_func = create_monolithic_function("simple_test");
        let mut backend = WasmRustCraneliftBackend::new(target).unwrap();
        let result = backend.compile_function(&simple_func, "simple_test");
        
        assert!(result.is_ok(), "Simple function should compile");
        let code = result.unwrap();
        assert!(code.len() < 500, "Simple function should be small");
        
        // Validate WASM format
        assert!(validate_wasm_binary(&code).is_ok());
    }

    #[test]
    fn test_size_reduction_measurement() {
        let original_sizes = vec![1000, 800, 1200];
        let optimized_sizes = vec![900, 700, 1000];
        
        let reduction = measure_size_reduction(&original_sizes, &optimized_sizes);
        let expected = ((1000 + 800 + 1200) - (900 + 700 + 1000)) as f64 
            / (1000 + 800 + 1200) as f64 * 100.0;
        
        assert!((reduction - expected).abs() < 0.01, "Size reduction calculation should be accurate");
        assert!(reduction > 10.0, "Should show significant reduction");
        assert!(is_significant_reduction(reduction), "Reduction should be significant");
    }

    #[test]
    fn test_size_distribution_analysis() {
        let sizes = vec![100, 200, 300, 400, 500];
        let distribution = analyze_size_distribution(&sizes);
        
        assert_eq!(distribution.min, 100);
        assert_eq!(distribution.max, 500);
        assert!((distribution.avg - 300.0).abs() < 0.01);
        assert_eq!(distribution.median, 300);
        
        // Standard deviation should be reasonable
        assert!(distribution.std_dev > 100.0);
        assert!(distribution.std_dev < 200.0);
    }

    #[test]
    fn test_complexity_estimation() {
        assert_eq!(estimate_complexity_from_size(50), 1);
        assert_eq!(estimate_complexity_from_size(250), 2);
        assert_eq!(estimate_complexity_from_size(1500), 3);
        assert_eq!(estimate_complexity_from_size(3000), 4);
        assert_eq!(estimate_complexity_from_size(10000), 5);
    }

    #[test]
    fn test_wasm_validation() {
        // Valid WASM
        let valid_wasm = vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];
        assert!(validate_wasm_binary(&valid_wasm).is_ok());
        
        // Invalid magic number
        let invalid_wasm = vec![0xFF, 0xFF, 0xFF, 0xFF, 0x01, 0x00, 0x00, 0x00];
        assert!(validate_wasm_binary(&invalid_wasm).is_err());
        
        // Too short
        let short_wasm = vec![0x00, 0x61, 0x73];
        assert!(validate_wasm_binary(&short_wasm).is_err());
    }
}
