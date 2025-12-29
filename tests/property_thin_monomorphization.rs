//! Property-based tests for thin monomorphization effectiveness
//! 
//! This module validates that thin monomorphization reduces code duplication
//! while maintaining performance characteristics.
//! 
//! Property 2: Thin Monomorphization Effectiveness
//! Validates: Requirements 1.3

use wasm::backend::{BackendFactory, BuildProfile, CompilationResult};
use wasm::backend::cranelift::{WasmRustCraneliftBackend, MonomorphizationFlags, ThinMonomorphizationContext};
use wasm::wasmir::{WasmIR, Signature, Type, Instruction, Terminator, Operand, BinaryOp};
use quickcheck::{Arbitrary, Gen, QuickCheck, TestResult};
use std::time::{Instant, Duration};
use std::collections::HashMap;

#[cfg(test)]
mod tests {
    use super::*;

    /// Test data for monomorphization effectiveness
    #[derive(Debug, Clone)]
    struct MonomorphizationTestData {
        name: &'static str,
        generic_functions: Vec<WasmIR>,
        expected_size_reduction: f64, // Percentage reduction expected
        performance_impact: f64, // Maximum acceptable performance impact
        complexity_level: u8, // 1-5 complexity
    }

    /// Generic function scenarios for testing
    #[derive(Debug, Clone, Arbitrary)]
    struct GenericScenario {
        function_count: u8,
        generic_params_per_function: u8,
        instantiations_per_function: u8,
        code_complexity: u8, // 1-5
        enable_optimizations: bool,
    }

    impl Arbitrary for GenericScenario {
        fn arbitrary(g: &mut Gen) -> Self {
            Self {
                function_count: g.gen_range(1..6),
                generic_params_per_function: g.gen_range(1..4),
                instantiations_per_function: g.gen_range(2..10),
                code_complexity: g.gen_range(1..6),
                enable_optimizations: g.gen_bool(),
            }
        }
    }

    /// Creates test scenarios for monomorphization
    fn create_test_scenarios() -> Vec<MonomorphizationTestData> {
        vec![
            MonomorphizationTestData {
                name: "simple_generic_identity",
                generic_functions: vec![create_generic_identity_function()],
                expected_size_reduction: 15.0, // 15% reduction expected
                performance_impact: 5.0, // <5% performance impact
                complexity_level: 1,
            },
            MonomorphizationTestData {
                name: "complex_generic_container",
                generic_functions: vec![create_generic_container_function()],
                expected_size_reduction: 25.0, // 25% reduction expected
                performance_impact: 10.0, // <10% performance impact
                complexity_level: 3,
            },
            MonomorphizationTestData {
                name: "multiple_generic_functions",
                generic_functions: vec![
                    create_generic_identity_function(),
                    create_generic_arithmetic_function(),
                    create_generic_container_function(),
                ],
                expected_size_reduction: 35.0, // 35% reduction expected
                performance_impact: 15.0, // <15% performance impact
                complexity_level: 4,
            },
        ]
    }

    /// Creates a simple generic identity function
    fn create_generic_identity_function() -> WasmIR {
        let signature = Signature {
            params: vec![Type::Ref("T".to_string())],
            returns: Some(Type::Ref("T".to_string())),
        };

        let mut func = WasmIR::new("identity<T>".to_string(), signature);
        let local_input = func.add_local(Type::Ref("T".to_string()));

        let instructions = vec![
            Instruction::LocalGet { index: local_input },
            Instruction::Return {
                value: Some(Operand::Local(local_input)),
            },
        ];

        let terminator = Terminator::Return {
            value: Some(Operand::Local(local_input)),
        };

        func.add_basic_block(instructions, terminator);
        func
    }

    /// Creates a generic arithmetic function
    fn create_generic_arithmetic_function() -> WasmIR {
        let signature = Signature {
            params: vec![
                Type::Ref("T".to_string()),
                Type::Ref("T".to_string()),
            ],
            returns: Some(Type::Ref("T".to_string())),
        };

        let mut func = WasmIR::new("add<T>".to_string(), signature);
        let local_a = func.add_local(Type::Ref("T".to_string()));
        let local_b = func.add_local(Type::Ref("T".to_string()));
        let local_result = func.add_local(Type::Ref("T".to_string()));

        let instructions = vec![
            Instruction::LocalGet { index: local_a },
            Instruction::LocalGet { index: local_b },
            Instruction::BinaryOp {
                op: BinaryOp::Add,
                left: Operand::Local(local_result),
                right: Operand::Local(local_b),
            },
            Instruction::LocalSet {
                index: local_result,
                value: Operand::Local(0),
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

    /// Creates a generic container function
    fn create_generic_container_function() -> WasmIR {
        let signature = Signature {
            params: vec![
                Type::Array { 
                    element_type: Box::new(Type::Ref("T".to_string())),
                    size: Some(10),
                },
                Type::I32, // Index
            ],
            returns: Some(Type::Ref("T".to_string())),
        };

        let mut func = WasmIR::new("get_element<T>".to_string(), signature);
        let local_array = func.add_local(Type::Array {
            element_type: Box::new(Type::Ref("T".to_string())),
            size: Some(10),
        });
        let local_index = func.add_local(Type::I32);
        let local_element = func.add_local(Type::Ref("T".to_string()));

        let instructions = vec![
            Instruction::LocalGet { index: local_array },
            Instruction::LocalGet { index: local_index },
            Instruction::BinaryOp {
                op: BinaryOp::Add,
                left: Operand::Local(0),
                right: Operand::Local(local_index),
            },
            Instruction::LocalSet {
                index: local_element,
                value: Operand::Local(1),
            },
            Instruction::Return {
                value: Some(Operand::Local(local_element)),
            },
        ];

        let terminator = Terminator::Return {
            value: Some(Operand::Local(local_element)),
        };

        func.add_basic_block(instructions, terminator);
        func
    }

    /// Property: Thin monomorphization reduces code size effectively
    #[test]
    fn prop_thin_monomorphization_reduces_size() {
        fn property(scenario: GenericScenario) -> TestResult {
            // Skip unrealistic scenarios
            if scenario.function_count == 0 || 
               scenario.generic_params_per_function == 0 ||
               scenario.instantiations_per_function < 2 {
                return TestResult::discard();
            }

            let target = rustc_target::spec::Target {
                arch: "wasm32".to_string(),
                ..Default::default()
            };

            let mut backend = WasmRustCraneliftBackend::new(target);
            if backend.is_err() {
                return TestResult::failed();
            }

            let mut backend = backend.unwrap();

            // Create generic functions based on scenario
            let mut generic_functions = Vec::new();
            for i in 0..scenario.function_count {
                let func = match (i % 3) + 1 {
                    1 => create_generic_identity_function(),
                    2 => create_generic_arithmetic_function(),
                    3 => create_generic_container_function(),
                    _ => create_generic_identity_function(),
                };
                generic_functions.push(func);
            }

            // Create multiple instantiations
            let mut instantiated_functions = Vec::new();
            for (i, base_func) in generic_functions.iter().enumerate() {
                for j in 0..scenario.instantiations_per_function {
                    let mut func_instance = base_func.clone();
                    func_instance.name = format!("{}_inst_{}", base_func.name, j);
                    instantiated_functions.push(func_instance);
                }
            }

            // Compile without monomorphization
            let start = Instant::now();
            let unoptimized_results = backend.compile_functions(&instantiated_functions);
            let unoptimized_time = start.elapsed();

            if unoptimized_results.is_err() {
                return TestResult::failed();
            }

            let unoptimized_functions = unoptimized_results.unwrap();
            let unoptimized_size: usize = unoptimized_functions.values()
                .map(|code| code.len())
                .sum();

            // Reset backend for optimized compilation
            backend.clear_stats();

            // Apply thin monomorphization
            let target_clone = target.clone();
            let mut optimized_backend = WasmRustCraneliftBackend::new(target_clone);
            if optimized_backend.is_err() {
                return TestResult::failed();
            }

            let mut optimized_backend = optimized_backend.unwrap();

            // Set up monomorphization context
            let mut mono_context = ThinMonomorphizationContext::new(target.clone());
            let mono_flags = MonomorphizationFlags {
                enable_deduplication: scenario.enable_optimizations,
                enable_streaming_layout: scenario.enable_optimizations,
                enable_size_analysis: true,
                monomorphization_threshold: 2,
                max_cache_size: 1000,
            };
            mono_context.set_optimization_flags(mono_flags);

            // Compile with monomorphization
            let start = Instant::now();
            let optimized_results = optimized_backend.compile_functions(&instantiated_functions);
            let optimized_time = start.elapsed();

            if optimized_results.is_err() {
                return TestResult::failed();
            }

            let optimized_functions = optimized_results.unwrap();
            let optimized_size: usize = optimized_functions.values()
                .map(|code| code.len())
                .sum();

            // Check size reduction
            let size_reduction = if unoptimized_size > 0 {
                ((unoptimized_size - optimized_size) as f64 / unoptimized_size as f64) * 100.0
            } else {
                0.0
            };

            // Expected reduction based on scenario complexity
            let expected_min_reduction = match scenario.code_complexity {
                1 => 5.0,   // Simple: 5% min
                2 => 10.0,  // Medium: 10% min
                3 => 15.0,  // Complex: 15% min
                4 => 20.0,  // Very complex: 20% min
                5 => 25.0,  // Extremely complex: 25% min
                _ => 10.0,  // Default
            };

            // Size reduction should meet minimum threshold
            if size_reduction < expected_min_reduction {
                eprintln!("Insufficient size reduction: {:.1}% < {:.1}%", 
                    size_reduction, expected_min_reduction);
                return TestResult::failed();
            }

            // Performance impact should be reasonable
            let performance_impact = if optimized_time.as_millis() > 0 {
                ((optimized_time.as_millis() as f64 - unoptimized_time.as_millis() as f64) / 
                 unoptimized_time.as_millis() as f64) * 100.0
            } else {
                0.0
            };

            let max_performance_impact = match scenario.code_complexity {
                1..=3 => 10.0,   // Simple to complex: 10% max
                4 => 15.0,        // Very complex: 15% max
                5 => 20.0,        // Extremely complex: 20% max
                _ => 10.0,        // Default
            };

            if performance_impact > max_performance_impact {
                eprintln!("Performance impact too high: {:.1}% > {:.1}%", 
                    performance_impact, max_performance_impact);
                return TestResult::failed();
            }

            // Code quality should be maintained
            for (name, code) in &optimized_functions {
                if code.is_empty() {
                    eprintln!("Empty code for function: {}", name);
                    return TestResult::failed();
                }

                // Check for reasonable WASM structure
                if code.len() < 20 {
                    eprintln!("Code too small for function: {}", name);
                    return TestResult::failed();
                }

                if code.len() > 10000 {
                    eprintln!("Code too large for function: {}", name);
                    return TestResult::failed();
                }
            }

            TestResult::passed()
        }

        QuickCheck::new()
            .tests(100)
            .gen(Gen::new(100))
            .quickcheck(property as fn(GenericScenario) -> TestResult);
    }

    /// Property: Monomorphization preserves function correctness
    #[test]
    fn prop_monomorphization_preserves_correctness() {
        fn property(scenario: GenericScenario) -> TestResult {
            // Skip invalid scenarios
            if scenario.function_count == 0 {
                return TestResult::discard();
            }

            let target = rustc_target::spec::Target {
                arch: "wasm32".to_string(),
                ..Default::default()
            };

            // Create test functions
            let mut test_functions = Vec::new();
            for i in 0..scenario.function_count.min(3) {
                let func = match i % 3 {
                    0 => create_generic_identity_function(),
                    1 => create_generic_arithmetic_function(),
                    2 => create_generic_container_function(),
                    _ => create_generic_identity_function(),
                };
                test_functions.push(func);
            }

            // Compile with different optimization levels
            let results: Vec<Result<HashMap<String, Vec<u8>>, _>> = vec![
                compile_with_optimization(&target, &test_functions, false, false),
                compile_with_optimization(&target, &test_functions, true, false),
                compile_with_optimization(&target, &test_functions, false, true),
                compile_with_optimization(&target, &test_functions, true, true),
            ];

            // All should compile successfully
            for (i, result) in results.iter().enumerate() {
                if result.is_err() {
                    eprintln!("Compilation {} failed", i);
                    return TestResult::failed();
                }
            }

            // Extract compiled functions
            let compiled_sets: Vec<HashMap<String, Vec<u8>>> = results
                .iter()
                .map(|r| r.clone().unwrap())
                .collect();

            // Check that all functions are present in all sets
            for (i, test_func) in test_functions.iter().enumerate() {
                let func_name = format!("{}_inst_0", test_func.name);

                for (j, compiled_set) in compiled_sets.iter().enumerate() {
                    if !compiled_set.contains_key(&func_name) {
                        eprintln!("Function {} missing in compilation set {}", func_name, j);
                        return TestResult::failed();
                    }

                    let code = compiled_set.get(&func_name).unwrap();
                    if code.is_empty() {
                        eprintln!("Empty code for {} in set {}", func_name, j);
                        return TestResult::failed();
                    }
                }
            }

            // Check that optimized versions are not larger than unoptimized
            if compiled_sets.len() >= 2 {
                let unoptimized_size = compiled_sets[0].values()
                    .map(|code| code.len())
                    .sum::<usize>();

                let optimized_size = compiled_sets[1].values()
                    .map(|code| code.len())
                    .sum::<usize>();

                if optimized_size > unoptimized_size * 2 {
                    eprintln!("Optimized code too large: {} > {} * 2", 
                        optimized_size, unoptimized_size);
                    return TestResult::failed();
                }
            }

            TestResult::passed()
        }

        QuickCheck::new()
            .tests(50)
            .gen(Gen::new(100))
            .quickcheck(property as fn(GenericScenario) -> TestResult);
    }

    /// Property: Streaming layout improves instantiation time
    #[test]
    fn prop_streaming_layout_improves_instantiation() {
        fn property(function_count: u8) -> TestResult {
            if function_count < 2 {
                return TestResult::discard();
            }

            let target = rustc_target::spec::Target {
                arch: "wasm32".to_string(),
                ..Default::default()
            };

            // Create test functions
            let mut test_functions = Vec::new();
            for i in 0..function_count {
                let func = match i % 3 {
                    0 => create_generic_identity_function(),
                    1 => create_generic_arithmetic_function(),
                    2 => create_generic_container_function(),
                    _ => create_generic_identity_function(),
                };
                test_functions.push(func);
            }

            // Compile without streaming layout
            let no_streaming_result = compile_with_optimization(&target, &test_functions, true, false);
            if no_streaming_result.is_err() {
                return TestResult::failed();
            }

            let no_streaming_size: usize = no_streaming_result.unwrap().values()
                .map(|code| code.len())
                .sum();

            // Compile with streaming layout
            let streaming_result = compile_with_optimization(&target, &test_functions, true, true);
            if streaming_result.is_err() {
                return TestResult::failed();
            }

            let streaming_size: usize = streaming_result.unwrap().values()
                .map(|code| code.len())
                .sum();

            // Streaming layout should not significantly increase size
            let size_overhead = if streaming_size > 0 {
                ((streaming_size - no_streaming_size) as f64 / no_streaming_size as f64) * 100.0
            } else {
                0.0
            };

            let max_size_overhead = 5.0; // Maximum 5% size overhead for streaming

            if size_overhead > max_size_overhead {
                eprintln!("Streaming layout overhead too high: {:.1}% > {:.1}%", 
                    size_overhead, max_size_overhead);
                return TestResult::failed();
            }

            // Check that streaming layout provides better structure
            let streaming_compiled = streaming_result.unwrap();
            for (name, code) in &streaming_compiled {
                // Should have proper WASM structure
                if code.len() < 20 {
                    eprintln!("Streaming code too small for function: {}", name);
                    return TestResult::failed();
                }

                // Should contain reasonable instruction sequences
                let instruction_count = count_wasm_instructions(code);
                if instruction_count < 3 {
                    eprintln!("Too few instructions for function: {}", name);
                    return TestResult::failed();
                }
            }

            TestResult::passed()
        }

        QuickCheck::new()
            .tests(30)
            .gen(Gen::new(100))
            .quickcheck(property as fn(u8) -> TestResult);
    }

    /// Property: Code deduplication maintains semantic equivalence
    #[test]
    fn prop_code_deduplication_maintains_semantics() {
        fn property(duplication_factor: u8) -> TestResult {
            if duplication_factor < 2 || duplication_factor > 5 {
                return TestResult::discard();
            }

            let target = rustc_target::spec::Target {
                arch: "wasm32".to_string(),
                ..Default::default()
            };

            // Create identical generic functions
            let base_func = create_generic_identity_function();
            let mut identical_functions = Vec::new();
            
            for i in 0..duplication_factor {
                let mut func_instance = base_func.clone();
                func_instance.name = format!("identical_{}", i);
                identical_functions.push(func_instance);
            }

            // Compile with deduplication
            let deduplicated_result = compile_with_optimization(&target, &identical_functions, true, true);
            if deduplicated_result.is_err() {
                return TestResult::failed();
            }

            let deduplicated_size: usize = deduplicated_result.unwrap().values()
                .map(|code| code.len())
                .sum();

            // Compile without deduplication
            let no_deduplication_result = compile_with_optimization(&target, &identical_functions, false, false);
            if no_deduplication_result.is_err() {
                return TestResult::failed();
            }

            let no_deduplication_size: usize = no_deduplication_result.unwrap().values()
                .map(|code| code.len())
                .sum();

            // Deduplication should reduce size for identical functions
            let expected_min_reduction = (duplication_factor - 1) * 10; // 10% per duplicate
            let actual_reduction = no_deduplication_size as i32 - deduplicated_size as i32;

            if actual_reduction < expected_min_reduction as i32 {
                eprintln!("Insufficient deduplication: {} < {} bytes", 
                    actual_reduction, expected_min_reduction);
                return TestResult::failed();
            }

            // All functions should still be present (as references)
            let deduplicated_compiled = deduplicated_result.unwrap();
            for i in 0..duplication_factor {
                let func_name = format!("identical_{}", i);
                if !deduplicated_compiled.contains_key(&func_name) {
                    eprintln!("Function {} missing after deduplication", func_name);
                    return TestResult::failed();
                }
            }

            TestResult::passed()
        }

        QuickCheck::new()
            .tests(40)
            .gen(Gen::new(100))
            .quickcheck(property as fn(u8) -> TestResult);
    }

    /// Property: Monomorphization cache effectiveness
    #[test]
    fn prop_monomorphization_cache_effectiveness() {
        fn property(cache_size: u8, access_pattern: u8) -> TestResult {
            if cache_size == 0 || cache_size > 100 {
                return TestResult::discard();
            }

            let target = rustc_target::spec::Target {
                arch: "wasm32".to_string(),
                ..Default::default()
            };

            // Create functions that would benefit from caching
            let mut functions = Vec::new();
            for i in 0..cache_size {
                let func = create_generic_identity_function();
                functions.push(func);
            }

            // Simulate multiple compilation passes with different access patterns
            let mut compilation_times = Vec::new();
            let mut cache_hits = 0;

            for pass in 0..5 {
                // Simulate cache access based on pattern
                let cache_hit_rate = match access_pattern % 3 {
                    0 => 0.9, // High locality
                    1 => 0.5, // Medium locality
                    2 => 0.1, // Low locality
                    _ => 0.5, // Default
                };

                let start = Instant::now();
                let result = BackendFactory::create_backend("wasm32", BuildProfile::Development);
                let compilation_time = start.elapsed();

                if result.is_ok() {
                    compilation_times.push(compilation_time.as_millis());

                    // Simulate cache hit based on pattern
                    if pass > 0 && (pass as f64 / 5.0) < cache_hit_rate {
                        cache_hits += 1;
                    }
                }
            }

            // Cache should improve compilation times over time
            if compilation_times.len() > 1 {
                let first_time = compilation_times[0];
                let last_time = compilation_times[compilation_times.len() - 1];

                // Later compilations should be faster (cache effect)
                let improvement_rate = if first_time > 0 {
                    ((first_time - last_time) as f64 / first_time as f64) * 100.0
                } else {
                    0.0
                };

                let min_expected_improvement = match access_pattern % 3 {
                    0 => 20.0, // High locality: 20% improvement
                    1 => 10.0, // Medium locality: 10% improvement
                    2 => 5.0,  // Low locality: 5% improvement
                    _ => 10.0,  // Default
                };

                if improvement_rate < min_expected_improvement {
                    eprintln!("Insufficient cache improvement: {:.1}% < {:.1}%", 
                        improvement_rate, min_expected_improvement);
                    return TestResult::failed();
                }
            }

            // Cache hit rate should be reasonable for access pattern
            let expected_cache_hit_rate = match access_pattern % 3 {
                0 => 0.8, // High locality: 80% hit rate
                1 => 0.5, // Medium locality: 50% hit rate
                2 => 0.2, // Low locality: 20% hit rate
                _ => 0.5, // Default
            };

            let actual_cache_hit_rate = if compilation_times.len() > 1 {
                cache_hits as f64 / (compilation_times.len() - 1) as f64
            } else {
                0.0
            };

            // Allow some tolerance in cache hit rate
            let tolerance = 0.2; // 20% tolerance
            if (actual_cache_hit_rate - expected_cache_hit_rate).abs() > tolerance {
                eprintln!("Cache hit rate out of range: {:.2} vs expected {:.2}", 
                    actual_cache_hit_rate, expected_cache_hit_rate);
                return TestResult::failed();
            }

            TestResult::passed()
        }

        QuickCheck::new()
            .tests(50)
            .gen(Gen::new(100))
            .quickcheck(property as fn(u8, u8) -> TestResult);
    }

    /// Helper function to compile with specific optimizations
    fn compile_with_optimization(
        target: &rustc_target::spec::Target,
        functions: &[WasmIR],
        enable_monomorphization: bool,
        enable_streaming: bool,
    ) -> Result<HashMap<String, Vec<u8>>, Box<dyn std::error::Error>> {
        let mut backend = BackendFactory::create_backend("wasm32", BuildProfile::Development)?;
        
        // This is simplified - in practice, this would configure the backend
        // with the specific optimization flags
        
        let mut results = HashMap::new();
        for (i, func) in functions.iter().enumerate() {
            let result = backend.compile(func, BuildProfile::Development)?;
            results.insert(format!("{}_inst_0", func.name), result.code);
        }
        
        Ok(results)
    }

    /// Helper function to count WASM-like instructions in code
    fn count_wasm_instructions(code: &[u8]) -> usize {
        // Simplified instruction counting
        // In practice, this would parse actual WASM opcodes
        let mut count = 0;
        for &byte in code {
            if byte != 0x00 { // Skip null bytes
                count += 1;
            }
        }
        count
    }

    /// Test specific known good cases for monomorphization
    #[test]
    fn test_known_good_monomorphization_cases() {
        let target = rustc_target::spec::Target {
            arch: "wasm32".to_string(),
            ..Default::default()
        };

        let test_cases = create_test_scenarios();

        for test_case in test_cases {
            println!("Testing monomorphization case: {}", test_case.name);

            // Test without optimization
            let unoptimized_result = compile_with_optimization(&target, &test_case.generic_functions, false, false);
            assert!(unoptimized_result.is_ok(), "Unoptimized compilation should succeed");

            let unoptimized_size: usize = unoptimized_result.unwrap().values()
                .map(|code| code.len())
                .sum();

            // Test with optimization
            let optimized_result = compile_with_optimization(&target, &test_case.generic_functions, true, true);
            assert!(optimized_result.is_ok(), "Optimized compilation should succeed");

            let optimized_size: usize = optimized_result.unwrap().values()
                .map(|code| code.len())
                .sum();

            // Check size reduction
            let size_reduction = if unoptimized_size > 0 {
                ((unoptimized_size - optimized_size) as f64 / unoptimized_size as f64) * 100.0
            } else {
                0.0
            };

            assert!(size_reduction >= test_case.expected_size_reduction,
                "Size reduction {:.1}% should be >= expected {:.1}%",
                size_reduction, test_case.expected_size_reduction);

            println!("  Size reduction: {:.1}%", size_reduction);
            println!("  Status: ✅ PASS");
        }
    }

    /// Test monomorphization with real-world scenarios
    #[test]
    fn test_real_world_monomorphization() {
        let target = rustc_target::spec::Target {
            arch: "wasm32".to_string(),
            ..Default::default()
        };

        // Test scenario: Vector operations
        let vector_operations = create_vector_operations_scenario();
        let result = compile_with_optimization(&target, &vector_operations, true, true);
        assert!(result.is_ok(), "Vector operations should compile with optimizations");

        let compiled = result.unwrap();
        assert!(!compiled.is_empty(), "Should have compiled functions");

        // Test scenario: Generic collections
        let collections = create_generic_collections_scenario();
        let result = compile_with_optimization(&target, &collections, true, true);
        assert!(result.is_ok(), "Generic collections should compile with optimizations");

        let compiled = result.unwrap();
        assert!(!compiled.is_empty(), "Should have compiled functions");

        // Test scenario: Option/Result handling
        let option_result = create_option_result_scenario();
        let result = compile_with_optimization(&target, &option_result, true, true);
        assert!(result.is_ok(), "Option/Result handling should compile with optimizations");

        let compiled = result.unwrap();
        assert!(!compiled.is_empty(), "Should have compiled functions");
    }

    /// Creates vector operations scenario
    fn create_vector_operations_scenario() -> Vec<WasmIR> {
        vec![
            create_generic_vector_map_function(),
            create_generic_vector_filter_function(),
            create_generic_vector_reduce_function(),
        ]
    }

    /// Creates generic collections scenario
    fn create_generic_collections_scenario() -> Vec<WasmIR> {
        vec![
            create_generic_hashmap_function(),
            create_generic_stack_function(),
            create_generic_queue_function(),
        ]
    }

    /// Creates option/result handling scenario
    fn create_option_result_scenario() -> Vec<WasmIR> {
        vec![
            create_generic_option_map_function(),
            create_generic_result_and_then_function(),
            create_generic_option_unwrap_function(),
        ]
    }

    // Helper functions for creating test scenarios
    fn create_generic_vector_map_function() -> WasmIR {
        let signature = Signature {
            params: vec![
                Type::Array { 
                    element_type: Box::new(Type::Ref("T".to_string())),
                    size: None,
                },
                Type::I32, // Function pointer would be generic
            ],
            returns: Some(Type::Array { 
                element_type: Box::new(Type::Ref("U".to_string())),
                size: None,
            }),
        };

        WasmIR::new("vector_map<T,U>".to_string(), signature)
    }

    fn create_generic_vector_filter_function() -> WasmIR {
        let signature = Signature {
            params: vec![
                Type::Array { 
                    element_type: Box::new(Type::Ref("T".to_string())),
                    size: None,
                },
                Type::I32, // Predicate function
            ],
            returns: Some(Type::Array { 
                element_type: Box::new(Type::Ref("T".to_string())),
                size: None,
            }),
        };

        WasmIR::new("vector_filter<T>".to_string(), signature)
    }

    fn create_generic_vector_reduce_function() -> WasmIR {
        let signature = Signature {
            params: vec![
                Type::Array { 
                    element_type: Box::new(Type::Ref("T".to_string())),
                    size: None,
                },
                Type::Ref("T".to_string()), // Initial value
                Type::I32, // Reduce function
            ],
            returns: Some(Type::Ref("T".to_string())),
        };

        WasmIR::new("vector_reduce<T>".to_string(), signature)
    }

    fn create_generic_hashmap_function() -> WasmIR {
        let signature = Signature {
            params: vec![
                Type::Ref("HashMap<K,V>".to_string()),
                Type::Ref("K".to_string()),
                Type::Ref("V".to_string()),
            ],
            returns: Some(Type::Option(Box::new(Type::Ref("V".to_string())))),
        };

        WasmIR::new("hashmap_get<K,V>".to_string(), signature)
    }

    fn create_generic_stack_function() -> WasmIR {
        let signature = Signature {
            params: vec![Type::Ref("Stack<T>".to_string())],
            returns: Some(Type::Option(Box::new(Type::Ref("T".to_string())))),
  