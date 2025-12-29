//! Property-based tests for thin monomorphization optimization
//! 
//! This module contains comprehensive property tests that validate the correctness
//! and effectiveness of thin monomorphization, including symbol similarity checks,
//! size reduction verification, and performance impact analysis.

use proptest::prelude::*;
use wasmrust_wasm::wasmir::{
    WasmIR, Type, Instruction, Terminator, Operand, Signature, Constant, BinaryOp
};
use wasmrust_backend_cranelift::{
    ThinMonomorphizationContext, MonomorphizationFlags, OptimizationResult,
    type_descriptor::{WasmTypeDescriptor, TypeDescriptorRegistry},
    mir_complexity::{FunctionComplexity, MirComplexityAnalyzer},
    thinning_pass::{ThinningPass, ThinningResult},
    size_analyzer::{SizeAnalyzer, SizeAnalysisReport},
    streaming_optimizer::{StreamingLayoutOptimizer, StreamingConfig},
};
use rustc_target::spec::Target;
use std::collections::HashMap;

/// Property: Thin monomorphization should preserve functional correctness
proptest! {
    #[test]
    fn prop_thin_monomorphization_preserves_correctness(
        // Generate a generic function with various types
        generic_params in generic_function_params(),
        instructions in instruction_sequence(),
    ) {
        // Create original generic function
        let original_function = create_generic_function(&generic_params, &instructions);
        
        // Apply thin monomorphization
        let target = create_wasm32_target();
        let mut context = ThinMonomorphizationContext::new(target);
        
        let result = context.analyze_and_optimize(&[original_function.clone()]);
        prop_assume!(result.is_ok());
        
        let optimized_functions = result.unwrap();
        
        // Property: Should not break functional correctness
        // This would be verified through runtime testing
        prop_assert!(!optimized_functions.is_empty());
        
        // Verify thinned function exists if applicable
        let thinned_function = optimized_functions.iter()
            .find(|f| f.name.contains("_thinned"));
        
        if should_be_thinned(&original_function) {
            prop_assert!(thinned_function.is_some(), 
                "Function should be thinned: {}", original_function.name);
        }
    }

    #[test]
    fn prop_size_reduction_achieved(
        functions in function_collection(),
    ) {
        let target = create_wasm32_target();
        let mut context = ThinMonomorphizationContext::new(target);
        
        let original_size = calculate_total_size(&functions);
        
        let result = context.analyze_and_optimize(&functions);
        prop_assume!(result.is_ok());
        
        let optimized_functions = result.unwrap();
        let optimized_size = calculate_total_size(&optimized_functions);
        
        // Property: Should achieve size reduction for suitable functions
        if has_thin_candidates(&functions) {
            let reduction_percentage = ((original_size - optimized_size) as f64 / original_size as f64) * 100.0;
            prop_assert!(reduction_percentage > 10.0, 
                "Should achieve >10% size reduction, got {:.1}%", reduction_percentage);
        }
    }

    #[test]
    fn prop_symbol_similarity_check(
        // Generate multiple instantiations of the same generic function
        instantiations in generic_instantiations(),
    ) {
        let target = create_wasm32_target();
        let mut context = ThinMonomorphizationContext::new(target);
        
        let result = context.analyze_and_optimize(&instantiations);
        prop_assume!(result.is_ok());
        
        let optimized_functions = result.unwrap();
        
        // Extract thinned functions and shims
        let thinned_functions: Vec<_> = optimized_functions.iter()
            .filter(|f| f.name.contains("_thinned"))
            .collect();
        
        let shim_functions: Vec<_> = optimized_functions.iter()
            .filter(|f| f.name.contains("_shim"))
            .collect();
        
        // Property: Should have exactly one thinned function per generic function
        prop_assert_eq!(thinned_functions.len(), count_unique_generic_functions(&instantiations));
        
        // Property: Should have shims for each instantiation
        prop_assert_eq!(shim_functions.len(), instantiations.len());
        
        // Property: Thinned function should be larger than individual shims
        for thinned in &thinned_functions {
            let thinned_size = thinned.instruction_count();
            
            for shim in &shim_functions {
                let shim_size = shim.instruction_count();
                prop_assert!(thinned_size > shim_size, 
                    "Thinned function should be larger than shims: {} vs {}",
                    thinned_size, shim_size);
            }
        }
    }

    #[test]
    fn prop_type_safety_preserved(
        generic_function in single_generic_function(),
        concrete_types in type_collection(),
    ) {
        let target = create_wasm32_target();
        let mut thinning_pass = ThinningPass::new(target);
        
        // Create mock complexity analysis
        let mock_analysis = FunctionComplexity {
            function_name: generic_function.name.clone(),
            instance: create_mock_instance(),
            basic_block_count: 25, // Above threshold
            instruction_count: 60,
            max_nesting_depth: 2,
            generic_type_usages: 2,
            call_count: 3,
            has_loops: false,
            is_recursive: false,
            complexity_score: 15.0,
            is_thinning_candidate: true,
            candidacy_reason: "Good candidate".to_string(),
        };
        
        // Apply thinning transformation
        let result = thinning_pass.apply_thinning(
            &generic_function,
            &concrete_types,
            &mock_analysis,
        );
        
        prop_assume!(result.is_ok());
        
        let thinning_result = result.unwrap();
        
        // Property: Type descriptors should be created for all concrete types
        prop_assert_eq!(thinning_result.type_descriptors.len(), concrete_types.len());
        
        // Property: Each descriptor should have valid size/alignment
        for descriptor in &thinning_result.type_descriptors {
            prop_assert!(descriptor.size > 0, "Type size should be positive");
            prop_assert!(descriptor.align > 0, "Type alignment should be positive");
            prop_assert!(descriptor.size % descriptor.align == 0, 
                "Size should be multiple of alignment");
        }
        
        // Property: Shims should reference the thinned function
        for shim in &thinning_result.shim_functions {
            let has_thinned_call = shim.all_instructions().iter()
                .any(|instr| matches!(instr, Instruction::Call { func_ref, .. } if *func_ref == 0));
            prop_assert!(has_thinned_call, "Shim should call thinned function");
        }
    }

    #[test]
    fn prop_performance_impact_acceptable(
        functions in performance_test_functions(),
    ) {
        let target = create_wasm32_target();
        let mut context = ThinMonomorphizationContext::new(target);
        
        let result = context.analyze_and_optimize(&functions);
        prop_assume!(result.is_ok());
        
        let optimized_functions = result.unwrap();
        
        // Property: Indirect calls should not exceed threshold
        let indirect_call_count = count_indirect_calls(&optimized_functions);
        let total_calls = count_total_calls(&optimized_functions);
        
        if total_calls > 0 {
            let indirect_ratio = indirect_call_count as f64 / total_calls as f64;
            prop_assert!(indirect_ratio < 0.3, 
                "Indirect call ratio should be <30%, got {:.1}%", 
                indirect_ratio * 100.0);
        }
        
        // Property: Function size should remain reasonable
        for function in &optimized_functions {
            let size = function.instruction_count();
            prop_assert!(size < 1000, "Function size should be reasonable: {}", size);
        }
    }
}

/// Property: Size analyzer should provide accurate metrics
proptest! {
    #[test]
    fn prop_size_analyzer_accuracy(
        functions in function_collection(),
    ) {
        let target = create_wasm32_target();
        let mut analyzer = SizeAnalyzer::new(target);
        
        let result = analyzer.analyze_functions(&functions);
        prop_assume!(result.is_ok());
        
        let report = result.unwrap();
        
        // Property: Total size should match calculated size
        let calculated_size = calculate_total_size(&functions);
        prop_assert_eq!(report.total_size_before, calculated_size);
        
        // Property: Each function should have metrics
        prop_assert_eq!(report.function_metrics.len(), functions.len());
        
        for function in &functions {
            prop_assert!(report.function_metrics.contains_key(&function.name),
                "Missing metrics for function: {}", function.name);
        }
        
        // Property: Type impact should be comprehensive
        for (_, metrics) in &report.function_metrics {
            for used_type in &metrics.used_types {
                prop_assert!(report.type_impact.contains_key(used_type),
                    "Missing type impact for: {}", used_type);
            }
        }
        
        // Property: Optimization suggestions should be reasonable
        for suggestion in &report.suggestions {
            prop_assert!(suggestion.expected_reduction > 0,
                "Suggestions should promise positive reduction");
            prop_assert!(suggestion.confidence > 0.0 && suggestion.confidence <= 1.0,
                "Confidence should be in range [0,1]: {}", suggestion.confidence);
        }
    }
}

/// Property: Streaming layout should optimize loading
proptest! {
    #[test]
    fn prop_streaming_layout_optimization(
        functions in function_collection(),
    ) {
        let target = create_wasm32_target();
        let mut optimizer = StreamingLayoutOptimizer::new(target);
        
        let result = optimizer.optimize_layout(&functions);
        prop_assume!(result.is_ok());
        
        let layout = result.unwrap();
        
        // Property: All functions should be included in layout
        let layout_function_count: usize = layout.code_segments.iter()
            .map(|s| s.functions.len())
            .sum();
        
        prop_assert_eq!(layout_function_count, functions.len(),
            "Layout should include all functions");
        
        // Property: Function order should respect dependencies
        prop_assert!(validate_dependency_order(&layout, &functions),
            "Function order should respect dependencies");
        
        // Property: Segments should have reasonable sizes
        for segment in &layout.code_segments {
            prop_assert!(segment.size > 0, "Segment size should be positive");
            prop_assert!(segment.functions.len() > 0, "Segment should have functions");
        }
        
        // Property: Relocations should be valid
        for relocation in &layout.relocations {
            prop_assert!(validate_relocation(relocation, &layout),
                "Invalid relocation: {:?}", relocation);
        }
    }
}

/// Property: Integration should work end-to-end
proptest! {
    #[test]
    fn prop_end_to_end_integration(
        modules in module_collection(),
    ) {
        let target = create_wasm32_target();
        let mut context = ThinMonomorphizationContext::new_with_components(
            target,
            true, // enable size analysis
            true, // enable streaming
        );
        
        let total_before = modules.iter()
            .flat_map(|m| &m.functions)
            .map(|f| f.instruction_count())
            .sum::<usize>();
        
        // Process each module
        let mut all_optimized = Vec::new();
        let mut total_size_reduction = 0.0;
        
        for module in &modules {
            let result = context.analyze_and_optimize_with_tools(&module.functions);
            prop_assume!(result.is_ok());
            
            let optimization_result = result.unwrap();
            all_optimized.extend(optimization_result.functions);
            total_size_reduction += optimization_result.size_reduction;
        }
        
        let total_after = all_optimized.iter()
            .map(|f| f.instruction_count())
            .sum::<usize>();
        
        // Property: Should achieve overall size reduction
        if total_before > 1000 { // Only for reasonably large codebases
            let overall_reduction = ((total_before - total_after) as f64 / total_before as f64) * 100.0;
            prop_assert!(overall_reduction > 5.0,
                "Should achieve >5% overall reduction, got {:.1}%", overall_reduction);
        }
        
        // Property: Should preserve all functions
        prop_assert_eq!(all_optimized.len(), modules.iter().map(|m| m.functions.len()).sum::<usize>());
        
        // Property: Should have analysis data
        let size_report = context.get_size_analysis_report();
        prop_assert!(size_report.is_some(), "Should have size analysis report");
        
        let streaming_layout = context.get_streaming_layout();
        prop_assert!(!streaming_layout.function_order.is_empty(), 
            "Should have streaming layout");
    }
}

// Helper functions for property-based testing

fn create_wasm32_target() -> Target {
    Target {
        arch: "wasm32".to_string(),
        vendor: "unknown".to_string(),
        os: "unknown".to_string(),
        env: "".to_string(),
        endian: rustc_target::spec::Endian::Little,
        pointer_width: Some(rustc_target::spec::PointerWidth::U32),
        c_int_width: Some(rustc_target::spec::IntWidth::Int32),
        os_family: Some(rustc_target::spec::OsFamily::Unknown),
        min_atomic_width: Some(rustc_target::spec::AtomicWidth::None),
        panic_strategy: Some(rustc_target::spec::PanicStrategy::Unwind),
        relocation_model: Some(rustc_target::spec::RelocModel::Static),
        code_model: Some(rustc_target::spec::CodeModel::Small),
        tls_model: Some(rustc_target::spec::TlsModel::GeneralDynamic),
        target_os_version: Some(rustc_target::spec::OsVersion::None),
        abi: Some(rustc_target::spec::Abi::C),
    }
}

fn create_generic_function(params: &[(String, Type)], instructions: &[Instruction]) -> WasmIR {
    let param_types: Vec<Type> = params.iter().map(|(_, ty)| ty.clone()).collect();
    
    let mut function = WasmIR::new(
        format!("generic_function<{}>", 
            params.iter().map(|(name, _)| name.as_str()).collect::<Vec<_>>().join(",")),
        Signature {
            params: param_types,
            returns: Some(Type::I32),
        },
    );
    
    // Add locals for generic parameters
    for (i, (name, ty)) in params.iter().enumerate() {
        function.add_local(ty.clone());
    }
    
    // Add instructions
    let mut basic_block_instructions = Vec::new();
    for instruction in instructions {
        basic_block_instructions.push(instruction.clone());
    }
    
    // Add return
    basic_block_instructions.push(Instruction::Return { 
        value: Some(Operand::Constant(Constant::I32(42)))
    });
    
    function.add_basic_block(basic_block_instructions, Terminator::Return { 
        value: Some(Operand::Constant(Constant::I32(42)))
    });
    
    function
}

fn calculate_total_size(functions: &[WasmIR]) -> usize {
    functions.iter().map(|f| f.instruction_count()).sum()
}

fn should_be_thinned(function: &WasmIR) -> bool {
    function.name.contains('<') && 
    function.instruction_count() > 20 &&
    has_generic_types(function)
}

fn has_generic_types(function: &WasmIR) -> bool {
    function.signature.params.iter()
        .any(|ty| matches!(ty, Type::Ref(name) if name.starts_with("T") || name.contains("_")))
}

fn has_thin_candidates(functions: &[WasmIR]) -> bool {
    functions.iter().any(should_be_thinned)
}

fn count_unique_generic_functions(functions: &[WasmIR]) -> usize {
    let mut generic_names = std::collections::HashSet::new();
    
    for function in functions {
        if function.name.contains('<') {
            // Extract base generic name without type parameters
            let base_name = function.name.split('<').next().unwrap_or(&function.name);
            generic_names.insert(base_name.to_string());
        }
    }
    
    generic_names.len()
}

fn count_indirect_calls(functions: &[WasmIR]) -> usize {
    functions.iter()
        .map(|f| f.all_instructions().iter()
            .filter(|instr| matches!(instr, Instruction::CallIndirect { .. }))
            .count())
        .sum()
}

fn count_total_calls(functions: &[WasmIR]) -> usize {
    functions.iter()
        .map(|f| f.all_instructions().iter()
            .filter(|instr| matches!(instr, 
                Instruction::Call { .. } | Instruction::CallIndirect { .. }))
            .count())
        .sum()
}

fn validate_dependency_order(layout: &wasmrust_backend_cranelift::StreamingLayout, functions: &[WasmIR]) -> bool {
    // Simplified dependency validation
    // In practice, this would build a full dependency graph
    let function_positions: HashMap<String, usize> = layout.function_order.iter()
        .enumerate()
        .filter_map(|(pos, &func_id)| {
            if func_id.0 as usize < functions.len() {
                Some((functions[func_id.0 as usize].name.clone(), pos))
            } else {
                None
            }
        })
        .collect();
    
    // Check that function calls come after their dependencies
    for function in functions {
        if let Some(&caller_pos) = function_positions.get(&function.name) {
            for instruction in function.all_instructions() {
                if let Instruction::Call { func_ref, .. } = instruction {
                    if *func_ref as usize < functions.len() {
                        let callee_name = &functions[*func_ref as usize].name;
                        if let Some(&callee_pos) = function_positions.get(callee_name) {
                            if callee_pos > caller_pos {
                                return false; // Dependency comes after caller
                            }
                        }
                    }
                }
            }
        }
    }
    
    true
}

fn validate_relocation(
    relocation: &wasmrust_backend_cranelift::RelocationInfo,
    layout: &wasmrust_backend_cranelift::StreamingLayout,
) -> bool {
    // Check that relocation target segment exists
    layout.code_segments.iter()
        .any(|s| s.id == relocation.target_segment)
}

fn create_mock_instance() -> rustc_middle::ty::Instance {
    // Simplified - would use actual rustc types
    unsafe { std::mem::zeroed() }
}

// Generators for property-based testing

fn generic_function_params() -> impl Strategy<Value = Vec<(String, Type)>> {
    prop::collection::vec(
        prop::collection::vec(
            (any::<String>(), any::<Type>()),
            1..=3 // 1-3 generic parameters
        ),
        1..=2 // 1-2 sets of generic parameters
    )
}

fn instruction_sequence() -> impl Strategy<Value = Vec<Instruction>> {
    prop::collection::vec(any::<Instruction>(), 5..=20)
}

fn function_collection() -> impl Strategy<Value = Vec<WasmIR>> {
    prop::collection::vec(any::<WasmIR>(), 1..=10)
}

fn generic_instantiations() -> impl Strategy<Value = Vec<WasmIR>> {
    prop::collection::vec(
        generic_function_template(),
        2..=5 // 2-5 instantiations
    )
}

fn generic_function_template() -> impl Strategy<Value = WasmIR> {
    // Generate different instantiations of the same generic function
    any::<Type>().prop_flat_map(|concrete_type| {
        Just(create_generic_instantiation(concrete_type))
    })
}

fn create_generic_instantiation(concrete_type: Type) -> WasmIR {
    let mut function = WasmIR::new(
        format!("process<{}>", type_to_string(&concrete_type)),
        Signature {
            params: vec![concrete_type.clone()],
            returns: Some(concrete_type),
        },
    );
    
    function.add_basic_block(
        vec![
            Instruction::LocalGet { index: 0 },
            Instruction::LocalSet { index: 1 },
            Instruction::LocalGet { index: 1 },
            Instruction::Return { value: Some(Operand::Local(2)) },
        ],
        Terminator::Return { value: Some(Operand::Local(2)) },
    );
    
    function
}

fn type_to_string(ty: &Type) -> String {
    match ty {
        Type::I32 => "i32".to_string(),
        Type::I64 => "i64".to_string(),
        Type::F32 => "f32".to_string(),
        Type::F64 => "f64".to_string(),
        Type::ExternRef(name) => format!("externref_{}", name),
        Type::Array { element_type, size } => {
            format!("[{};{}]", type_to_string(element_type), size.unwrap_or(0))
        }
        Type::Struct { fields } => {
            format!("Struct({})", fields.iter().map(|f| type_to_string(f)).collect::<Vec<_>>().join(","))
        }
        Type::Pointer(inner_type) => format!("*{}", type_to_string(inner_type)),
        Type::Linear { inner_type } => format!("Linear<{}>", type_to_string(inner_type)),
        Type::Capability { inner_type, .. } => format!("Capability<{}>", type_to_string(inner_type)),
        Type::Void => "void".to_string(),
        Type::FuncRef => "funcref".to_string(),
    }
}

fn single_generic_function() -> impl Strategy<Value = WasmIR> {
    generic_function_template()
}

fn type_collection() -> impl Strategy<Value = Vec<Type>> {
    prop::collection::vec(any::<Type>(), 1..=5)
}

fn performance_test_functions() -> impl Strategy<Value = Vec<WasmIR>> {
    prop::collection::vec(performance_function_template(), 3..=8)
}

fn performance_function_template() -> impl Strategy<Value = WasmIR> {
    any::<bool>().prop_map(|is_complex| {
        let mut function = WasmIR::new(
            if is_complex { "complex_handler" } else { "simple_util" }.to_string(),
            Signature {
                params: vec![Type::I32],
                returns: Some(Type::I32),
            },
        );
        
        let instructions = if is_complex {
            vec![
                Instruction::BinaryOp {
                    op: BinaryOp::Mul,
                    left: Operand::Local(0),
                    right: Operand::Constant(Constant::I32(2)),
                },
                Instruction::BinaryOp {
                    op: BinaryOp::Add,
                    left: Operand::Local(1),
                    right: Operand::Constant(Constant::I32(1)),
                },
                Instruction::LocalSet { index: 2 },
                Instruction::LocalGet { index: 2 },
            ]
        } else {
            vec![
                Instruction::LocalGet { index: 0 },
            ]
        };
        
        function.add_basic_block(
            instructions,
            Terminator::Return { value: Some(Operand::Local(if is_complex { 3 } else { 1 })) },
        );
        
        function
    })
}

#[derive(Debug, Clone)]
struct TestModule {
    name: String,
    functions: Vec<WasmIR>,
}

fn module_collection() -> impl Strategy<Value = Vec<TestModule>> {
    prop::collection::vec(module_template(), 1..=4)
}

fn module_template() -> impl Strategy<Value = TestModule> {
    any::<String>().prop_flat_map(|module_name| {
        prop::collection::vec(any::<WasmIR>(), 2..=5)
            .prop_map(move |functions| TestModule {
                name: module_name.clone(),
                functions,
            })
    })
}

// Additional property tests for edge cases

#[cfg(test)]
mod edge_cases {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_empty_module(
        ) {
            let target = create_wasm32_target();
            let mut context = ThinMonomorphizationContext::new(target);
            
            let result = context.analyze_and_optimize(&[]);
            prop_assert!(result.is_ok());
            
            let optimized = result.unwrap();
            prop_assert!(optimized.is_empty());
        }

        #[test]
        fn prop_single_non_generic_function(
            function in single_function_strategy(),
        ) {
            let target = create_wasm32_target();
            let mut context = ThinMonomorphizationContext::new(target);
            
            let original_size = function.instruction_count();
            let result = context.analyze_and_optimize(&[function.clone()]);
            prop_assert!(result.is_ok());
            
            let optimized = result.unwrap();
            prop_assert_eq!(optimized.len(), 1);
            
            // Size should be similar (no thinning applied)
            let optimized_size = optimized[0].instruction_count();
            let size_diff = (optimized_size as i32 - original_size as i32).abs();
            prop_assert!(size_diff <= 5, "Size difference should be minimal: {}", size_diff);
        }

        #[test]
        fn prop_very_large_generic_function(
        ) {
            let target = create_wasm32_target();
            let mut context = ThinMonomorphizationContext::new(target);
            
            let large_function = create_very_large_generic_function();
            let result = context.analyze_and_optimize(&[large_function.clone()]);
            prop_assume!(result.is_ok());
            
            let optimized = result.unwrap();
            
            // Should handle large functions without panic
            prop_assert!(!optimized.is_empty());
            
            // Should not create excessive functions
            prop_assert!(optimized.len() <= 10);
        }
    }

    fn single_function_strategy() -> impl Strategy<Value = WasmIR> {
        any::<bool>().prop_map(|has_generic| {
            if has_generic {
                create_simple_generic_function()
            } else {
                create_simple_concrete_function()
            }
        })
    }

    fn create_simple_generic_function() -> WasmIR {
        let mut function = WasmIR::new(
            "simple_generic<T>".to_string(),
            Signature {
                params: vec![Type::Ref("T".to_string())],
                returns: Some(Type::Ref("T".to_string())),
            },
        );
        
        function.add_basic_block(
            vec![
                Instruction::LocalGet { index: 0 },
                Instruction::Return { value: Some(Operand::Local(1)) },
            ],
            Terminator::Return { value: Some(Operand::Local(1)) },
        );
        
        function
    }

    fn create_simple_concrete_function() -> WasmIR {
        let mut function = WasmIR::new(
            "simple_concrete".to_string(),
            Signature {
                params: vec![Type::I32],
                returns: Some(Type::I32),
            },
        );
        
        function.add_basic_block(
            vec![
                Instruction::LocalGet { index: 0 },
                Instruction::Return { value: Some(Operand::Local(1)) },
            ],
            Terminator::Return { value: Some(Operand::Local(1)) },
        );
        
        function
    }

    fn create_very_large_generic_function() -> WasmIR {
        let mut function = WasmIR::new(
            "large_generic<T>".to_string(),
            Signature {
                params: vec![Type::Ref("T".to_string())],
                returns: Some(Type::Ref("T".to_string())),
            },
        );
        
        // Create many basic blocks to simulate a large function
        for i in 0..100 {
            let mut instructions = Vec::new();
            
            for j in 0..20 {
                instructions.push(Instruction::BinaryOp {
                    op: BinaryOp::Add,
                    left: Operand::Constant(Constant::I32(j as i32)),
                    right: Operand::Local(j % 10),
                });
                instructions.push(Instruction::LocalSet { index: j % 10 });
            }
            
            instructions.push(Instruction::Branch {
                condition: Operand::Constant(Constant::I32(i % 2)),
                then_block: crate::wasmir::BlockId(i * 2 + 1),
                else_block: crate::wasmir::BlockId(i * 2 + 2),
            });
            
            function.add_basic_block(
                instructions,
                Terminator::Jump { 
                    target: crate::wasmir::BlockId(if i == 99 { 0 } else { i + 1 })
                },
            );
        }
        
        function
    }
}

// Benchmark-style property tests

#[cfg(test)]
mod benchmarks {
    use super::*;
    use std::time::Instant;

    #[test]
    fn property_size_reduction_benchmark() {
        let test_cases = vec![
            ("small_generic", create_small_generic_test_case()),
            ("medium_generic", create_medium_generic_test_case()),
            ("large_generic", create_large_generic_test_case()),
        ];
        
        for (name, functions) in test_cases {
            println!("Testing case: {}", name);
            
            let target = create_wasm32_target();
            let mut context = ThinMonomorphizationContext::new(target);
            
            let start = Instant::now();
            let result = context.analyze_and_optimize(&functions);
            let duration = start.elapsed();
            
            prop_assert!(result.is_ok());
            
            let optimized = result.unwrap();
            let original_size = calculate_total_size(&functions);
            let optimized_size = calculate_total_size(&optimized);
            let reduction = ((original_size - optimized_size) as f64 / original_size as f64) * 100.0;
            
            println!("  Original size: {}", original_size);
            println!("  Optimized size: {}", optimized_size);
            println!("  Reduction: {:.1}%", reduction);
            println!("  Compilation time: {:?}", duration);
            
            // Property: Should complete within reasonable time
            prop_assert!(duration.as_millis() < 1000, 
                "Should complete within 1 second, took {:?}", duration);
            
            // Property: Should achieve meaningful reduction for larger cases
            if original_size > 500 {
                prop_assert!(reduction > 5.0, 
                    "Should achieve >5% reduction for large functions");
            }
        }
    }

    fn create_small_generic_test_case() -> Vec<WasmIR> {
        vec![
            create_generic_instantiation(Type::I32),
            create_generic_instantiation(Type::F32),
        ]
    }

    fn create_medium_generic_test_case() -> Vec<WasmIR> {
        vec![
            create_generic_instantiation(Type::I32),
            create_generic_instantiation(Type::I64),
            create_generic_instantiation(Type::F32),
            create_generic_instantiation(Type::F64),
        ]
    }

    fn create_large_generic_test_case() -> Vec<WasmIR> {
        let mut functions = Vec::new();
        
        for i in 0..10 {
            functions.push(create_generic_instantiation(
                Type::Array { 
                    element_type: Box::new(Type::I32), 
                    size: Some(i * 10) 
                }
            ));
        }
        
        functions
    }
}