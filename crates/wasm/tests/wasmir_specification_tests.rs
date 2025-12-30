//! Comprehensive test cases for WasmIR specification
//! 
//! This module contains test cases that validate the WasmIR specification
//! implementation, covering all major features and edge cases.

use wasm::wasmir::{
    WasmIR, Instruction, Terminator, BasicBlock, BlockId, Type, Signature, Operand, 
    BinaryOp, UnaryOp, Constant, AtomicOp, LinearOp, MemoryOrder, Capability,
    OwnershipAnnotation, OwnershipState, SourceLocation, ValidationError
};

/// Test basic WasmIR function creation and validation
#[test]
fn test_basic_function_creation() {
    let signature = Signature {
        params: vec![Type::I32, Type::I32],
        returns: Some(Type::I32),
    };
    
    let mut func = WasmIR::new("test_add".to_string(), signature.clone());
    assert_eq!(func.name, "test_add");
    assert_eq!(func.signature, signature);
    assert_eq!(func.basic_blocks.len(), 0);
    assert_eq!(func.locals.len(), 0);
    assert_eq!(func.capabilities.len(), 0);
}

/// Test arithmetic operations
#[test]
fn test_arithmetic_operations() {
    let signature = Signature {
        params: vec![Type::I32, Type::I32],
        returns: Some(Type::I32),
    };
    
    let mut func = WasmIR::new("arithmetic_test".to_string(), signature);
    let result_local = func.add_local(Type::I32);
    
    // Test all binary operations
    let operations = vec![
        BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div, BinaryOp::Mod,
        BinaryOp::And, BinaryOp::Or, BinaryOp::Xor,
        BinaryOp::Shl, BinaryOp::Shr, BinaryOp::Sar,
        BinaryOp::Eq, BinaryOp::Ne, BinaryOp::Lt, BinaryOp::Le, BinaryOp::Gt, BinaryOp::Ge,
    ];
    
    for (i, op) in operations.iter().enumerate() {
        let instructions = vec![
            Instruction::BinaryOp {
                op: *op,
                left: Operand::Local(0),   // First parameter
                right: Operand::Local(1),  // Second parameter
            },
            Instruction::LocalSet {
                index: result_local,
                value: Operand::Local(0),  // Result of operation
            },
        ];
        
        let terminator = Terminator::Return {
            value: Some(Operand::Local(result_local)),
        };
        
        func.add_basic_block(instructions, terminator);
    }
    
    assert_eq!(func.basic_blocks.len(), operations.len());
    
    // Validate the function
    let validation_result = func.validate();
    assert!(validation_result.is_ok(), "Validation failed: {:?}", validation_result);
}

/// Test memory operations
#[test]
fn test_memory_operations() {
    let signature = Signature {
        params: vec![Type::I32, Type::I32],  // ptr, value
        returns: Some(Type::I32),
    };
    
    let mut func = WasmIR::new("memory_test".to_string(), signature);
    let loaded_local = func.add_local(Type::I32);
    
    let instructions = vec![
        // Store value to memory
        Instruction::MemoryStore {
            address: Operand::Local(0),  // ptr
            value: Operand::Local(1),    // value
            ty: Type::I32,
            align: Some(4),
            offset: 0,
        },
        // Load value from memory
        Instruction::MemoryLoad {
            address: Operand::Local(0),  // ptr
            ty: Type::I32,
            align: Some(4),
            offset: 0,
        },
        Instruction::LocalSet {
            index: loaded_local,
            value: Operand::Local(0),  // Result of load
        },
    ];
    
    let terminator = Terminator::Return {
        value: Some(Operand::Local(loaded_local)),
    };
    
    func.add_basic_block(instructions, terminator);
    
    let validation_result = func.validate();
    assert!(validation_result.is_ok(), "Memory operations validation failed: {:?}", validation_result);
}

/// Test control flow with branches
#[test]
fn test_control_flow() {
    let signature = Signature {
        params: vec![Type::I32, Type::I32],
        returns: Some(Type::I32),
    };
    
    let mut func = WasmIR::new("control_flow_test".to_string(), signature);
    let condition_local = func.add_local(Type::I32);
    let result_local = func.add_local(Type::I32);
    
    // Block 0: Compare and branch
    let block0_instructions = vec![
        Instruction::BinaryOp {
            op: BinaryOp::Gt,
            left: Operand::Local(0),   // First parameter
            right: Operand::Local(1),  // Second parameter
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
    
    // Block 1: True branch
    let block1_instructions = vec![
        Instruction::LocalSet {
            index: result_local,
            value: Operand::Local(0),  // Return first parameter
        },
    ];
    let block1_terminator = Terminator::Jump { target: BlockId(3) };
    func.add_basic_block(block1_instructions, block1_terminator);
    
    // Block 2: False branch
    let block2_instructions = vec![
        Instruction::LocalSet {
            index: result_local,
            value: Operand::Local(1),  // Return second parameter
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
    
    assert_eq!(func.basic_blocks.len(), 4);
    
    let validation_result = func.validate();
    assert!(validation_result.is_ok(), "Control flow validation failed: {:?}", validation_result);
}

/// Test JavaScript interop with ExternRef
#[test]
fn test_js_interop() {
    let signature = Signature {
        params: vec![Type::ExternRef("JsObject".to_string())],
        returns: Some(Type::I32),
    };
    
    let mut func = WasmIR::new("js_interop_test".to_string(), signature);
    func.add_capability(Capability::JsInterop);
    
    let result_local = func.add_local(Type::I32);
    
    let instructions = vec![
        Instruction::CapabilityCheck {
            capability: Capability::JsInterop,
        },
        Instruction::JSMethodCall {
            object: Operand::Local(0),  // ExternRef parameter
            method: "getValue".to_string(),
            args: vec![],
            return_type: Some(Type::I32),
        },
        Instruction::LocalSet {
            index: result_local,
            value: Operand::Local(0),  // Result of JS call
        },
    ];
    
    let terminator = Terminator::Return {
        value: Some(Operand::Local(result_local)),
    };
    
    func.add_basic_block(instructions, terminator);
    
    assert!(func.capabilities.contains(&Capability::JsInterop));
    
    let validation_result = func.validate();
    assert!(validation_result.is_ok(), "JS interop validation failed: {:?}", validation_result);
}

/// Test atomic operations for threading
#[test]
fn test_atomic_operations() {
    let signature = Signature {
        params: vec![Type::I32],  // counter pointer
        returns: Some(Type::I32),
    };
    
    let mut func = WasmIR::new("atomic_test".to_string(), signature);
    func.add_capability(Capability::Threading);
    func.add_capability(Capability::AtomicMemory);
    
    let old_value_local = func.add_local(Type::I32);
    
    let instructions = vec![
        Instruction::CapabilityCheck {
            capability: Capability::AtomicMemory,
        },
        Instruction::AtomicOp {
            op: AtomicOp::Add,
            address: Operand::Local(0),  // counter pointer
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
    
    assert!(func.capabilities.contains(&Capability::Threading));
    assert!(func.capabilities.contains(&Capability::AtomicMemory));
    
    let validation_result = func.validate();
    assert!(validation_result.is_ok(), "Atomic operations validation failed: {:?}", validation_result);
}

/// Test linear types and ownership annotations
#[test]
fn test_linear_types() {
    let signature = Signature {
        params: vec![Type::Linear {
            inner_type: Box::new(Type::ExternRef("FileHandle".to_string())),
        }],
        returns: Some(Type::I32),
    };
    
    let mut func = WasmIR::new("linear_test".to_string(), signature);
    
    // Add ownership annotation
    func.add_ownership_annotation(OwnershipAnnotation {
        variable: 0,  // handle parameter
        state: OwnershipState::Owned,
        source_location: SourceLocation {
            file: "test.rs".to_string(),
            line: 1,
            column: 1,
        },
    });
    
    let result_local = func.add_local(Type::I32);
    
    let instructions = vec![
        Instruction::LinearOp {
            op: LinearOp::Consume,
            value: Operand::Local(0),  // Consume the linear handle
        },
        Instruction::LocalSet {
            index: result_local,
            value: Operand::Constant(Constant::I32(0)),  // Success
        },
    ];
    
    let terminator = Terminator::Return {
        value: Some(Operand::Local(result_local)),
    };
    
    func.add_basic_block(instructions, terminator);
    
    assert_eq!(func.ownership_annotations.len(), 1);
    assert_eq!(func.ownership_annotations[0].variable, 0);
    assert_eq!(func.ownership_annotations[0].state, OwnershipState::Owned);
    
    let validation_result = func.validate();
    assert!(validation_result.is_ok(), "Linear types validation failed: {:?}", validation_result);
}

/// Test component model operations
#[test]
fn test_component_model() {
    let signature = Signature {
        params: vec![Type::I32, Type::I32],  // data_ptr, data_len
        returns: Some(Type::I32),            // result_ptr
    };
    
    let mut func = WasmIR::new("component_test".to_string(), signature);
    func.add_capability(Capability::ComponentModel);
    
    let result_local = func.add_local(Type::I32);
    
    let instructions = vec![
        Instruction::CapabilityCheck {
            capability: Capability::ComponentModel,
        },
        Instruction::Call {
            func_ref: 1,  // Some internal function
            args: vec![
                Operand::Local(0),  // data_ptr
                Operand::Local(1),  // data_len
            ],
        },
        Instruction::LocalSet {
            index: result_local,
            value: Operand::Local(0),  // Result of call
        },
    ];
    
    let terminator = Terminator::Return {
        value: Some(Operand::Local(result_local)),
    };
    
    func.add_basic_block(instructions, terminator);
    
    assert!(func.capabilities.contains(&Capability::ComponentModel));
    
    let validation_result = func.validate();
    assert!(validation_result.is_ok(), "Component model validation failed: {:?}", validation_result);
}

/// Test validation error cases
#[test]
fn test_validation_errors() {
    let signature = Signature {
        params: vec![Type::I32],
        returns: Some(Type::I32),
    };
    
    let mut func = WasmIR::new("validation_error_test".to_string(), signature);
    
    // Test invalid local index
    let instructions = vec![
        Instruction::LocalGet { index: 999 },  // Invalid index
    ];
    
    let terminator = Terminator::Return { value: None };
    func.add_basic_block(instructions, terminator);
    
    let validation_result = func.validate();
    assert!(validation_result.is_err());
    
    match validation_result.unwrap_err() {
        ValidationError::InvalidLocalIndex(idx) => assert_eq!(idx, 999),
        _ => panic!("Expected InvalidLocalIndex error"),
    }
}

/// Test invalid block ID references
#[test]
fn test_invalid_block_references() {
    let signature = Signature {
        params: vec![Type::I32],
        returns: Some(Type::I32),
    };
    
    let mut func = WasmIR::new("invalid_block_test".to_string(), signature);
    
    let instructions = vec![];
    let terminator = Terminator::Jump { target: BlockId(999) };  // Invalid block ID
    func.add_basic_block(instructions, terminator);
    
    let validation_result = func.validate();
    assert!(validation_result.is_err());
    
    match validation_result.unwrap_err() {
        ValidationError::InvalidBlockId(_) => {}, // Expected
        _ => panic!("Expected InvalidBlockId error"),
    }
}

/// Test function analysis utilities
#[test]
fn test_function_analysis() {
    let signature = Signature {
        params: vec![Type::I32, Type::I32],
        returns: Some(Type::I32),
    };
    
    let mut func = WasmIR::new("analysis_test".to_string(), signature);
    
    let local1 = func.add_local(Type::I32);
    let local2 = func.add_local(Type::I32);
    let _local3 = func.add_local(Type::I32);  // Unused local
    
    let instructions = vec![
        Instruction::LocalGet { index: local1 },
        Instruction::BinaryOp {
            op: BinaryOp::Add,
            left: Operand::Local(0),   // Parameter
            right: Operand::Local(local1),
        },
        Instruction::LocalSet {
            index: local2,
            value: Operand::Local(0),  // Result of add
        },
    ];
    
    let terminator = Terminator::Return {
        value: Some(Operand::Local(local2)),
    };
    
    func.add_basic_block(instructions, terminator);
    
    // Test instruction count
    assert_eq!(func.instruction_count(), 3);
    
    // Test used locals analysis
    let used_locals = func.used_locals();
    assert!(used_locals.contains(&0));      // Parameter
    assert!(used_locals.contains(&local1)); // Used local
    assert!(used_locals.contains(&local2)); // Used local
    assert!(!used_locals.contains(&_local3)); // Unused local
    
    // Test entry block
    let entry = func.entry_block();
    assert!(entry.is_some());
    assert_eq!(entry.unwrap().id, BlockId(0));
}

/// Test complex control flow with switch statement
#[test]
fn test_switch_statement() {
    let signature = Signature {
        params: vec![Type::I32],
        returns: Some(Type::I32),
    };
    
    let mut func = WasmIR::new("switch_test".to_string(), signature);
    let result_local = func.add_local(Type::I32);
    
    // Block 0: Switch statement
    let block0_instructions = vec![];
    let block0_terminator = Terminator::Switch {
        value: Operand::Local(0),  // Switch on parameter
        targets: vec![
            (Operand::Constant(Constant::I32(1)), BlockId(1)),
            (Operand::Constant(Constant::I32(2)), BlockId(2)),
            (Operand::Constant(Constant::I32(3)), BlockId(3)),
        ],
        default_target: BlockId(4),
    };
    func.add_basic_block(block0_instructions, block0_terminator);
    
    // Case 1
    let block1_instructions = vec![
        Instruction::LocalSet {
            index: result_local,
            value: Operand::Constant(Constant::I32(10)),
        },
    ];
    let block1_terminator = Terminator::Jump { target: BlockId(5) };
    func.add_basic_block(block1_instructions, block1_terminator);
    
    // Case 2
    let block2_instructions = vec![
        Instruction::LocalSet {
            index: result_local,
            value: Operand::Constant(Constant::I32(20)),
        },
    ];
    let block2_terminator = Terminator::Jump { target: BlockId(5) };
    func.add_basic_block(block2_instructions, block2_terminator);
    
    // Case 3
    let block3_instructions = vec![
        Instruction::LocalSet {
            index: result_local,
            value: Operand::Constant(Constant::I32(30)),
        },
    ];
    let block3_terminator = Terminator::Jump { target: BlockId(5) };
    func.add_basic_block(block3_instructions, block3_terminator);
    
    // Default case
    let block4_instructions = vec![
        Instruction::LocalSet {
            index: result_local,
            value: Operand::Constant(Constant::I32(0)),
        },
    ];
    let block4_terminator = Terminator::Jump { target: BlockId(5) };
    func.add_basic_block(block4_instructions, block4_terminator);
    
    // Return block
    let block5_instructions = vec![];
    let block5_terminator = Terminator::Return {
        value: Some(Operand::Local(result_local)),
    };
    func.add_basic_block(block5_instructions, block5_terminator);
    
    assert_eq!(func.basic_blocks.len(), 6);
    
    let validation_result = func.validate();
    assert!(validation_result.is_ok(), "Switch statement validation failed: {:?}", validation_result);
}

/// Test all constant types
#[test]
fn test_constant_types() {
    let signature = Signature {
        params: vec![],
        returns: Some(Type::I32),
    };
    
    let mut func = WasmIR::new("constants_test".to_string(), signature);
    
    let constants = vec![
        Constant::I32(42),
        Constant::I64(1234567890),
        Constant::F32(3.14),
        Constant::F64(2.718281828),
        Constant::Boolean(true),
        Constant::Boolean(false),
        Constant::Null,
        Constant::String("hello".to_string()),
    ];
    
    for (i, constant) in constants.iter().enumerate() {
        let local = func.add_local(Type::I32);
        let instructions = vec![
            Instruction::LocalSet {
                index: local,
                value: Operand::Constant(constant.clone()),
            },
        ];
        
        let terminator = if i == constants.len() - 1 {
            Terminator::Return {
                value: Some(Operand::Local(local)),
            }
        } else {
            Terminator::Jump { target: BlockId(i + 1) }
        };
        
        func.add_basic_block(instructions, terminator);
    }
    
    let validation_result = func.validate();
    assert!(validation_result.is_ok(), "Constants validation failed: {:?}", validation_result);
}

/// Test all unary operations
#[test]
fn test_unary_operations() {
    let signature = Signature {
        params: vec![Type::I32],
        returns: Some(Type::I32),
    };
    
    let mut func = WasmIR::new("unary_test".to_string(), signature);
    
    let unary_ops = vec![
        UnaryOp::Neg,
        UnaryOp::Not,
        UnaryOp::Clz,
        UnaryOp::Ctz,
        UnaryOp::Popcnt,
    ];
    
    for (i, op) in unary_ops.iter().enumerate() {
        let result_local = func.add_local(Type::I32);
        let instructions = vec![
            Instruction::UnaryOp {
                op: *op,
                value: Operand::Local(0),  // Parameter
            },
            Instruction::LocalSet {
                index: result_local,
                value: Operand::Local(0),  // Result of unary op
            },
        ];
        
        let terminator = if i == unary_ops.len() - 1 {
            Terminator::Return {
                value: Some(Operand::Local(result_local)),
            }
        } else {
            Terminator::Jump { target: BlockId(i + 1) }
        };
        
        func.add_basic_block(instructions, terminator);
    }
    
    let validation_result = func.validate();
    assert!(validation_result.is_ok(), "Unary operations validation failed: {:?}", validation_result);
}

/// Test capability system
#[test]
fn test_capability_system() {
    let signature = Signature {
        params: vec![Type::I32],
        returns: Some(Type::I32),
    };
    
    let mut func = WasmIR::new("capability_test".to_string(), signature);
    
    // Add multiple capabilities
    let capabilities = vec![
        Capability::JsInterop,
        Capability::Threading,
        Capability::AtomicMemory,
        Capability::ComponentModel,
        Capability::MemoryRegion("us-east-1".to_string()),
        Capability::Custom("custom_feature".to_string()),
    ];
    
    for cap in &capabilities {
        func.add_capability(cap.clone());
    }
    
    assert_eq!(func.capabilities.len(), capabilities.len());
    
    for cap in &capabilities {
        assert!(func.capabilities.contains(cap));
    }
    
    // Test capability checks in instructions
    let instructions = vec![
        Instruction::CapabilityCheck {
            capability: Capability::JsInterop,
        },
        Instruction::CapabilityCheck {
            capability: Capability::Threading,
        },
    ];
    
    let terminator = Terminator::Return {
        value: Some(Operand::Constant(Constant::I32(0))),
    };
    
    func.add_basic_block(instructions, terminator);
    
    let validation_result = func.validate();
    assert!(validation_result.is_ok(), "Capability system validation failed: {:?}", validation_result);
}

/// Benchmark test for large functions
#[test]
fn test_large_function_performance() {
    let signature = Signature {
        params: vec![Type::I32],
        returns: Some(Type::I32),
    };
    
    let mut func = WasmIR::new("large_function_test".to_string(), signature);
    
    // Create a function with many basic blocks and instructions
    let num_blocks = 1000;
    let instructions_per_block = 10;
    
    for block_idx in 0..num_blocks {
        let mut instructions = Vec::new();
        
        for inst_idx in 0..instructions_per_block {
            let local = func.add_local(Type::I32);
            instructions.push(Instruction::BinaryOp {
                op: BinaryOp::Add,
                left: Operand::Local(0),  // Parameter
                right: Operand::Constant(Constant::I32(inst_idx as i32)),
            });
            instructions.push(Instruction::LocalSet {
                index: local,
                value: Operand::Local(0),  // Result of add
            });
        }
        
        let terminator = if block_idx == num_blocks - 1 {
            Terminator::Return {
                value: Some(Operand::Local(0)),
            }
        } else {
            Terminator::Jump { target: BlockId(block_idx + 1) }
        };
        
        func.add_basic_block(instructions, terminator);
    }
    
    assert_eq!(func.basic_blocks.len(), num_blocks);
    assert_eq!(func.instruction_count(), num_blocks * instructions_per_block * 2);
    
    // Measure validation time
    let start = std::time::Instant::now();
    let validation_result = func.validate();
    let validation_time = start.elapsed();
    
    assert!(validation_result.is_ok(), "Large function validation failed: {:?}", validation_result);
    println!("Validation time for {} blocks: {:?}", num_blocks, validation_time);
    
    // Validation should complete in reasonable time (< 100ms for 1000 blocks)
    assert!(validation_time.as_millis() < 100, "Validation took too long: {:?}", validation_time);
}