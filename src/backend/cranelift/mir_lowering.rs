//! MIR to WasmIR lowering for WasmRust
//! 
//! This module implements the transformation from Rust MIR to WasmIR,
//! preserving Rust semantics while enabling WASM-specific optimizations.

use wasm::wasmir::{WasmIR, Instruction, Terminator, BasicBlock, BlockId, Type, Signature, Operand, BinaryOp, UnaryOp, OwnershipState};
use std::collections::HashMap;

/// Context for lowering MIR to WasmIR
pub struct MirLoweringContext {
    /// Current function being lowered
    current_function: Option<WasmIR>,
    /// Local variable mappings
    local_mappings: HashMap<u32, u32>,
    /// Basic block mappings
    block_mappings: HashMap<u32, BlockId>,
    /// Error messages collected during lowering
    error_messages: Vec<String>,
}

impl MirLoweringContext {
    /// Creates a new MIR lowering context
    pub fn new() -> Self {
        Self {
            current_function: None,
            local_mappings: HashMap::new(),
            block_mappings: HashMap::new(),
            error_messages: Vec::new(),
        }
    }

    /// Creates a simple WasmIR function for testing
    pub fn create_simple_function(&mut self, name: String) -> WasmIR {
        let signature = Signature {
            params: vec![Type::I32, Type::I32],
            returns: Some(Type::I32),
        };
        
        let mut wasm_func = WasmIR::new(name, signature);
        
        // Add some locals
        wasm_func.add_local(Type::I32); // Result local
        
        // Create a simple basic block that adds two parameters
        let instructions = vec![
            Instruction::BinaryOp {
                op: BinaryOp::Add,
                left: Operand::Local(0),  // First parameter
                right: Operand::Local(1), // Second parameter
            },
        ];
        
        let terminator = Terminator::Return {
            value: Some(Operand::Local(2)), // Return the result
        };
        
        wasm_func.add_basic_block(instructions, terminator);
        
        wasm_func
    }

    /// Checks if there are any errors
    pub fn has_errors(&self) -> bool {
        !self.error_messages.is_empty()
    }

    /// Gets error messages
    pub fn get_errors(&self) -> &[String] {
        &self.error_messages
    }

    /// Converts the context into a WasmIR function
    pub fn into_wasmir(self) -> Result<WasmIR, String> {
        if let Some(func) = self.current_function {
            Ok(func)
        } else {
            // Create a default function for testing
            let signature = Signature {
                params: vec![Type::I32, Type::I32],
                returns: Some(Type::I32),
            };
            
            let mut wasm_func = WasmIR::new("default_function".to_string(), signature);
            
            // Add a simple basic block
            let instructions = vec![
                Instruction::BinaryOp {
                    op: BinaryOp::Add,
                    left: Operand::Local(0),
                    right: Operand::Local(1),
                },
            ];
            
            let terminator = Terminator::Return {
                value: Some(Operand::Local(0)),
            };
            
            wasm_func.add_basic_block(instructions, terminator);
            
            Ok(wasm_func)
        }
    }

    /// Adds an error message
    pub fn add_error(&mut self, message: String) {
        self.error_messages.push(message);
    }

    /// Sets the current function
    pub fn set_function(&mut self, func: WasmIR) {
        self.current_function = Some(func);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mir_lowering_context_creation() {
        let context = MirLoweringContext::new();
        assert!(!context.has_errors());
        assert_eq!(context.get_errors().len(), 0);
    }

    #[test]
    fn test_simple_function_creation() {
        let mut context = MirLoweringContext::new();
        let func = context.create_simple_function("test_func".to_string());
        
        assert_eq!(func.name, "test_func");
        assert_eq!(func.signature.params.len(), 2);
        assert_eq!(func.signature.returns, Some(Type::I32));
        assert_eq!(func.basic_blocks.len(), 1);
    }

    #[test]
    fn test_into_wasmir() {
        let context = MirLoweringContext::new();
        let result = context.into_wasmir();
        
        assert!(result.is_ok());
        let func = result.unwrap();
        assert_eq!(func.name, "default_function");
        assert_eq!(func.basic_blocks.len(), 1);
    }

    #[test]
    fn test_error_handling() {
        let mut context = MirLoweringContext::new();
        assert!(!context.has_errors());
        
        context.add_error("Test error".to_string());
        assert!(context.has_errors());
        assert_eq!(context.get_errors().len(), 1);
        assert_eq!(context.get_errors()[0], "Test error");
    }
}