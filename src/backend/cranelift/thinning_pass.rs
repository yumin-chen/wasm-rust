//! WasmIR Thinning Pass
//! 
//! This module implements the core thinning transformation that converts
//! type-aware generic functions into pointer-based thinned functions
//! with type descriptors for runtime type information.

use crate::wasmir::{
    WasmIR, Type, Instruction, Terminator, Operand, 
    Signature, BasicBlock, BlockId, Constant, BinaryOp
};
use crate::backend::cranelift::{
    type_descriptor::{WasmTypeDescriptor, TypeDescriptorRegistry},
    mir_complexity::{FunctionComplexity, MirComplexityAnalyzer},
};
use rustc_middle::ty::{TyS, TyKind};
use rustc_middle::mir::{Body, Place};
use rustc_target::spec::Target;
use std::collections::{HashMap, HashSet};

/// Thinning transformation result
#[derive(Debug, Clone)]
pub struct ThinningResult {
    /// Thinned main function (shared across all instances)
    pub thinned_function: WasmIR,
    /// Generated shim functions (one per concrete type)
    pub shim_functions: Vec<WasmIR>,
    /// Type descriptors for each concrete type
    pub type_descriptors: Vec<WasmTypeDescriptor>,
    /// Function table indices for type-specific functions
    pub function_table: HashMap<String, u32>,
    /// Size reduction achieved
    pub size_reduction: f64,
}

/// Thinning pass context
pub struct ThinningPass {
    /// Target architecture
    target: Target,
    /// Type descriptor registry
    type_registry: TypeDescriptorRegistry,
    /// MIR complexity analyzer
    complexity_analyzer: MirComplexityAnalyzer,
    /// Function being processed
    current_function: Option<String>,
    /// Generic type substitutions
    substitutions: HashMap<String, Type>,
    /// Next available local variable index
    next_local_index: u32,
    /// Next available function index
    next_function_index: u32,
}

impl ThinningPass {
    /// Creates a new thinning pass
    pub fn new(target: Target) -> Self {
        Self {
            target,
            type_registry: TypeDescriptorRegistry::new(),
            complexity_analyzer: MirComplexityAnalyzer::new(target.clone()),
            current_function: None,
            substitutions: HashMap::new(),
            next_local_index: 0,
            next_function_index: 0,
        }
    }

    /// Applies thinning transformation to a generic function
    pub fn apply_thinning(
        &mut self,
        generic_function: &WasmIR,
        concrete_types: &[Type],
        complexity_analysis: &FunctionComplexity,
    ) -> Result<ThinningResult, ThinningError> {
        self.current_function = Some(generic_function.name.clone());
        
        // 1. Analyze generic function and create type descriptors
        let type_descriptors = self.create_type_descriptors(concrete_types)?;
        
        // 2. Create thinned function body
        let thinned_function = self.create_thinned_function(generic_function, &type_descriptors)?;
        
        // 3. Generate shim functions for each concrete type
        let shim_functions = self.create_shim_functions(
            generic_function,
            concrete_types,
            &type_descriptors,
        )?;
        
        // 4. Calculate size reduction
        let original_size = self.estimate_function_size(generic_function);
        let thinned_size = self.estimate_function_size(&thinned_function) + 
                           shim_functions.iter().map(|f| self.estimate_function_size(f)).sum::<usize>();
        let size_reduction = if original_size > 0 {
            ((original_size - thinned_size) as f64 / original_size as f64) * 100.0
        } else {
            0.0
        };
        
        // 5. Build function table
        let function_table = self.build_function_table(&type_descriptors);
        
        self.current_function = None;
        
        Ok(ThinningResult {
            thinned_function,
            shim_functions,
            type_descriptors,
            function_table,
            size_reduction,
        })
    }

    /// Creates type descriptors for concrete types
    fn create_type_descriptors(
        &mut self,
        concrete_types: &[Type],
    ) -> Result<Vec<WasmTypeDescriptor>, ThinningError> {
        let mut descriptors = Vec::new();
        
        for (i, concrete_type) in concrete_types.iter().enumerate() {
            let type_id = (i + 1) as u32;
            let type_name = self.type_to_string(concrete_type);
            let (size, align) = self.calculate_type_size_align(concrete_type)?;
            
            let mut descriptor = WasmTypeDescriptor::new(type_id, size, align, type_name);
            
            // Determine type characteristics
            descriptor = descriptor.with_copy(self.is_type_copy(concrete_type));
            descriptor = descriptor.with_pod(self.is_type_pod(concrete_type));
            
            // Generate drop glue if needed
            if self.type_needs_drop(concrete_type) {
                let drop_fn_id = self.generate_drop_glue(concrete_type)?;
                descriptor = descriptor.with_drop_glue(drop_fn_id);
            }
            
            descriptors.push(descriptor);
        }
        
        Ok(descriptors)
    }

    /// Creates the main thinned function
    fn create_thinned_function(
        &mut self,
        generic_function: &WasmIR,
        type_descriptors: &[WasmTypeDescriptor],
    ) -> Result<WasmIR, ThinningError> {
        let mut thinned_function = WasmIR::new(
            format!("{}_thinned", generic_function.name),
            self.create_thinned_signature(generic_function, type_descriptors),
        );
        
        // Reset local indices for new function
        self.next_local_index = 0;
        
        // Add locals for thinned parameters
        let item_ptr_local = thinned_function.add_local(Type::I32); // Opaque pointer
        let desc_ptr_local = thinned_function.add_local(Type::I32); // Descriptor pointer
        let temp_locals = self.add_temporary_locals(&mut thinned_function, generic_function);
        
        // Transform basic blocks
        for basic_block in &generic_function.basic_blocks {
            let transformed_instructions = self.transform_instructions(
                &basic_block.instructions,
                item_ptr_local,
                desc_ptr_local,
                &temp_locals,
            )?;
            
            let transformed_terminator = self.transform_terminator(
                &basic_block.terminator,
                item_ptr_local,
                desc_ptr_local,
                &temp_locals,
            )?;
            
            thinned_function.add_basic_block(transformed_instructions, transformed_terminator);
        }
        
        Ok(thinned_function)
    }

    /// Creates thinned function signature
    fn create_thinned_signature(
        &self,
        generic_function: &WasmIR,
        _type_descriptors: &[WasmTypeDescriptor],
    ) -> Signature {
        // Thinned signature: (item_ptr: i32, desc_ptr: i32) -> original_return_type
        let mut params = vec![Type::I32, Type::I32]; // item_ptr, desc_ptr
        
        // Preserve original return type
        let returns = generic_function.signature.returns.clone();
        
        Signature { params, returns }
    }

    /// Adds temporary locals for thinned function
    fn add_temporary_locals(
        &mut self,
        thinned_function: &mut WasmIR,
        generic_function: &WasmIR,
    ) -> HashMap<String, u32> {
        let mut temp_locals = HashMap::new();
        
        // Add locals that might be needed during transformation
        for local_type in &generic_function.locals {
            if matches!(local_type, Type::Ref(_)) {
                // Original generic parameter locals become opaque pointers
                temp_locals.insert("generic_local".to_string(), 
                    thinned_function.add_local(Type::I32));
            }
        }
        
        // Add local for function pointers loaded from descriptor
        temp_locals.insert("function_ptr".to_string(), 
            thinned_function.add_local(Type::I32));
        
        // Add local for temporary calculations
        temp_locals.insert("temp".to_string(), 
            thinned_function.add_local(Type::I32));
        
        temp_locals
    }

    /// Transforms instructions for thinned context
    fn transform_instructions(
        &mut self,
        instructions: &[Instruction],
        item_ptr_local: u32,
        desc_ptr_local: u32,
        temp_locals: &HashMap<String, u32>,
    ) -> Result<Vec<Instruction>, ThinningError> {
        let mut transformed = Vec::new();
        
        for instruction in instructions {
            let transformed_instr = self.transform_instruction(
                instruction,
                item_ptr_local,
                desc_ptr_local,
                temp_locals,
            )?;
            transformed.push(transformed_instr);
        }
        
        Ok(transformed)
    }

    /// Transforms a single instruction
    fn transform_instruction(
        &mut self,
        instruction: &Instruction,
        item_ptr_local: u32,
        desc_ptr_local: u32,
        temp_locals: &HashMap<String, u32>,
    ) -> Result<Instruction, ThinningError> {
        match instruction {
            Instruction::LocalGet { index } => {
                // Transform generic parameter accesses
                if self.is_generic_parameter(*index) {
                    Ok(Instruction::LocalGet { index: item_ptr_local })
                } else {
                    Ok(instruction.clone())
                }
            }
            
            Instruction::MemoryLoad { address, ty, align, offset } => {
                // Transform generic type loads
                if self.is_generic_type(ty) {
                    let transformed_address = self.transform_operand(
                        address, item_ptr_local, desc_ptr_local, temp_locals
                    )?;
                    Ok(Instruction::MemoryLoad {
                        address: transformed_address,
                        ty: Type::I32, // Load as opaque pointer/bytes
                        align: *align,
                        offset: *offset,
                    })
                } else {
                    Ok(instruction.clone())
                }
            }
            
            Instruction::MemoryStore { address, value, ty, align, offset } => {
                // Transform generic type stores
                if self.is_generic_type(ty) {
                    let transformed_address = self.transform_operand(
                        address, item_ptr_local, desc_ptr_local, temp_locals
                    )?;
                    let transformed_value = self.transform_operand(
                        value, item_ptr_local, desc_ptr_local, temp_locals
                    )?;
                    Ok(Instruction::MemoryStore {
                        address: transformed_address,
                        value: transformed_value,
                        ty: Type::I32, // Store as opaque pointer/bytes
                        align: *align,
                        offset: *offset,
                    })
                } else {
                    Ok(instruction.clone())
                }
            }
            
            Instruction::Call { func_ref, args } => {
                // Transform function calls that might need type information
                let transformed_args: Result<Vec<_>, _> = args.iter()
                    .map(|arg| self.transform_operand(arg, item_ptr_local, desc_ptr_local, temp_locals))
                    .collect();
                
                Ok(Instruction::Call {
                    func_ref: *func_ref,
                    args: transformed_args?,
                })
            }
            
            Instruction::Return { value } => {
                // Transform return value
                if let Some(return_value) = value {
                    let transformed_value = self.transform_operand(
                        return_value, item_ptr_local, desc_ptr_local, temp_locals
                    )?;
                    Ok(Instruction::Return { value: Some(transformed_value) })
                } else {
                    Ok(Instruction::Return { value: None })
                }
            }
            
            // For drop operations, we need to use the descriptor
            Instruction::DropObject { object } => {
                self.transform_drop_operation(object, item_ptr_local, desc_ptr_local, temp_locals)
            }
            
            _ => Ok(instruction.clone()),
        }
    }

    /// Transforms drop operation using type descriptor
    fn transform_drop_operation(
        &mut self,
        object: &Operand,
        item_ptr_local: u32,
        desc_ptr_local: u32,
        temp_locals: &HashMap<String, u32>,
    ) -> Result<Instruction, ThinningError> {
        // Transform object operand to get pointer
        let object_ptr = self.transform_operand(
            object, item_ptr_local, desc_ptr_local, temp_locals
        )?;
        
        let function_ptr_local = temp_locals.get("function_ptr")
            .ok_or(ThinningError::MissingLocal("function_ptr".to_string()))?;
        
        // Load drop function pointer from descriptor
        // descriptor layout: [type_id(4), size(4), align(4), drop_glue(4), clone_fn(4), flags(4)]
        let load_drop_ptr = Instruction::MemoryLoad {
            address: Operand::Local(*desc_ptr_local),
            ty: Type::I32,
            align: Some(4),
            offset: 12, // Offset to drop_glue field
        };
        
        // Store drop function pointer
        let store_drop_ptr = Instruction::LocalSet {
            index: *function_ptr_local,
            value: Operand::Local(self.next_local_index), // Result of load
        };
        
        // Indirect call to drop function
        let call_drop = Instruction::CallIndirect {
            table_index: Operand::Local(*function_ptr_local),
            function_index: object_ptr,
            args: vec![object_ptr],
            signature: Signature {
                params: vec![Type::I32], // *mut u8
                returns: None, // void
            },
        };
        
        // Note: In a real implementation, this would generate a sequence of instructions
        // For now, we return a placeholder that represents the sequence
        Ok(Instruction::DropObject { object: object_ptr })
    }

    /// Transforms terminator for thinned context
    fn transform_terminator(
        &mut self,
        terminator: &Terminator,
        item_ptr_local: u32,
        desc_ptr_local: u32,
        temp_locals: &HashMap<String, u32>,
    ) -> Result<Terminator, ThinningError> {
        match terminator {
            Terminator::Return { value } => {
                if let Some(return_value) = value {
                    let transformed_value = self.transform_operand(
                        return_value, item_ptr_local, desc_ptr_local, temp_locals
                    )?;
                    Ok(Terminator::Return { value: Some(transformed_value) })
                } else {
                    Ok(Terminator::Return { value: None })
                }
            }
            
            Terminator::Branch { condition, then_block, else_block } => {
                let transformed_condition = self.transform_operand(
                    condition, item_ptr_local, desc_ptr_local, temp_locals
                )?;
                Ok(Terminator::Branch {
                    condition: transformed_condition,
                    then_block: *then_block,
                    else_block: *else_block,
                })
            }
            
            Terminator::Switch { value, targets, default_target } => {
                let transformed_value = self.transform_operand(
                    value, item_ptr_local, desc_ptr_local, temp_locals
                )?;
                Ok(Terminator::Switch {
                    value: transformed_value,
                    targets: targets.clone(),
                    default_target: *default_target,
                })
            }
            
            Terminator::Jump { target } => Ok(Terminator::Jump { target: *target }),
            Terminator::Unreachable => Ok(Terminator::Unreachable),
            Terminator::Panic { message } => {
                if let Some(msg) = message {
                    let transformed_msg = self.transform_operand(
                        msg, item_ptr_local, desc_ptr_local, temp_locals
                    )?;
                    Ok(Terminator::Panic { message: Some(transformed_msg) })
                } else {
                    Ok(Terminator::Panic { message: None })
                }
            }
        }
    }

    /// Transforms an operand for thinned context
    fn transform_operand(
        &mut self,
        operand: &Operand,
        item_ptr_local: u32,
        desc_ptr_local: u32,
        temp_locals: &HashMap<String, u32>,
    ) -> Result<Operand, ThinningError> {
        match operand {
            Operand::Local(index) => {
                if self.is_generic_parameter(*index) {
                    Ok(Operand::Local(item_ptr_local))
                } else {
                    Ok(operand.clone())
                }
            }
            Operand::Constant(_) => Ok(operand.clone()),
            Operand::Global(_) => Ok(operand.clone()),
            Operand::FunctionRef(_) => Ok(operand.clone()),
            Operand::ExternRef(_) => Ok(operand.clone()),
            Operand::FuncRef(_) => Ok(operand.clone()),
            Operand::MemoryAddress(addr) => {
                let transformed_addr = self.transform_operand(
                    addr, item_ptr_local, desc_ptr_local, temp_locals
                )?;
                Ok(Operand::MemoryAddress(Box::new(transformed_addr)))
            }
            Operand::StackValue(_) => Ok(operand.clone()),
        }
    }

    /// Creates shim functions for each concrete type
    fn create_shim_functions(
        &mut self,
        generic_function: &WasmIR,
        concrete_types: &[Type],
        type_descriptors: &[WasmTypeDescriptor],
    ) -> Result<Vec<WasmIR>, ThinningError> {
        let mut shim_functions = Vec::new();
        
        for (i, (concrete_type, descriptor)) in concrete_types.iter().zip(type_descriptors.iter()).enumerate() {
            let shim_function = self.create_single_shim(
                generic_function,
                concrete_type,
                descriptor,
                i as u32,
            )?;
            shim_functions.push(shim_function);
        }
        
        Ok(shim_functions)
    }

    /// Creates a single shim function
    fn create_single_shim(
        &mut self,
        generic_function: &WasmIR,
        concrete_type: &Type,
        descriptor: &WasmTypeDescriptor,
        type_index: u32,
    ) -> Result<WasmIR, ThinningError> {
        let type_name = self.type_to_string(concrete_type);
        let shim_name = format!("{}_{}_shim", generic_function.name, type_name);
        
        // Shim signature matches original generic function
        let mut shim_function = WasmIR::new(shim_name, generic_function.signature.clone());
        
        // Add locals
        let item_local = shim_function.add_local(concrete_type.clone());
        let item_ptr_local = shim_function.add_local(Type::I32);
        let desc_ptr_local = shim_function.add_local(Type::I32);
        
        // Create instructions
        let mut instructions = Vec::new();
        
        // 1. Get address of the concrete item
        let get_addr = Instruction::MemoryLoad {
            address: Operand::Local(item_local),
            ty: Type::I32,
            align: Some(4),
            offset: 0,
        };
        instructions.push(get_addr);
        let set_addr = Instruction::LocalSet {
            index: item_ptr_local,
            value: Operand::Local(shim_function.locals.len() - 1), // Result of load
        };
        instructions.push(set_addr);
        
        // 2. Load descriptor pointer
        let load_desc = Instruction::MemoryLoad {
            address: Operand::Global(descriptor.type_id),
            ty: Type::I32,
            align: Some(4),
            offset: 0,
        };
        instructions.push(load_desc);
        let set_desc = Instruction::LocalSet {
            index: desc_ptr_local,
            value: Operand::Local(shim_function.locals.len() - 1), // Result of load
        };
        instructions.push(set_desc);
        
        // 3. Call thinned function
        let call_thinned = Instruction::Call {
            func_ref: self.next_function_index,
            args: vec![
                Operand::Local(item_ptr_local),
                Operand::Local(desc_ptr_local),
            ],
        };
        instructions.push(call_thinned);
        
        // 4. Return result
        let return_instr = Instruction::Return { 
            value: Some(Operand::Local(shim_function.locals.len() - 1)) // Result of call
        };
        instructions.push(return_instr);
        
        shim_function.add_basic_block(instructions, Terminator::Return { 
            value: Some(Operand::Local(shim_function.locals.len() - 1))
        });
        
        Ok(shim_function)
    }

    /// Helper methods for type analysis
    fn type_to_string(&self, ty: &Type) -> String {
        match ty {
            Type::I32 => "i32".to_string(),
            Type::I64 => "i64".to_string(),
            Type::F32 => "f32".to_string(),
            Type::F64 => "f64".to_string(),
            Type::ExternRef(name) => format!("externref_{}", name),
            Type::Array { element_type, size } => {
                format!("[{};{}]", self.type_to_string(element_type), 
                    size.map_or("".to_string(), |s| s.to_string()))
            }
            Type::Struct { fields } => {
                format!("Struct({})", fields.iter()
                    .map(|f| self.type_to_string(f))
                    .collect::<Vec<_>>()
                    .join(", "))
            }
            Type::Pointer(inner_type) => {
                format!("*{}", self.type_to_string(inner_type))
            }
            Type::Linear { inner_type } => {
                format!("Linear<{}>", self.type_to_string(inner_type))
            }
            Type::Capability { inner_type, capability } => {
                format!("Capability<{}:{:?}>", self.type_to_string(inner_type), capability)
            }
            Type::Void => "void".to_string(),
        }
    }

    fn calculate_type_size_align(&self, ty: &Type) -> Result<(u32, u32), ThinningError> {
        match ty {
            Type::I32 | Type::ExternRef(_) => Ok((4, 4)),
            Type::I64 | Type::F64 => Ok((8, 8)),
            Type::F32 => Ok((4, 4)),
            Type::Pointer(_) => Ok((4, 4)), // 32-bit pointers
            Type::Array { element_type, size } => {
                let (elem_size, elem_align) = self.calculate_type_size_align(element_type)?;
                let array_size = size.unwrap_or(1) * elem_size;
                Ok((array_size, elem_align))
            }
            Type::Struct { fields } => {
                let mut total_size = 0;
                let mut max_align = 1;
                
                for field in fields {
                    let (field_size, field_align) = self.calculate_type_size_align(field)?;
                    total_size = (total_size + field_align - 1) & !(field_align - 1);
                    total_size += field_size;
                    max_align = max_align.max(field_align);
                }
                
                total_size = (total_size + max_align - 1) & !(max_align - 1);
                Ok((total_size, max_align))
            }
            Type::Linear { inner_type } => self.calculate_type_size_align(inner_type),
            Type::Capability { inner_type, .. } => self.calculate_type_size_align(inner_type),
            Type::Void => Ok((0, 1)),
        }
    }

    fn is_type_copy(&self, ty: &Type) -> bool {
        matches!(ty, Type::I32 | Type::I64 | Type::F32 | Type::F64 | Type::ExternRef(_))
    }

    fn is_type_pod(&self, ty: &Type) -> bool {
        self.is_type_copy(ty) && !self.type_needs_drop(ty)
    }

    fn type_needs_drop(&self, ty: &Type) -> bool {
        !self.is_type_copy(ty) && !matches!(ty, Type::Pointer(_) | Type::Void)
    }

    fn is_generic_type(&self, ty: &Type) -> bool {
        matches!(ty, Type::Ref(name) if name.starts_with("T"))
    }

    fn is_generic_parameter(&self, index: u32) -> bool {
        // Simplified check - in practice, this would use MIR local info
        index < 2 // Assume first two locals are generic parameters
    }

    fn estimate_function_size(&self, function: &WasmIR) -> usize {
        function.basic_blocks.iter()
            .map(|bb| bb.instructions.len() + 1) // +1 for terminator
            .sum()
    }

    fn generate_drop_glue(&mut self, _ty: &Type) -> Result<u32, ThinningError> {
        let function_index = self.next_function_index;
        self.next_function_index += 1;
        Ok(function_index)
    }

    fn build_function_table(&self, type_descriptors: &[WasmTypeDescriptor]) -> HashMap<String, u32> {
        let mut function_table = HashMap::new();
        
        for descriptor in type_descriptors {
            if let Some(drop_glue) = descriptor.drop_glue {
                function_table.insert(format!("drop_{}", descriptor.name), drop_glue);
            }
            
            if let Some(clone_fn) = descriptor.clone_fn {
                function_table.insert(format!("clone_{}", descriptor.name), clone_fn);
            }
        }
        
        function_table
    }
}

/// Errors that can occur during thinning
#[derive(Debug, Clone)]
pub enum ThinningError {
    /// Type transformation failed
    TypeTransformationError(String),
    /// Invalid generic function
    InvalidGenericFunction(String),
    /// Missing local variable
    MissingLocal(String),
    /// Unsupported operation for thinning
    UnsupportedOperation(String),
}

impl std::fmt::Display for ThinningError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ThinningError::TypeTransformationError(msg) => {
                write!(f, "Type transformation error: {}", msg)
            }
            ThinningError::InvalidGenericFunction(msg) => {
                write!(f, "Invalid generic function: {}", msg)
            }
            ThinningError::MissingLocal(name) => {
                write!(f, "Missing local variable: {}", name)
            }
            ThinningError::UnsupportedOperation(msg) => {
                write!(f, "Unsupported operation: {}", msg)
            }
        }
    }
}

impl std::error::Error for ThinningError {}

#[cfg(test)]
mod tests {
    use super::*;
    use rustc_target::spec::Target;

    #[test]
    fn test_thinning_pass_creation() {
        let target = rustc_target::spec::Target {
            arch: "wasm32".to_string(),
            ..Default::default()
        };
        
        let pass = ThinningPass::new(target);
        assert_eq!(pass.next_local_index, 0);
        assert_eq!(pass.next_function_index, 0);
    }

    #[test]
    fn test_type_descriptor_creation() {
        let target = rustc_target::spec::Target {
            arch: "wasm32".to_string(),
            ..Default::default()
        };
        
        let mut pass = ThinningPass::new(target);
        
        let concrete_types = vec![Type::I32, Type::F64];
        let descriptors = pass.create_type_descriptors(&concrete_types).unwrap();
        
        assert_eq!(descriptors.len(), 2);
        assert_eq!(descriptors[0].type_id, 1);
        assert_eq!(descriptors[1].type_id, 2);
        assert!(descriptors[0].is_copy);
        assert!(descriptors[1].is_copy);
    }

    #[test]
    fn test_thinned_signature_creation() {
        let target = rustc_target::spec::Target {
            arch: "wasm32".to_string(),
            ..Default::default()
        };
        
        let pass = ThinningPass::new(target);
        
        let generic_function = WasmIR::new(
            "test_function<T>".to_string(),
            Signature {
                params: vec![Type::Ref("T".to_string())],
                returns: Some(Type::Ref("T".to_string())),
            },
        );
        
        let thinned_signature = pass.create_thinned_signature(&generic_function, &[]);
        
        assert_eq!(thinned_signature.params.len(), 2);
        assert_eq!(thinned_signature.params[0], Type::I32); // item_ptr
        assert_eq!(thinned_signature.params[1], Type::I32); // desc_ptr
        assert_eq!(thinned_signature.returns, Some(Type::Ref("T".to_string())));
    }

    #[test]
    fn test_simple_instruction_transformation() {
        let target = rustc_target::spec::Target {
            arch: "wasm32".to_string(),
            ..Default::default()
        };
        
        let mut pass = ThinningPass::new(target);
        let temp_locals = HashMap::new();
        temp_locals.insert("generic_local".to_string(), 5);
        
        // Test generic parameter access transformation
        let generic_get = Instruction::LocalGet { index: 0 };
        let transformed = pass.transform_instruction(
            &generic_get, 1, 2, &temp_locals
        ).unwrap();
        
        match transformed {
            Instruction::LocalGet { index } => assert_eq!(index, 1), // Should become item_ptr_local
            _ => panic!("Expected LocalGet instruction"),
        }
    }

    #[test]
    fn test_shim_function_creation() {
        let target = rustc_target::spec::Target {
            arch: "wasm32".to_string(),
            ..Default::default()
        };
        
        let mut pass = ThinningPass::new(target);
        
        let generic_function = WasmIR::new(
            "process<T>".to_string(),
            Signature {
                params: vec![Type::Ref("T".to_string())],
                returns: Some(Type::Void),
            },
        );
        
        let concrete_type = Type::I32;
        let descriptor = WasmTypeDescriptor::new(1, 4, 4, "i32".to_string())
            .with_copy(true)
            .with_pod(true);
        
        let shim = pass.create_single_shim(&generic_function, &concrete_type, &descriptor, 0).unwrap();
        
        assert!(shim.name.starts_with("process_i32_shim"));
        assert_eq!(shim.signature.params.len(), 1);
        assert_eq!(shim.signature.params[0], Type::I32);
    }

    #[test]
    fn test_size_estimation() {
        let target = rustc_target::spec::Target {
            arch: "wasm32".to_string(),
            ..Default::default()
        };
        
        let pass = ThinningPass::new(target);
        
        let mut function = WasmIR::new(
            "test".to_string(),
            Signature {
                params: vec![Type::I32],
                returns: Some(Type::I32),
            },
        );
        
        // Add some basic blocks
        function.add_basic_block(
            vec![
                Instruction::LocalGet { index: 0 },
                Instruction::Return { value: Some(Operand::Local(1)) },
            ],
            Terminator::Return { value: Some(Operand::Local(1)) },
        );
        
        let size = pass.estimate_function_size(&function);
        assert!(size > 0);
        assert!(size <= 10); // Should be reasonable
    }
}