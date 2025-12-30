//! WasmRust Intermediate Representation (WasmIR)
//! 
//! This module provides the stable intermediate representation between
//! rustc frontend and WasmRust backends. WasmIR serves as
//! the boundary layer that encodes WASM-specific optimizations,
//! ownership annotations, and capability hints.

use alloc::collections::BTreeMap as HashMap;
use alloc::vec::{self, Vec};
use alloc::string::{String, ToString};
use alloc::boxed::Box;
use alloc::format;
use core::fmt;

/// WasmIR - Stable Intermediate Representation
/// 
/// WasmIR is designed to be a stable boundary between frontend and backends,
/// encoding WASM-specific optimizations and ownership invariants that may not
/// be present in standard Rust MIR.
#[derive(Debug, Clone)]
pub struct WasmIR {
    /// Function name
    pub name: String,
    /// Function signature
    pub signature: Signature,
    /// Basic blocks comprising the function body
    pub basic_blocks: Vec<BasicBlock>,
    /// Local variables in this function
    pub locals: Vec<Type>,
    /// Capability annotations for optimization
    pub capabilities: Vec<Capability>,
    /// Ownership annotations for linear types
    pub ownership_annotations: Vec<OwnershipAnnotation>,
}

/// Function signature in WasmIR
#[derive(Debug, Clone, PartialEq)]
pub struct Signature {
    /// Parameter types
    pub params: Vec<Type>,
    /// Return type (None for void functions)
    pub returns: Option<Type>,
}

/// Basic block in WasmIR control flow
#[derive(Debug, Clone)]
pub struct BasicBlock {
    /// Unique identifier for this block
    pub id: BlockId,
    /// Instructions in this basic block
    pub instructions: Vec<Instruction>,
    /// Terminator instruction
    pub terminator: Terminator,
}

/// Instruction identifier for basic blocks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub usize);

/// WasmIR instruction set
/// 
/// This instruction set is designed to map efficiently to WebAssembly
/// while preserving high-level semantic information for optimizations.
#[derive(Debug, Clone)]
pub enum Instruction {
    /// Get a local variable
    LocalGet { index: u32 },
    
    /// Set a local variable
    LocalSet { index: u32, value: Operand },
    
    /// Binary operation
    BinaryOp {
        op: BinaryOp,
        left: Operand,
        right: Operand,
    },
    
    /// Unary operation
    UnaryOp {
        op: UnaryOp,
        value: Operand,
    },
    
    /// Function call
    Call {
        func_ref: u32,
        args: Vec<Operand>,
    },
    
    /// Return from function
    Return { value: Option<Operand> },
    
    /// Conditional branch
    Branch {
        condition: Operand,
        then_block: BlockId,
        else_block: BlockId,
    },
    
    /// Unconditional branch
    Jump { target: BlockId },
    
    /// Switch statement
    Switch {
        value: Operand,
        targets: Vec<BlockId>,
        default_target: BlockId,
    },
    
    /// Load from memory
    MemoryLoad {
        address: Operand,
        ty: Type,
        align: Option<u32>,
        offset: u32,
    },
    
    /// Store to memory
    MemoryStore {
        address: Operand,
        value: Operand,
        ty: Type,
        align: Option<u32>,
        offset: u32,
    },
    
    /// Allocate memory on the heap
    MemoryAlloc { size: Operand, align: Option<u32> },
    
    /// Deallocate memory
    MemoryFree { address: Operand },
    
    /// Create a new object reference
    NewObject { type_id: u32, args: Vec<Operand> },
    
    /// Drop an object reference
    DropObject { object: Operand },
    
    /// Load from ExternRef
    ExternRefLoad { 
        externref: Operand, 
        field: String, 
        field_type: Type,
    },
    
    /// Store to ExternRef
    ExternRefStore { 
        externref: Operand, 
        field: String, 
        value: Operand,
        field_type: Type,
    },
    
    /// Call JavaScript method
    JSMethodCall {
        object: Operand,
        method: String,
        args: Vec<Operand>,
        return_type: Option<Type>,
    },
    
    /// Create function reference
    MakeFuncRef { 
        function_index: u32,
        signature: Signature,
    },
    
    /// Call function through reference
    FuncRefCall {
        funcref: Operand,
        args: Vec<Operand>,
        signature: Signature,
    },
    
    /// Create new ExternRef from value
    ExternRefNew { 
        value: Operand,
        target_type: Type,
    },
    
    /// Convert ExternRef to internal value
    ExternRefCast { 
        externref: Operand,
        target_type: Type,
    },
    
    /// Check if ExternRef is null
    ExternRefIsNull { externref: Operand },
    
    /// Compare two ExternRefs for equality
    ExternRefEq { 
        left: Operand,
        right: Operand,
    },
    
    /// Create new FuncRef from function index
    FuncRefNew { function_index: u32 },
    
    /// Check if FuncRef is null
    FuncRefIsNull { funcref: Operand },
    
    /// Compare two FuncRefs for equality
    FuncRefEq { 
        left: Operand,
        right: Operand,
    },
    
    /// Dynamic function call through function table
    CallIndirect {
        table_index: Operand,
        function_index: Operand,
        args: Vec<Operand>,
        signature: Signature,
    },
    
    /// Atomic operation
    AtomicOp {
        op: AtomicOp,
        address: Operand,
        value: Operand,
        order: MemoryOrder,
    },
    
    /// Compare and swap
    CompareExchange {
        address: Operand,
        expected: Operand,
        new_value: Operand,
        order: MemoryOrder,
    },
    
    /// Linear type operation
    LinearOp {
        op: LinearOp,
        value: Operand,
    },
    
    /// Capability check
    CapabilityCheck {
        capability: Capability,
    },
    
    /// NOP instruction
    Nop,
}

/// Binary operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add, Sub, Mul, Div, Mod,
    And, Or, Xor,
    Shl, Shr, Sar,
    Eq, Ne, Lt, Le, Gt, Ge,
}

/// Unary operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg, Not, Clz, Ctz, Popcnt,
}

/// Atomic operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicOp {
    Add, Sub, And, Or, Xor, Exchange,
}

/// Linear type operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearOp {
    Consume, Move, Clone, Drop,
}

/// Memory ordering for atomic operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOrder {
    Relaxed, Acquire, Release, AcqRel, SeqCst,
}

/// Terminator instruction for basic blocks
#[derive(Debug, Clone)]
pub enum Terminator {
    /// Return from function
    Return { value: Option<Operand> },
    
    /// Conditional branch
    Branch { condition: Operand, then_block: BlockId, else_block: BlockId },
    
    /// Switch statement
    Switch {
        value: Operand,
        targets: Vec<(Operand, BlockId)>,
        default_target: BlockId,
    },
    
    /// Unconditional branch
    Jump { target: BlockId },
    
    /// Unreachable (indicates program cannot reach this point)
    Unreachable,
    
    /// Panic/abort
    Panic { message: Option<Operand> },
}

/// Operand in WasmIR instructions
#[derive(Debug, Clone)]
pub enum Operand {
    /// Local variable
    Local(u32),
    
    /// Constant value
    Constant(Constant),
    
    /// Global variable
    Global(u32),
    
    /// Function reference
    FunctionRef(u32),
    
    /// ExternRef (JavaScript object)
    ExternRef(u32),
    
    /// Function reference
    FuncRef(u32),
    
    /// Memory address
    MemoryAddress(Box<Operand>),
    
    /// Stack value (temporary)
    StackValue(u32),
}

/// Constant values
#[derive(Debug, Clone)]
pub enum Constant {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Null,
    Boolean(bool),
    String(String),
}

/// Types in WasmIR
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    /// 32-bit integer
    I32,
    
    /// 64-bit integer
    I64,
    
    /// 32-bit float
    F32,
    
    /// 64-bit float
    F64,
    
    /// External reference (JavaScript object)
    ExternRef(String),
    
    /// Function reference
    FuncRef,
    
    /// Array type
    Array { element_type: Box<Type>, size: Option<u32> },
    
    /// Struct type
    Struct { fields: Vec<Type> },
    
    /// Pointer type
    Pointer(Box<Type>),
    
    /// Linear type (use-once semantics)
    Linear { inner_type: Box<Type> },
    
    /// Capability-annotated type
    Capability { inner_type: Box<Type>, capability: Capability },
    
    /// Void type
    Void,
}

/// Capability annotations for optimization
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Capability {
    /// JavaScript interop capability
    JsInterop,
    
    /// Threading capability
    Threading,
    
    /// Atomic memory capability
    AtomicMemory,
    
    /// Component model capability
    ComponentModel,
    
    /// Memory region access
    MemoryRegion(String),
    
    /// Custom capability
    Custom(String),
}

/// Ownership annotations for linear types
#[derive(Debug, Clone)]
pub struct OwnershipAnnotation {
    /// Variable being annotated
    pub variable: u32,
    /// Ownership state
    pub state: OwnershipState,
    /// Source location for error reporting
    pub source_location: SourceLocation,
}

/// Ownership states for linear types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OwnershipState {
    /// Value is owned (can be used or moved)
    Owned,
    
    /// Value has been moved (cannot be used)
    Moved,
    
    /// Value is borrowed (temporary ownership)
    Borrowed,
    
    /// Value is consumed (used exactly once)
    Consumed,
}

/// Source location for error reporting
#[derive(Debug, Clone)]
pub struct SourceLocation {
    /// File path
    pub file: String,
    
    /// Line number
    pub line: u32,
    
    /// Column number
    pub column: u32,
}

impl WasmIR {
    /// Creates a new WasmIR function
    pub fn new(name: String, signature: Signature) -> Self {
        Self {
            name,
            signature,
            basic_blocks: Vec::new(),
            locals: Vec::new(),
            capabilities: Vec::new(),
            ownership_annotations: Vec::new(),
        }
    }

    /// Adds a basic block to the function
    pub fn add_basic_block(&mut self, instructions: Vec<Instruction>, terminator: Terminator) -> BlockId {
        let block_id = BlockId(self.basic_blocks.len());
        let block = BasicBlock {
            id: block_id,
            instructions,
            terminator,
        };
        self.basic_blocks.push(block);
        block_id
    }

    /// Adds a local variable to the function
    pub fn add_local(&mut self, ty: Type) -> u32 {
        let index = self.locals.len() as u32;
        self.locals.push(ty);
        index
    }

    /// Adds a capability annotation to the function
    pub fn add_capability(&mut self, capability: Capability) {
        self.capabilities.push(capability);
    }

    /// Adds an ownership annotation
    pub fn add_ownership_annotation(&mut self, annotation: OwnershipAnnotation) {
        self.ownership_annotations.push(annotation);
    }

    /// Validates the WasmIR function
    pub fn validate(&self) -> Result<(), ValidationError> {
        // Check that all block IDs are valid
        for block in &self.basic_blocks {
            match &block.terminator {
                Terminator::Branch { then_block, else_block, .. } => {
                    if !self.is_valid_block_id(*then_block) {
                        return Err(ValidationError::InvalidBlockId("then_block"));
                    }
                    if !self.is_valid_block_id(*else_block) {
                        return Err(ValidationError::InvalidBlockId("else_block"));
                    }
                }
                Terminator::Switch { default_target, targets, .. } => {
                    if !self.is_valid_block_id(*default_target) {
                        return Err(ValidationError::InvalidBlockId("default_target"));
                    }
                    for (_, target) in targets {
                        if !self.is_valid_block_id(*target) {
                            return Err(ValidationError::InvalidBlockId("switch_target"));
                        }
                    }
                }
                Terminator::Jump { target } => {
                    if !self.is_valid_block_id(*target) {
                        return Err(ValidationError::InvalidBlockId("jump_target"));
                    }
                }
                _ => {}
            }
        }

        // Check that all operand indices are valid
        for instruction in self.basic_blocks.iter().flat_map(|bb| &bb.instructions) {
            self.validate_instruction_operands(instruction)?;
        }

        Ok(())
    }

    /// Validates operand indices in an instruction
    fn validate_instruction_operands(&self, instruction: &Instruction) -> Result<(), ValidationError> {
        match instruction {
            Instruction::LocalGet { index } => {
                if *index >= self.locals.len() as u32 {
                    return Err(ValidationError::InvalidLocalIndex(*index));
                }
            }
            Instruction::LocalSet { index, .. } => {
                if *index >= self.locals.len() as u32 {
                    return Err(ValidationError::InvalidLocalIndex(*index));
                }
            }
            Instruction::Call { args, .. } => {
                for (i, arg) in args.iter().enumerate() {
                    self.validate_operand(arg, "arg")?;
                }
            }
            Instruction::MemoryLoad { address, .. } => {
                self.validate_operand(address, "address")?;
            }
            Instruction::MemoryStore { address, value, .. } => {
                self.validate_operand(address, "address")?;
                self.validate_operand(value, "value")?;
            }
            Instruction::Branch { condition, .. } => {
                self.validate_operand(condition, "condition")?;
            }
            Instruction::JSMethodCall { args, .. } => {
                for (i, arg) in args.iter().enumerate() {
                    self.validate_operand(arg, "js_arg")?;
                }
            }
            _ => {
                // Additional validation for other instruction types
            }
        }
        Ok(())
    }

    /// Validates an operand
    fn validate_operand(&self, operand: &Operand, context: &str) -> Result<(), ValidationError> {
        match operand {
            Operand::Local(index) => {
                if *index >= self.locals.len() as u32 {
                    return Err(ValidationError::InvalidLocalIndex(*index));
                }
            }
            Operand::Constant(_) => {} // Constants are always valid
            Operand::Global(_) => {} // Globals are checked at link time
            Operand::FunctionRef(_) => {} // Function refs are checked at link time
            Operand::ExternRef(_) => {} // ExternRefs are checked at link time
            Operand::FuncRef(_) => {} // FuncRefs are checked at link time
            Operand::MemoryAddress(addr) => {
                self.validate_operand(addr, "address")?;
            }
            Operand::StackValue(_) => {} // Stack values are checked during compilation
        }
        Ok(())
    }

    /// Checks if a block ID is valid
    fn is_valid_block_id(&self, block_id: BlockId) -> bool {
        block_id.0 < self.basic_blocks.len()
    }

    /// Gets the entry block (first basic block)
    pub fn entry_block(&self) -> Option<&BasicBlock> {
        self.basic_blocks.first()
    }

    /// Calculates the size of the function in terms of instructions
    pub fn instruction_count(&self) -> usize {
        self.basic_blocks.iter().map(|bb| bb.instructions.len()).sum()
    }

    /// Gets all instructions in the function
    pub fn all_instructions(&self) -> impl Iterator<Item = &Instruction> {
        self.basic_blocks.iter().flat_map(|bb| bb.instructions.iter())
    }

    /// Finds all local variables that are used
    pub fn used_locals(&self) -> HashMap<u32, ()> {
        let mut used_locals = HashMap::new();
        
        for instruction in self.all_instructions() {
            match instruction {
                Instruction::LocalGet { index } => {
                    used_locals.insert(*index, ());
                }
                Instruction::LocalSet { index, .. } => {
                    used_locals.insert(*index, ());
                }
                Instruction::BinaryOp { left, right, .. } => {
                    self.collect_used_locals_from_operand(left, &mut used_locals);
                    self.collect_used_locals_from_operand(right, &mut used_locals);
                }
                Instruction::Call { args, .. } => {
                    for arg in args {
                        self.collect_used_locals_from_operand(arg, &mut used_locals);
                    }
                }
                Instruction::Branch { condition, .. } => {
                    self.collect_used_locals_from_operand(condition, &mut used_locals);
                }
                _ => {}
            }
        }
        
        used_locals
    }

    /// Collects local variables used in an operand
    fn collect_used_locals_from_operand(&self, operand: &Operand, used_locals: &mut HashMap<u32, ()>) {
        match operand {
            Operand::Local(index) => {
                used_locals.insert(*index, ());
            }
            Operand::MemoryAddress(addr) => {
                self.collect_used_locals_from_operand(addr, used_locals);
            }
            _ => {}
        }
    }
}

/// Validation errors for WasmIR
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationError {
    /// Invalid local variable index
    InvalidLocalIndex(u32),
    
    /// Invalid basic block ID
    InvalidBlockId(&'static str),
    
    /// Type mismatch error
    TypeMismatch { expected: Type, actual: Type },
    
    /// Control flow error
    ControlFlowError(&'static str),
    
    /// Capability violation
    CapabilityViolation(Capability),
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationError::InvalidLocalIndex(idx) => write!(f, "Invalid local index: {}", idx),
            ValidationError::InvalidBlockId(desc) => write!(f, "Invalid block ID: {}", desc),
            ValidationError::TypeMismatch { expected, actual } => {
                write!(f, "Type mismatch: expected {:?}, got {:?}", expected, actual)
            }
            ValidationError::ControlFlowError(msg) => write!(f, "Control flow error: {}", msg),
            ValidationError::CapabilityViolation(cap) => write!(f, "Capability violation: {:?}", cap),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_function_creation() {
        let signature = Signature {
            params: vec![Type::I32, Type::I32],
            returns: Some(Type::I32),
        };
        
        let func = WasmIR::new("test".to_string(), signature);
        assert_eq!(func.name, "test");
        assert_eq!(func.locals.len(), 0);
        assert_eq!(func.basic_blocks.len(), 0);
    }

    #[test]
    fn test_basic_block_addition() {
        let func = WasmIR::new("test".to_string(), Signature {
            params: vec![Type::I32],
            returns: Some(Type::I32),
        });
        
        let instructions = vec![
            Instruction::BinaryOp {
                op: BinaryOp::Add,
                left: Operand::Local(0),
                right: Operand::Local(0),
            },
            Instruction::Return {
                value: Some(Operand::Local(1)), // Would be result of add
            },
        ];
        
        let terminator = Terminator::Return {
            value: Some(Operand::Local(1)),
        };
        
        let block_id = func.add_basic_block(instructions, terminator);
        assert_eq!(block_id.0, 0);
        assert_eq!(func.basic_blocks.len(), 1);
    }

    #[test]
    fn test_local_variable_addition() {
        let func = WasmIR::new("test".to_string(), Signature {
            params: vec![Type::I32],
            returns: Some(Type::I32),
        });
        
        let index = func.add_local(Type::I32);
        assert_eq!(index, 0);
        
        let second_index = func.add_local(Type::F32);
        assert_eq!(second_index, 1);
        
        assert_eq!(func.locals.len(), 2);
        assert_eq!(func.locals[0], Type::I32);
        assert_eq!(func.locals[1], Type::F32);
    }

    #[test]
    fn test_capability_annotation() {
        let func = WasmIR::new("test".to_string(), Signature {
            params: vec![Type::I32],
            returns: Some(Type::I32),
        });
        
        func.add_capability(Capability::JsInterop);
        func.add_capability(Capability::Threading);
        
        assert_eq!(func.capabilities.len(), 2);
        assert!(func.capabilities.contains(&Capability::JsInterop));
        assert!(func.capabilities.contains(&Capability::Threading));
    }

    #[test]
    fn test_validation_valid_function() {
        let func = WasmIR::new("test".to_string(), Signature {
            params: vec![Type::I32, Type::I32],
            returns: Some(Type::I32),
        });
        
        let local_index = func.add_local(Type::I32);
        
        let instructions = vec![
            Instruction::LocalGet { index: local_index },
        ];
        
        let terminator = Terminator::Return { value: None };
        func.add_basic_block(instructions, terminator);
        
        let result = func.validate();
        assert!(result.is_ok());
    }

    #[test]
    fn test_validation_invalid_local_index() {
        let func = WasmIR::new("test".to_string(), Signature {
            params: vec![Type::I32],
            returns: Some(Type::I32),
        });
        
        let instructions = vec![
            Instruction::LocalGet { index: 999 }, // Invalid index
        ];
        
        let terminator = Terminator::Return { value: None };
        func.add_basic_block(instructions, terminator);
        
        let result = func.validate();
        assert!(result.is_err());
        
        match result.unwrap_err() {
            ValidationError::InvalidLocalIndex(idx) => assert_eq!(idx, 999),
            _ => panic!("Unexpected error type"),
        }
    }

    #[test]
    fn test_instruction_count() {
        let func = WasmIR::new("test".to_string(), Signature {
            params: vec![Type::I32],
            returns: Some(Type::I32),
        });
        
        let block1_instructions = vec![
            Instruction::LocalGet { index: 0 },
            Instruction::LocalGet { index: 0 },
        ];
        
        let block2_instructions = vec![
            Instruction::BinaryOp {
                op: BinaryOp::Add,
                left: Operand::Local(0),
                right: Operand::Local(0),
            },
        ];
        
        func.add_basic_block(block1_instructions, Terminator::Jump { target: BlockId(1) });
        func.add_basic_block(block2_instructions, Terminator::Return { value: None });
        
        assert_eq!(func.instruction_count(), 5); // 2 + 2 + 1 terminator
    }

    #[test]
    fn test_used_locals() {
        let func = WasmIR::new("test".to_string(), Signature {
            params: vec![Type::I32],
            returns: Some(Type::I32),
        });
        
        let local1 = func.add_local(Type::I32);
        let local2 = func.add_local(Type::I32);
        let local3 = func.add_local(Type::I32);
        
        let instructions = vec![
            Instruction::LocalGet { index: local1 },
            Instruction::BinaryOp {
                op: BinaryOp::Add,
                left: Operand::Local(local1),
                right: Operand::Constant(Constant::I32(42)),
            },
            Instruction::LocalSet { index: local2, value: Operand::Local(1) }, // Result of add
        ];
        
        func.add_basic_block(instructions, Terminator::Return { value: None });
        
        let used_locals = func.used_locals();
        assert!(used_locals.contains_key(&local1));
        assert!(used_locals.contains_key(&local2));
        assert!(!used_locals.contains_key(&local3)); // local3 is never used
    }
}
