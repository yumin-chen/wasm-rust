//! WasmRust Cranelift Backend
//! 
//! This module provides a Cranelift-based codegen backend for WasmRust,
//! optimized for fast development compilation. It integrates with rustc's
//! codegen interface while adding WasmRust-specific optimizations.

use cranelift_codegen::*;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_codegen::ir::{Function, InstBuilder, Signature, AbiParam, types, Type};
use cranelift_codegen::isa::TargetIsa;
use cranelift_codegen::settings::{Flags, Configurable};
use cranelift_codegen::Context as CodegenContext;
use cranelift_codegen::ir::{condcodes::IntCC, Block};
use cranelift_codegen::entity::EntityRef;
use cranelift_control::ControlPlane;
use std::collections::HashMap;
use std::sync::Arc;

use wasm::wasmir::{WasmIR, Instruction, Terminator, BasicBlock, BlockId, Type as WasmIRType, Signature as WasmIRSignature, Operand, BinaryOp, UnaryOp, Constant, AtomicOp, LinearOp, MemoryOrder, Capability};

pub mod mir_lowering;

/// Cranelift codegen backend for WasmRust
pub struct WasmRustCraneliftBackend {
    /// Target ISA for code generation
    isa: Arc<dyn TargetIsa>,
    /// WasmRust-specific optimization flags
    optimization_flags: WasmRustOptimizationFlags,
    /// Function compilation cache
    function_cache: HashMap<u64, Vec<u8>>,
    /// Compilation statistics
    stats: CompilationStats,
}

/// WasmRust-specific optimization flags
#[derive(Debug, Clone)]
pub struct WasmRustOptimizationFlags {
    /// Enable thin monomorphization for code deduplication
    pub thin_monomorphization: bool,
    /// Enable streaming layout optimization
    pub streaming_layout: bool,
    /// Enable WASM-specific optimizations
    pub wasm_optimizations: bool,
    /// Enable zero-cost abstractions
    pub zero_cost_abstractions: bool,
}

impl Default for WasmRustOptimizationFlags {
    fn default() -> Self {
        Self {
            thin_monomorphization: true,
            streaming_layout: true,
            wasm_optimizations: true,
            zero_cost_abstractions: true,
        }
    }
}

/// Compilation statistics for performance monitoring
#[derive(Debug, Default)]
pub struct CompilationStats {
    pub functions_compiled: usize,
    pub instructions_generated: usize,
    pub optimization_passes: usize,
    pub compilation_time_ms: u64,
}

impl WasmRustCraneliftBackend {
    /// Creates a new Cranelift backend for WasmRust
    pub fn new() -> Result<Self, CodegenError> {
        let isa = create_target_isa()?;
        let optimization_flags = WasmRustOptimizationFlags::default();
        
        Ok(Self {
            isa,
            optimization_flags,
            function_cache: HashMap::new(),
            stats: CompilationStats::default(),
        })
    }

    /// Compiles a WasmIR function to machine code
    pub fn compile_function(
        &mut self,
        wasmir_func: &WasmIR,
        function_name: &str,
    ) -> Result<Vec<u8>, CodegenError> {
        let start_time = std::time::Instant::now();

        // Convert WasmIR to Cranelift IR
        let func = self.convert_function_body(wasmir_func)?;
        
        // Apply WasmRust-specific optimizations
        let mut optimized_func = func;
        self.apply_optimizations(&mut optimized_func)?;
        
        // Get instruction count before moving the function
        let instruction_count = optimized_func.dfg.num_insts();
        
        // Compile to machine code
        let mut code_gen_context = CodegenContext::new();
        code_gen_context.func = optimized_func;
        let mut ctrl_plane = ControlPlane::default();
        let compiled = code_gen_context.compile(&*self.isa, &mut ctrl_plane)?;

        let code = compiled.code_buffer().to_vec();

        // Update statistics
        self.stats.functions_compiled += 1;
        self.stats.instructions_generated += instruction_count;
        self.stats.compilation_time_ms += start_time.elapsed().as_millis() as u64;

        // Cache compiled function
        let function_hash = self.hash_function(wasmir_func);
        self.function_cache.insert(function_hash, code.clone());

        Ok(code)
    }

    /// Gets compilation statistics
    pub fn get_stats(&self) -> &CompilationStats {
        &self.stats
    }

    /// Clears compilation statistics
    pub fn clear_stats(&mut self) {
        self.stats = CompilationStats::default();
    }

    /// Converts WasmIR signature to Cranelift signature
    fn convert_signature(&self, wasmir_sig: &WasmIRSignature) -> Result<Signature, CodegenError> {
        let mut signature = Signature::new(cranelift_codegen::isa::CallConv::SystemV);

        // Convert parameters
        for param in &wasmir_sig.params {
            let cranelift_param = self.convert_type(param)?;
            signature.params.push(AbiParam::new(cranelift_param));
        }

        // Convert return type
        if let Some(ret_type) = &wasmir_sig.returns {
            let cranelift_ret = self.convert_type(ret_type)?;
            signature.returns.push(AbiParam::new(cranelift_ret));
        }

        Ok(signature)
    }

    /// Converts WasmIR function body to Cranelift IR
    fn convert_function_body(&self, wasmir_func: &WasmIR) -> Result<Function, CodegenError> {
        let signature = self.convert_signature(&wasmir_func.signature)?;
        let mut func = Function::with_name_signature(
            cranelift_codegen::ir::UserFuncName::user(0, 0),
            signature,
        );

        let mut builder_context = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut func, &mut builder_context);

        // Create blocks for each basic block
        let mut block_map = HashMap::new();
        for (i, _) in wasmir_func.basic_blocks.iter().enumerate() {
            let block = builder.create_block();
            block_map.insert(BlockId(i), block);
        }

        // Convert basic blocks
        for (bb_id, bb) in wasmir_func.basic_blocks.iter().enumerate() {
            let block = block_map[&BlockId(bb_id)];
            builder.switch_to_block(block);

            // Convert instructions in this basic block
            for instruction in &bb.instructions {
                self.convert_instruction(&mut builder, instruction)?;
            }

            // Add terminator for this block
            self.add_block_terminator(&mut builder, &bb.terminator, &block_map)?;
        }

        builder.finalize();
        Ok(func)
    }

    /// Converts a WasmIR instruction to Cranelift IR
    fn convert_instruction(
        &self,
        builder: &mut FunctionBuilder,
        instruction: &Instruction,
    ) -> Result<Option<cranelift_codegen::ir::Value>, CodegenError> {
        match instruction {
            Instruction::LocalGet { index } => {
                let var = Variable::from_u32(*index);
                let value = builder.use_var(var);
                Ok(Some(value))
            }
            Instruction::LocalSet { index, value } => {
                let var = Variable::from_u32(*index);
                let converted_value = self.convert_operand(builder, value)?;
                builder.def_var(var, converted_value);
                Ok(None)
            }
            Instruction::BinaryOp { op, left, right } => {
                let left_val = self.convert_operand(builder, left)?;
                let right_val = self.convert_operand(builder, right)?;
                let result = match op {
                    BinaryOp::Add => builder.ins().iadd(left_val, right_val),
                    BinaryOp::Sub => builder.ins().isub(left_val, right_val),
                    BinaryOp::Mul => builder.ins().imul(left_val, right_val),
                    BinaryOp::Div => builder.ins().sdiv(left_val, right_val),
                    BinaryOp::Mod => builder.ins().srem(left_val, right_val),
                    BinaryOp::And => builder.ins().band(left_val, right_val),
                    BinaryOp::Or => builder.ins().bor(left_val, right_val),
                    BinaryOp::Xor => builder.ins().bxor(left_val, right_val),
                    BinaryOp::Shl => builder.ins().ishl(left_val, right_val),
                    BinaryOp::Shr => builder.ins().sshr(left_val, right_val),
                    BinaryOp::Sar => builder.ins().sshr(left_val, right_val),
                    BinaryOp::Eq => builder.ins().icmp(IntCC::Equal, left_val, right_val),
                    BinaryOp::Ne => builder.ins().icmp(IntCC::NotEqual, left_val, right_val),
                    BinaryOp::Lt => builder.ins().icmp(IntCC::SignedLessThan, left_val, right_val),
                    BinaryOp::Le => builder.ins().icmp(IntCC::SignedLessThanOrEqual, left_val, right_val),
                    BinaryOp::Gt => builder.ins().icmp(IntCC::SignedGreaterThan, left_val, right_val),
                    BinaryOp::Ge => builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, left_val, right_val),
                };
                Ok(Some(result))
            }
            Instruction::UnaryOp { op, value } => {
                let value_val = self.convert_operand(builder, value)?;
                let result = match op {
                    UnaryOp::Neg => builder.ins().ineg(value_val),
                    UnaryOp::Not => builder.ins().bnot(value_val),
                    UnaryOp::Clz => builder.ins().clz(value_val),
                    UnaryOp::Ctz => builder.ins().ctz(value_val),
                    UnaryOp::Popcnt => builder.ins().popcnt(value_val),
                };
                Ok(Some(result))
            }
            Instruction::Return { value } => {
                if let Some(val) = value {
                    let converted_val = self.convert_operand(builder, val)?;
                    builder.ins().return_(&[converted_val]);
                } else {
                    builder.ins().return_(&[]);
                }
                Ok(None)
            }
            Instruction::Nop => Ok(None),
            _ => {
                // For now, return Ok(None) for unimplemented instructions
                // This allows the basic backend to compile
                Ok(None)
            }
        }
    }

    /// Converts a WasmIR operand to Cranelift value
    fn convert_operand(
        &self,
        builder: &mut FunctionBuilder,
        operand: &Operand,
    ) -> Result<cranelift_codegen::ir::Value, CodegenError> {
        match operand {
            Operand::Local(index) => {
                let var = Variable::from_u32(*index);
                Ok(builder.use_var(var))
            }
            Operand::Constant(value) => {
                let const_val = self.convert_constant(value)?;
                Ok(builder.ins().iconst(types::I32, const_val as i64))
            }
            Operand::Global(_global_index) => {
                // Global variables need special handling in WASM
                Err(CodegenError::Unsupported("Global variables not yet implemented"))
            }
            _ => Err(CodegenError::Unsupported("Unsupported operand type")),
        }
    }

    /// Converts a WasmIR type to Cranelift type
    fn convert_type(&self, wasmir_ty: &WasmIRType) -> Result<Type, CodegenError> {
        match wasmir_ty {
            WasmIRType::I32 => Ok(types::I32),
            WasmIRType::I64 => Ok(types::I64),
            WasmIRType::F32 => Ok(types::F32),
            WasmIRType::F64 => Ok(types::F64),
            WasmIRType::ExternRef(_) => Ok(types::R32), // Handle as i32 for now
            WasmIRType::FuncRef => Ok(types::R32), // Handle as i32 for now
            _ => Err(CodegenError::Unsupported("Unsupported type")),
        }
    }

    /// Converts a constant value to Cranelift-compatible value
    fn convert_constant(&self, value: &Constant) -> Result<i32, CodegenError> {
        match value {
            Constant::I32(v) => Ok(*v),
            Constant::I64(v) => Ok(*v as i32),
            Constant::F32(v) => Ok(v.to_bits() as i32),
            Constant::F64(v) => Ok(v.to_bits() as i32),
            Constant::Boolean(b) => Ok(if *b { 1 } else { 0 }),
            _ => Err(CodegenError::Unsupported("Unsupported constant type")),
        }
    }

    /// Applies WasmRust-specific optimizations to the function
    fn apply_optimizations(&mut self, func: &mut Function) -> Result<(), CodegenError> {
        if self.optimization_flags.thin_monomorphization {
            self.apply_thin_monomorphization(func)?;
        }

        if self.optimization_flags.streaming_layout {
            self.apply_streaming_layout(func)?;
        }

        if self.optimization_flags.wasm_optimizations {
            self.apply_wasm_optimizations(func)?;
        }

        self.stats.optimization_passes += 1;
        Ok(())
    }

    /// Applies thin monomorphization to reduce code duplication
    fn apply_thin_monomorphization(&mut self, _func: &mut Function) -> Result<(), CodegenError> {
        // Implementation for thin monomorphization
        // This would analyze generic functions and create specialized versions
        // for common monomorphic instantiations
        
        // For now, placeholder implementation
        Ok(())
    }

    /// Applies streaming layout optimization for fast WASM instantiation
    fn apply_streaming_layout(&mut self, _func: &mut Function) -> Result<(), CodegenError> {
        // Implementation for streaming layout optimization
        // This would arrange code layout for optimal streaming
        
        // For now, placeholder implementation
        Ok(())
    }

    /// Applies WASM-specific optimizations
    fn apply_wasm_optimizations(&mut self, _func: &mut Function) -> Result<(), CodegenError> {
        // Implementation of WASM-specific optimizations
        // This would include optimizations like:
        // - Zero-cost abstractions
        // - WASM instruction selection
        // - Memory access pattern optimization
        
        // For now, placeholder implementation
        Ok(())
    }

    /// Adds terminator instruction to a basic block
    fn add_block_terminator(
        &self,
        builder: &mut FunctionBuilder,
        terminator: &Terminator,
        block_map: &HashMap<BlockId, Block>,
    ) -> Result<(), CodegenError> {
        match terminator {
            Terminator::Return { value } => {
                if let Some(val) = value {
                    let converted_val = self.convert_operand(builder, val)?;
                    builder.ins().return_(&[converted_val]);
                } else {
                    builder.ins().return_(&[]);
                }
            }
            Terminator::Branch { condition, then_block, else_block } => {
                let cond_val = self.convert_operand(builder, condition)?;
                let then_block_ref = block_map[then_block];
                let else_block_ref = block_map[else_block];
                builder.ins().brif(cond_val, then_block_ref, &[], else_block_ref, &[]);
            }
            Terminator::Jump { target } => {
                let target_block = block_map[target];
                builder.ins().jump(target_block, &[]);
            }
            Terminator::Unreachable => {
                builder.ins().trap(cranelift_codegen::ir::TrapCode::UnreachableCodeReached);
            }
            Terminator::Panic { message: _ } => {
                builder.ins().trap(cranelift_codegen::ir::TrapCode::User(0));
            }
            _ => {
                // For now, handle other terminators as unreachable
                builder.ins().trap(cranelift_codegen::ir::TrapCode::UnreachableCodeReached);
            }
        }
        Ok(())
    }

    /// Hashes a function for caching purposes
    fn hash_function(&self, wasmir_func: &WasmIR) -> u64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        
        let mut hasher = DefaultHasher::new();
        wasmir_func.name.hash(&mut hasher);
        wasmir_func.signature.params.len().hash(&mut hasher);
        hasher.finish()
    }
}

/// Creates target ISA for compilation
fn create_target_isa() -> Result<Arc<dyn TargetIsa>, CodegenError> {
    use cranelift_codegen::isa;
    use cranelift_codegen::settings;
    use cranelift_native;
    
    let mut flag_builder = settings::builder();
    flag_builder.enable("enable_probestack").unwrap();
    flag_builder.enable("enable_jump_tables").unwrap();
    flag_builder.set("is_pic", "false").unwrap();
    
    // Use native target detection instead of hardcoded x86_64
    let isa_builder = cranelift_native::builder()
        .map_err(|_| CodegenError::TargetConfig("Failed to detect native target"))?;
    
    let isa = isa_builder.finish(settings::Flags::new(flag_builder))
        .map_err(|_| CodegenError::TargetConfig("Failed to create ISA"))?;
    
    Ok(isa)
}

/// Code generation errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodegenError {
    /// Unsupported operation or type
    Unsupported(&'static str),
    /// Type conversion error
    TypeConversion(&'static str),
    /// Instruction generation error
    InstructionGeneration(&'static str),
    /// Optimization error
    Optimization(&'static str),
    /// Target configuration error
    TargetConfig(&'static str),
}

impl std::fmt::Display for CodegenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodegenError::Unsupported(msg) => write!(f, "Unsupported operation: {}", msg),
            CodegenError::TypeConversion(msg) => write!(f, "Type conversion error: {}", msg),
            CodegenError::InstructionGeneration(msg) => write!(f, "Instruction generation error: {}", msg),
            CodegenError::Optimization(msg) => write!(f, "Optimization error: {}", msg),
            CodegenError::TargetConfig(msg) => write!(f, "Target configuration error: {}", msg),
        }
    }
}

impl std::error::Error for CodegenError {}

impl From<cranelift_codegen::CodegenError> for CodegenError {
    fn from(_err: cranelift_codegen::CodegenError) -> Self {
        CodegenError::InstructionGeneration("Cranelift codegen error")
    }
}

impl<'a> From<cranelift_codegen::CompileError<'a>> for CodegenError {
    fn from(_err: cranelift_codegen::CompileError<'a>) -> Self {
        CodegenError::InstructionGeneration("Cranelift compile error")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_creation() {
        let backend = WasmRustCraneliftBackend::new();
        assert!(backend.is_ok());
        
        let backend = backend.unwrap();
        assert_eq!(backend.get_stats().functions_compiled, 0);
    }

    #[test]
    fn test_optimization_flags() {
        let flags = WasmRustOptimizationFlags::default();
        assert!(flags.thin_monomorphization);
        assert!(flags.streaming_layout);
        assert!(flags.wasm_optimizations);
        assert!(flags.zero_cost_abstractions);
    }

    #[test]
    fn test_compilation_stats() {
        let mut stats = CompilationStats::default();
        assert_eq!(stats.functions_compiled, 0);
        
        stats.functions_compiled = 10;
        stats.instructions_generated = 1000;
        stats.optimization_passes = 5;
        stats.compilation_time_ms = 150;
        
        assert_eq!(stats.functions_compiled, 10);
        assert_eq!(stats.instructions_generated, 1000);
        assert_eq!(stats.optimization_passes, 5);
        assert_eq!(stats.compilation_time_ms, 150);
    }
}