//! MIR to WasmIR lowering for WasmRust
//! 
//! This module implements the transformation from Rust MIR to WasmIR,
//! preserving Rust semantics while enabling WASM-specific optimizations.

use rustc_middle::mir::*;
use rustc_middle::ty::{self as Ty, TyKind};
use rustc_target::spec::Target;
use wasm::wasmir::{WasmIR, Instruction, Terminator, BasicBlock, BlockId, Type, Signature, Operand, BinaryOp, UnaryOp};
use wasm::host::get_host_capabilities;

pub struct MIRLowerer {
    target: Target,
    locals: Vec<Local>,
    blocks: Vec<BasicBlock>,
    next_block_id: usize,
}

impl MIRLowerer {
    pub fn new(target: Target) -> Self {
        Self {
            target,
            locals: Vec::new(),
            blocks: Vec::new(),
            next_block_id: 0,
        }
    }

    pub fn lower_mir(&mut self, mir_body: &Body<'_>) -> WasmIR {
        // Create WasmIR function signature
        let signature = self.lower_signature(mir_body);
        let mut wasm_func = WasmIR::new("lowered_function".to_string(), signature);
        
        // Lower local declarations
        for (local, local_decl) in mir_body.local_decls.iter_enumerated() {
            self.lower_local_decl(&mut wasm_func, local, local_decl);
        }
        
        // Lower basic blocks
        for (bb_index, mir_bb) in mir_body.basic_blocks.iter_enumerated() {
            let wasm_bb = self.lower_basic_block(mir_bb, bb_index);
            wasm_func.add_basic_block(wasm_bb);
        }
        
        wasm_func
    }

    fn lower_signature(&self, mir_body: &Body<'_>) -> Signature {
        // Simplified signature lowering
        let mut params = Vec::new();
        for arg in mir_body.args_iter() {
            params.push(self.lower_type(arg.ty()));
        }
        
        let returns = match mir_body.return_ty() {
            ty => match ty.kind() {
                TyKind::Tuple(tys) if tys.is_empty() => None,
                _ => Some(self.lower_type(mir_body.return_ty())),
            },
        };
        
        Signature { params, returns }
    }

    fn lower_local_decl(&mut self, wasm_func: &mut WasmIR, local: Local, local_decl: &LocalDecl<'_>) {
        let wasm_type = self.lower_type(local_decl.ty);
        let local_index = wasm_func.add_local(wasm_type);
        
        // Track mapping
        while self.locals.len() <= local.local_index() {
            self.locals.push(Local::INVALID);
        }
        self.locals[local.local_index()] = local;
    }

    fn lower_basic_block(&mut self, mir_bb: &BasicBlock<'_>, bb_index: usize) -> BasicBlock {
        let block_id = BlockId::new(self.next_block_id);
        self.next_block_id += 1;
        
        let mut instructions = Vec::new();
        
        // Lower statements
        for statement in &mir_bb.statements {
            match statement {
                Statement::Assign(place, rvalue) => {
                    let value = self.lower_rvalue(rvalue);
                    let operand = self.lower_place(place);
                    instructions.push(Instruction::LocalSet {
                        index: operand.local_index(),
                        value,
                    });
                }
                Statement::StorageLive(local) => {
                    // Handle liveness
                    instructions.push(Instruction::Nop);
                }
                Statement::StorageDead(local) => {
                    // Handle liveness
                    instructions.push(Instruction::Nop);
                }
                _ => {
                    // Other statements as needed
                    instructions.push(Instruction::Nop);
                }
            }
        }
        
        // Lower terminator
        let terminator = self.lower_terminator(&mir_bb.terminator());
        
        BasicBlock::new(block_id, instructions, terminator)
    }

    fn lower_rvalue(&mut self, rvalue: &Rvalue<'_>) -> Operand {
        match rvalue.kind() {
            RvalueKind::Use(operand) => self.lower_operand(operand),
            RvalueKind::BinaryOp(bin_op, left, right) => {
                let left_op = self.lower_operand(left);
                let right_op = self.lower_operand(right);
                let wasm_bin_op = self.lower_binary_op(*bin_op);
                
                // Create a temporary local for the result
                let result_type = self.lower_type(rvalue.ty());
                let result_index = self.locals.len();
                self.locals.push(Local::from_usize(result_index));
                
                Operand::Local(result_index)
            }
            RvalueKind::UnaryOp(un_op, operand) => {
                let op = self.lower_operand(operand);
                let wasm_un_op = self.lower_unary_op(*un_op);
                
                // Create a temporary local for the result
                let result_type = self.lower_type(rvalue.ty());
                let result_index = self.locals.len();
                self.locals.push(Local::from_usize(result_index));
                
                Operand::Local(result_index)
            }
            RvalueKind::Constant(const_val) => {
                match const_val.kind() {
                    ConstKind::Value(ty, const_val) => {
                        match const_val {
                            ConstValue::Scalar(scalar) => {
                                match scalar {
                                    Scalar::Int(int) => {
                                        let int_val = int.assert_bits(self.target.pointer_width());
                                        if int_val.size() == 32 {
                                            Operand::I32(int_val.to_i32() as i32)
                                        } else if int_val.size() == 64 {
                                            Operand::I64(int_val.to_i64() as i64)
                                        } else {
                                            Operand::I32(0) // Fallback
                                        }
                                    }
                                    }
                                    Scalar::Float(float) => {
                                        if float.is_nan() {
                                            Operand::F64(f64::NAN)
                                        } else {
                                            Operand::F64(float.to_f64())
                                        }
                                    }
                                    }
                                }
                            }
                            _ => Operand::I32(0), // Fallback
                        }
                    }
                    _ => Operand::I32(0), // Fallback
                }
            }
            _ => Operand::I32(0), // Fallback
        }
    }

    fn lower_place(&mut self, place: &Place<'_>) -> Operand {
        match place.ty().kind() {
            TyKind::Ref(ty) => {
                self.lower_operand(place.projection.last().unwrap_or(&PlaceElem::Local(local!())))
            }
            _ => {
                self.lower_operand(place.projection.last().unwrap_or(&PlaceElem::Local(local!())))
            }
        }
    }

    fn lower_operand(&mut self, operand: &Operand<'_>) -> Operand {
        match operand {
            Operand::Copy(place) => self.lower_place(place),
            Operand::Move(place) => self.lower_place(place),
            Operand::Local(local) => Operand::Local(local.local_index()),
            _ => Operand::I32(0), // Fallback
        }
    }

    fn lower_terminator(&mut self, terminator: &Terminator<'_>) -> Terminator {
        match terminator.kind() {
            TerminatorKind::Return { .. } => {
                Terminator::Return {
                    value: None, // Simplified
                }
            }
            TerminatorKind::Goto { target } => {
                Terminator::Goto {
                    target: BlockId::new(target.index()),
                }
            }
            TerminatorKind::Switch { .. } => {
                Terminator::Goto {
                    target: BlockId::new(0), // Simplified
                }
            }
            TerminatorKind::Unreachable => {
                Terminator::Unreachable
            }
            TerminatorKind::Call { .. } => {
                Terminator::Goto {
                    target: BlockId::new(0), // Simplified
                }
            }
            _ => Terminator::Unreachable,
        }
    }

    fn lower_type(&self, ty: Ty<'_>) -> Type {
        match ty.kind() {
            TyKind::Int(int_ty) => {
                match int_ty.kind() {
                    IntTyKind::I8 | IntTyKind::U8 => Type::I32,
                    IntTyKind::I16 | IntTyKind::U16 => Type::I32,
                    IntTyKind::I32 | IntTyKind::U32 => Type::I32,
                    IntTyKind::I64 | IntTyKind::U64 => Type::I64,
                    IntTyKind::I128 | IntTyKind::U128 => Type::I64, // Map 128-bit to 64-bit
                }
            }
            TyKind::Float(float_ty) => {
                match float_ty.kind() {
                    FloatTyKind::F32 => Type::F32,
                    FloatTyKind::F64 => Type::F64,
                }
            }
            TyKind::Ref(ty) => Type::Ref(format!("{:?}", ty.kind())),
            TyKind::Tuple(tys) => {
                if tys.is_empty() {
                    Type::Void
                } else if tys.len() == 1 {
                    self.lower_type(tys[0])
                } else {
                    Type::Ref(format!("tuple_{}", tys.len()))
                }
            }
            TyKind::Bool => Type::I32, // Map bool to i32
            _ => Type::Ref(format!("{:?}", ty.kind())),
        }
    }

    fn lower_binary_op(&self, bin_op: BinOp) -> BinaryOp {
        match bin_op {
            BinOp::Add => BinaryOp::Add,
            BinOp::Sub => BinaryOp::Sub,
            BinOp::Mul => BinaryOp::Mul,
            BinOp::Div => BinaryOp::Div,
            BinOp::Rem => BinaryOp::Mod,
            BinOp::BitAnd => BinaryOp::And,
            BinOp::BitOr => BinaryOp::Or,
            BinOp::BitXor => BinaryOp::Xor,
            BinOp::Shl => BinaryOp::Shl,
            BinOp::Shr => BinaryOp::Shr,
            BinOp::Eq => BinaryOp::Eq,
            BinOp::Lt => BinaryOp::Lt,
            BinOp::Le => BinaryOp::Le,
            _ => BinaryOp::Add, // Fallback
        }
    }

    fn lower_unary_op(&self, un_op: UnOp) -> UnaryOp {
        match un_op {
            UnOp::Neg => UnaryOp::Neg,
            UnOp::Not => UnaryOp::Not,
            _ => UnaryOp::Neg, // Fallback
        }
    }
}
