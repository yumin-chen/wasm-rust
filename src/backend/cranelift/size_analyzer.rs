//! Size Analysis Tool for WasmRust
//! 
//! This module implements comprehensive size analysis for WasmRust functions,
//! providing detailed metrics, type impact analysis, and optimization
//! suggestions for thin monomorphization.

use crate::wasmir::{WasmIR, Type, Instruction, Terminator, Operand};
use crate::backend::cranelift::type_descriptor::WasmTypeDescriptor;
use rustc_target::spec::Target;
use std::collections::{HashMap, BTreeMap};

/// Size analysis report containing comprehensive analysis results
#[derive(Debug, Clone)]
pub struct SizeAnalysisReport {
    /// Total code size before optimization
    pub total_size_before: usize,
    /// Total code size after optimization
    pub total_size_after: usize,
    /// Size reduction percentage
    pub size_reduction: f64,
    /// Function-level metrics
    pub function_metrics: HashMap<String, FunctionMetrics>,
    /// Type impact analysis
    pub type_impact: HashMap<String, TypeImpact>,
    /// Optimization suggestions
    pub suggestions: Vec<OptimizationSuggestion>,
    /// Functions that were optimized
    pub optimized_functions: Vec<String>,
    /// Analysis timestamp
    pub timestamp: String,
}

/// Detailed metrics for a single function
#[derive(Debug, Clone)]
pub struct FunctionMetrics {
    /// Function name
    pub name: String,
    /// Code size in bytes
    pub code_size: usize,
    /// Number of instructions
    pub instruction_count: usize,
    /// Number of basic blocks
    pub basic_block_count: usize,
    /// Generic complexity score
    pub generic_complexity: f64,
    /// Inlining potential (0.0 to 1.0)
    pub inlining_potential: f64,
    /// Thinning suitability (0.0 to 1.0)
    pub thinning_suitability: f64,
    /// Size contribution percentage
    pub size_contribution: f64,
    /// Types used in this function
    pub used_types: Vec<String>,
    /// Performance characteristics
    pub performance_profile: PerformanceProfile,
}

/// Impact analysis for a specific type
#[derive(Debug, Clone)]
pub struct TypeImpact {
    /// Type name
    pub type_name: String,
    /// Total code size impact
    pub code_size_impact: usize,
    /// Number of functions using this type
    pub usage_count: usize,
    /// Average function size using this type
    pub average_function_size: f64,
    /// Generic complexity contribution
    pub generic_complexity_contribution: f64,
    /// Optimization potential
    pub optimization_potential: OptimizationPotential,
}

/// Performance profile for a function
#[derive(Debug, Clone)]
pub struct PerformanceProfile {
    /// Call frequency estimate
    pub call_frequency: CallFrequency,
    /// Memory access pattern
    pub memory_pattern: MemoryPattern,
    /// Control flow complexity
    pub control_flow_complexity: ControlFlowComplexity,
    /// Hot path analysis
    pub hot_path_percentage: f64,
}

/// Call frequency estimation
#[derive(Debug, Clone, PartialEq)]
pub enum CallFrequency {
    /// Function called rarely (e.g., error handlers)
    Rare,
    /// Function called occasionally (e.g., initialization)
    Occasional,
    /// Function called frequently (e.g., utility functions)
    Frequent,
    /// Function called very frequently (e.g., core operations)
    VeryFrequent,
    /// Unknown frequency
    Unknown,
}

/// Memory access patterns
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryPattern {
    /// Sequential access (good for caching)
    Sequential,
    /// Random access (may cause cache misses)
    Random,
    /// Mixed access pattern
    Mixed,
    /// Streaming access (good for prefetching)
    Streaming,
    /// Unknown pattern
    Unknown,
}

/// Control flow complexity classification
#[derive(Debug, Clone, PartialEq)]
pub enum ControlFlowComplexity {
    /// Simple linear control flow
    Linear,
    /// Simple branching
    Branching,
    /// Contains loops
    Loopy,
    /// Complex nested control flow
    Complex,
    /// Recursive control flow
    Recursive,
}

/// Optimization potential for a type
#[derive(Debug, Clone)]
pub struct OptimizationPotential {
    /// Thinning potential (0.0 to 1.0)
    pub thinning_potential: f64,
    /// Monomorphization potential (0.0 to 1.0)
    pub monomorphization_potential: f64,
    /// Inlining potential (0.0 to 1.0)
    pub inlining_potential: f64,
    /// Streaming layout potential (0.0 to 1.0)
    pub streaming_potential: f64,
}

/// Optimization suggestion for improving code size
#[derive(Debug, Clone)]
pub struct OptimizationSuggestion {
    /// Suggestion type
    pub suggestion_type: SuggestionType,
    /// Target function or type
    pub target: String,
    /// Expected size reduction
    pub expected_reduction: usize,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Detailed description
    pub description: String,
    /// Implementation complexity
    pub implementation_complexity: ImplementationComplexity,
}

/// Types of optimization suggestions
#[derive(Debug, Clone, PartialEq)]
pub enum SuggestionType {
    /// Apply thin monomorphization
    Thinning,
    /// Function inlining
    Inlining,
    /// Code deduplication
    Deduplication,
    /// Constant folding
    ConstantFolding,
    /// Dead code elimination
    DeadCodeElimination,
    /// Loop unrolling
    LoopUnrolling,
    /// Type specialization
    TypeSpecialization,
    /// Streaming layout optimization
    StreamingLayout,
}

/// Implementation complexity for suggestions
#[derive(Debug, Clone, PartialEq)]
pub enum ImplementationComplexity {
    /// Easy to implement
    Easy,
    /// Moderate difficulty
    Moderate,
    /// Difficult to implement
    Difficult,
    /// Very difficult
    VeryDifficult,
}

/// Size analyzer implementation
pub struct SizeAnalyzer {
    /// Target architecture
    target: Target,
    /// Last analysis report
    last_report: Option<SizeAnalysisReport>,
    /// Analysis configuration
    config: AnalysisConfig,
}

/// Configuration for size analysis
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    /// Include detailed instruction analysis
    pub detailed_instruction_analysis: bool,
    /// Include type impact analysis
    pub include_type_impact: bool,
    /// Include optimization suggestions
    pub include_suggestions: bool,
    /// Threshold for function significance (percentage)
    pub significance_threshold: f64,
    /// Enable performance profiling
    pub enable_performance_profiling: bool,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            detailed_instruction_analysis: true,
            include_type_impact: true,
            include_suggestions: true,
            significance_threshold: 1.0, // 1% of total size
            enable_performance_profiling: true,
        }
    }
}

impl SizeAnalyzer {
    /// Creates a new size analyzer
    pub fn new(target: Target) -> Self {
        Self {
            target,
            last_report: None,
            config: AnalysisConfig::default(),
        }
    }

    /// Creates a size analyzer with custom configuration
    pub fn with_config(target: Target, config: AnalysisConfig) -> Self {
        Self {
            target,
            last_report: None,
            config,
        }
    }

    /// Analyzes WasmIR functions and generates comprehensive report
    pub fn analyze_functions(
        &mut self,
        functions: &[WasmIR],
    ) -> Result<SizeAnalysisReport, AnalysisError> {
        let mut function_metrics = HashMap::new();
        let mut total_size_before = 0;

        // Phase 1: Analyze individual functions
        for function in functions {
            let metrics = self.analyze_function(function)?;
            function_metrics.insert(function.name.clone(), metrics.clone());
            total_size_before += metrics.code_size;
        }

        // Phase 2: Type impact analysis
        let type_impact = if self.config.include_type_impact {
            self.analyze_type_impact(&function_metrics)?
        } else {
            HashMap::new()
        };

        // Phase 3: Generate optimization suggestions
        let suggestions = if self.config.include_suggestions {
            self.generate_optimization_suggestions(&function_metrics, &type_impact)?
        } else {
            Vec::new()
        };

        // Phase 4: Estimate optimized size
        let total_size_after = self.estimate_optimized_size(&function_metrics, &suggestions)?;
        let size_reduction = if total_size_before > 0 {
            ((total_size_before - total_size_after) as f64 / total_size_before as f64) * 100.0
        } else {
            0.0
        };

        // Phase 5: Identify optimized functions
        let optimized_functions = suggestions
            .iter()
            .filter(|s| matches!(s.suggestion_type, SuggestionType::Thinning | SuggestionType::Deduplication))
            .map(|s| s.target.clone())
            .collect();

        let report = SizeAnalysisReport {
            total_size_before,
            total_size_after,
            size_reduction,
            function_metrics,
            type_impact,
            suggestions,
            optimized_functions,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        self.last_report = Some(report.clone());
        Ok(report)
    }

    /// Analyzes a single function
    fn analyze_function(&self, function: &WasmIR) -> Result<FunctionMetrics, AnalysisError> {
        let code_size = self.estimate_function_size(function)?;
        let instruction_count = function.instruction_count();
        let basic_block_count = function.basic_blocks.len();
        let used_types = self.extract_used_types(function);

        // Calculate generic complexity
        let generic_complexity = self.calculate_generic_complexity(function, &used_types);

        // Analyze performance profile
        let performance_profile = if self.config.enable_performance_profiling {
            self.analyze_performance_profile(function)?
        } else {
            PerformanceProfile::default()
        };

        // Calculate optimization potentials
        let inlining_potential = self.calculate_inlining_potential(function);
        let thinning_suitability = self.calculate_thinning_suitability(function, &used_types);

        let metrics = FunctionMetrics {
            name: function.name.clone(),
            code_size,
            instruction_count,
            basic_block_count,
            generic_complexity,
            inlining_potential,
            thinning_suitability,
            size_contribution: 0.0, // Will be calculated later
            used_types,
            performance_profile,
        };

        Ok(metrics)
    }

    /// Estimates the size of a function in bytes
    fn estimate_function_size(&self, function: &WasmIR) -> Result<usize, AnalysisError> {
        let mut total_size = 0;

        // Base size for function signature and metadata
        total_size += 16; // Function header overhead

        // Size per instruction
        for instruction in function.all_instructions() {
            total_size += self.estimate_instruction_size(instruction)?;
        }

        // Size per terminator
        for basic_block in &function.basic_blocks {
            total_size += self.estimate_terminator_size(&basic_block.terminator)?;
        }

        // Size per local variable
        for local_type in &function.locals {
            total_size += self.estimate_type_size(local_type)?;
        }

        Ok(total_size)
    }

    /// Estimates the size of an instruction in bytes
    fn estimate_instruction_size(&self, instruction: &Instruction) -> Result<usize, AnalysisError> {
        let base_size = match instruction {
            Instruction::LocalGet { .. } => 2,
            Instruction::LocalSet { .. } => 3,
            Instruction::BinaryOp { .. } => 2,
            Instruction::UnaryOp { .. } => 2,
            Instruction::Call { args, .. } => 1 + args.len(),
            Instruction::Return { .. } => 1,
            Instruction::Branch { .. } => 2,
            Instruction::Jump { .. } => 1,
            Instruction::Switch { targets, .. } => 1 + targets.len(),
            Instruction::MemoryLoad { .. } => 3,
            Instruction::MemoryStore { .. } => 3,
            Instruction::MemoryAlloc { .. } => 2,
            Instruction::MemoryFree { .. } => 1,
            Instruction::NewObject { args, .. } => 2 + args.len(),
            Instruction::DropObject { .. } => 1,
            Instruction::ExternRefLoad { .. } => 4,
            Instruction::ExternRefStore { .. } => 4,
            Instruction::JSMethodCall { args, .. } => 3 + args.len(),
            Instruction::MakeFuncRef { .. } => 2,
            Instruction::FuncRefCall { args, .. } => 1 + args.len(),
            Instruction::ExternRefNew { .. } => 2,
            Instruction::ExternRefCast { .. } => 2,
            Instruction::ExternRefIsNull { .. } => 1,
            Instruction::ExternRefEq { .. } => 2,
            Instruction::FuncRefNew { .. } => 1,
            Instruction::FuncRefIsNull { .. } => 1,
            Instruction::FuncRefEq { .. } => 2,
            Instruction::CallIndirect { args, .. } => 2 + args.len(),
            Instruction::AtomicOp { .. } => 3,
            Instruction::CompareExchange { .. } => 4,
            Instruction::LinearOp { .. } => 2,
            Instruction::CapabilityCheck { .. } => 1,
            Instruction::Nop => 1,
        };

        // Add operand sizes
        let operand_size = self.estimate_instruction_operand_size(instruction)?;

        Ok(base_size + operand_size)
    }

    /// Estimates the size of operands in an instruction
    fn estimate_instruction_operand_size(&self, instruction: &Instruction) -> Result<usize, AnalysisError> {
        let mut operand_size = 0;

        match instruction {
            Instruction::BinaryOp { left, right, .. } => {
                operand_size += self.estimate_operand_size(left)?;
                operand_size += self.estimate_operand_size(right)?;
            }
            Instruction::UnaryOp { value, .. } => {
                operand_size += self.estimate_operand_size(value)?;
            }
            Instruction::Call { args, .. } => {
                for arg in args {
                    operand_size += self.estimate_operand_size(arg)?;
                }
            }
            Instruction::MemoryStore { address, value, .. } => {
                operand_size += self.estimate_operand_size(address)?;
                operand_size += self.estimate_operand_size(value)?;
            }
            Instruction::MemoryLoad { address, .. } => {
                operand_size += self.estimate_operand_size(address)?;
            }
            Instruction::Branch { condition, .. } => {
                operand_size += self.estimate_operand_size(condition)?;
            }
            Instruction::NewObject { args, .. } => {
                for arg in args {
                    operand_size += self.estimate_operand_size(arg)?;
                }
            }
            Instruction::JSMethodCall { object, args, .. } => {
                operand_size += self.estimate_operand_size(object)?;
                for arg in args {
                    operand_size += self.estimate_operand_size(arg)?;
                }
            }
            Instruction::FuncRefCall { args, .. } => {
                operand_size += self.estimate_operand_size(&/* funcref operand */)?;
                for arg in args {
                    operand_size += self.estimate_operand_size(arg)?;
                }
            }
            Instruction::Return { value } => {
                if let Some(val) = value {
                    operand_size += self.estimate_operand_size(val)?;
                }
            }
            _ => {}
        }

        Ok(operand_size)
    }

    /// Estimates the size of an operand in bytes
    fn estimate_operand_size(&self, operand: &Operand) -> Result<usize, AnalysisError> {
        match operand {
            Operand::Local(_) => Ok(1), // Local index
            Operand::Constant(_) => Ok(4), // 32-bit constant
            Operand::Global(_) => Ok(1), // Global index
            Operand::FunctionRef(_) => Ok(1), // Function index
            Operand::ExternRef(_) => Ok(1), // ExternRef handle
            Operand::FuncRef(_) => Ok(1), // FuncRef index
            Operand::MemoryAddress(addr) => {
                self.estimate_operand_size(addr)
            }
            Operand::StackValue(_) => Ok(1), // Stack index
        }
    }

    /// Estimates the size of a terminator
    fn estimate_terminator_size(&self, terminator: &Terminator) -> Result<usize, AnalysisError> {
        let base_size = match terminator {
            Terminator::Return { .. } => 1,
            Terminator::Branch { .. } => 2,
            Terminator::Switch { targets, .. } => 1 + targets.len(),
            Terminator::Jump { .. } => 1,
            Terminator::Unreachable => 1,
            Terminator::Panic { .. } => 2,
        };

        // Add operand sizes for terminators with operands
        let operand_size = match terminator {
            Terminator::Return { value } => {
                if let Some(val) = value {
                    self.estimate_operand_size(val)?
                } else {
                    0
                }
            }
            Terminator::Branch { condition, .. } => self.estimate_operand_size(condition)?,
            Terminator::Switch { value, .. } => self.estimate_operand_size(value)?,
            Terminator::Panic { message } => {
                if let Some(msg) = message {
                    self.estimate_operand_size(msg)?
                } else {
                    0
                }
            }
            _ => 0,
        };

        Ok(base_size + operand_size)
    }

    /// Estimates the size of a type in bytes
    fn estimate_type_size(&self, ty: &Type) -> Result<usize, AnalysisError> {
        match ty {
            Type::I32 => Ok(4),
            Type::I64 => Ok(8),
            Type::F32 => Ok(4),
            Type::F64 => Ok(8),
            Type::ExternRef(_) => Ok(4), // Handle
            Type::FuncRef => Ok(4), // Index
            Type::Array { element_type, size } => {
                let elem_size = self.estimate_type_size(element_type)?;
                let array_size = size.unwrap_or(1);
                Ok(elem_size * array_size)
            }
            Type::Struct { fields } => {
                let mut total_size = 0;
                for field in fields {
                    total_size += self.estimate_type_size(field)?;
                }
                Ok(total_size)
            }
            Type::Pointer(_) => Ok(4), // 32-bit pointer
            Type::Linear { inner_type } => self.estimate_type_size(inner_type),
            Type::Capability { inner_type, .. } => self.estimate_type_size(inner_type),
            Type::Void => Ok(0),
        }
    }

    /// Extracts all types used in a function
    fn extract_used_types(&self, function: &WasmIR) -> Vec<String> {
        let mut used_types = std::collections::HashSet::new();

        // Extract from signature
        for param_type in &function.signature.params {
            used_types.insert(self.type_to_string(param_type));
        }
        if let Some(ret_type) = &function.signature.returns {
            used_types.insert(self.type_to_string(ret_type));
        }

        // Extract from locals
        for local_type in &function.locals {
            used_types.insert(self.type_to_string(local_type));
        }

        // Extract from instructions
        for instruction in function.all_instructions() {
            self.extract_types_from_instruction(instruction, &mut used_types);
        }

        used_types.into_iter().collect()
    }

    /// Extracts types from an instruction
    fn extract_types_from_instruction(&self, instruction: &Instruction, used_types: &mut std::collections::HashSet<String>) {
        match instruction {
            Instruction::MemoryLoad { ty, .. } => {
                used_types.insert(self.type_to_string(ty));
            }
            Instruction::MemoryStore { ty, .. } => {
                used_types.insert(self.type_to_string(ty));
            }
            Instruction::NewObject { .. } => {
                // Would extract type from type_id in practice
            }
            Instruction::ExternRefLoad { field_type, .. } => {
                used_types.insert(self.type_to_string(field_type));
            }
            Instruction::ExternRefStore { field_type, .. } => {
                used_types.insert(self.type_to_string(field_type));
            }
            Instruction::JSMethodCall { return_type, .. } => {
                if let Some(ret_ty) = return_type {
                    used_types.insert(self.type_to_string(ret_ty));
                }
            }
            Instruction::MakeFuncRef { signature, .. } => {
                used_types.extend(self.extract_types_from_signature(signature));
            }
            Instruction::FuncRefCall { signature, .. } => {
                used_types.extend(self.extract_types_from_signature(signature));
            }
            Instruction::ExternRefNew { target_type, .. } => {
                used_types.insert(self.type_to_string(target_type));
            }
            Instruction::ExternRefCast { target_type, .. } => {
                used_types.insert(self.type_to_string(target_type));
            }
            _ => {}
        }
    }

    /// Extracts types from a function signature
    fn extract_types_from_signature(&self, signature: &crate::wasmir::Signature) -> Vec<String> {
        let mut types = Vec::new();

        for param_type in &signature.params {
            types.push(self.type_to_string(param_type));
        }

        if let Some(ret_type) = &signature.returns {
            types.push(self.type_to_string(ret_type));
        }

        types
    }

    /// Converts a type to string representation
    fn type_to_string(&self, ty: &Type) -> String {
        match ty {
            Type::I32 => "i32".to_string(),
            Type::I64 => "i64".to_string(),
            Type::F32 => "f32".to_string(),
            Type::F64 => "f64".to_string(),
            Type::ExternRef(name) => format!("externref_{}", name),
            Type::FuncRef => "funcref".to_string(),
            Type::Array { element_type, size } => {
                let elem_str = self.type_to_string(element_type);
                if let Some(s) = size {
                    format!("[{}; {}]", elem_str, s)
                } else {
                    format!("[{}]", elem_str)
                }
            }
            Type::Struct { fields } => {
                let field_strings: Vec<_> = fields.iter()
                    .map(|f| self.type_to_string(f))
                    .collect();
                format!("Struct({})", field_strings.join(", "))
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

    /// Calculates generic complexity score for a function
    fn calculate_generic_complexity(&self, function: &WasmIR, used_types: &[String]) -> f64 {
        let mut complexity = 0.0;

        // Base complexity from instruction count
        complexity += function.instruction_count() as f64 * 0.1;

        // Complexity from basic blocks
        complexity += function.basic_blocks.len() as f64 * 1.0;

        // Complexity from generic type usage
        let generic_type_count = used_types.iter()
            .filter(|ty| ty.starts_with("T") || ty.contains('<'))
            .count();
        complexity += generic_type_count as f64 * 3.0;

        // Complexity from control flow
        let loop_count = function.basic_blocks.iter()
            .filter(|bb| self.basic_block_has_loop(bb))
            .count();
        complexity += loop_count as f64 * 5.0;

        complexity
    }

    /// Checks if a basic block contains a loop
    fn basic_block_has_loop(&self, basic_block: &crate::wasmir::BasicBlock) -> bool {
        // Simplified loop detection - check for back edges
        match &basic_block.terminator {
            Terminator::Jump { target } => {
                target.0 < basic_block.id.0 // Back edge indicates loop
            }
            Terminator::Branch { then_block, else_block, .. } => {
                then_block.0 < basic_block.id.0 || else_block.0 < basic_block.id.0
            }
            _ => false,
        }
    }

    /// Analyzes performance profile of a function
    fn analyze_performance_profile(&self, function: &WasmIR) -> Result<PerformanceProfile, AnalysisError> {
        let call_frequency = self.estimate_call_frequency(function);
        let memory_pattern = self.analyze_memory_pattern(function);
        let control_flow_complexity = self.analyze_control_flow_complexity(function);
        let hot_path_percentage = self.calculate_hot_path_percentage(function);

        Ok(PerformanceProfile {
            call_frequency,
            memory_pattern,
            control_flow_complexity,
            hot_path_percentage,
        })
    }

    /// Estimates call frequency based on function characteristics
    fn estimate_call_frequency(&self, function: &WasmIR) -> CallFrequency {
        let name = &function.name;

        // Heuristics based on function name patterns
        if name.starts_with("__wasmrust_") || name.contains("init") {
            CallFrequency::Occasional
        } else if name.contains("panic") || name.contains("error") {
            CallFrequency::Rare
        } else if name.contains("core") || name.contains("util") {
            CallFrequency::VeryFrequent
        } else if name.len() < 5 {
            CallFrequency::Frequent
        } else {
            CallFrequency::Unknown
        }
    }

    /// Analyzes memory access patterns
    fn analyze_memory_pattern(&self, function: &WasmIR) -> MemoryPattern {
        let mut sequential_accesses = 0;
        let mut random_accesses = 0;

        for instruction in function.all_instructions() {
            match instruction {
                Instruction::MemoryLoad { address, .. } => {
                    if self.is_sequential_access(address) {
                        sequential_accesses += 1;
                    } else {
                        random_accesses += 1;
                    }
                }
                Instruction::MemoryStore { address, .. } => {
                    if self.is_sequential_access(address) {
                        sequential_accesses += 1;
                    } else {
                        random_accesses += 1;
                    }
                }
                _ => {}
            }
        }

        let total_accesses = sequential_accesses + random_accesses;
        if total_accesses == 0 {
            return MemoryPattern::Unknown;
        }

        let sequential_ratio = sequential_accesses as f64 / total_accesses as f64;
        if sequential_ratio > 0.8 {
            MemoryPattern::Sequential
        } else if sequential_ratio > 0.5 {
            MemoryPattern::Mixed
        } else {
            MemoryPattern::Random
        }
    }

    /// Checks if memory access is sequential
    fn is_sequential_access(&self, address: &Operand) -> bool {
        // Simplified heuristic - check for simple arithmetic patterns
        match address {
            Operand::BinaryOp { op, left, right } => {
                matches!(op, BinaryOp::Add | BinaryOp::Sub) &&
                self.is_simple_operand(left) && self.is_simple_operand(right)
            }
            _ => false,
        }
    }

    /// Checks if an operand is simple (constant or local)
    fn is_simple_operand(&self, operand: &Operand) -> bool {
        matches!(operand, Operand::Local(_) | Operand::Constant(_))
    }

    /// Analyzes control flow complexity
    fn analyze_control_flow_complexity(&self, function: &WasmIR) -> ControlFlowComplexity {
        let basic_block_count = function.basic_blocks.len();
        let loop_count = function.basic_blocks.iter()
            .filter(|bb| self.basic_block_has_loop(bb))
            .count();

        if loop_count > 0 {
            ControlFlowComplexity::Loopy
        } else if basic_block_count > 10 {
            ControlFlowComplexity::Complex
        } else if basic_block_count > 3 {
            ControlFlowComplexity::Branching
        } else {
            ControlFlowComplexity::Linear
        }
    }

    /// Calculates hot path percentage
    fn calculate_hot_path_percentage(&self, function: &WasmIR) -> f64 {
        // Simplified heuristic - assume early returns indicate hot path
        let early_return_blocks = function.basic_blocks.iter()
            .filter(|bb| matches!(&bb.terminator, Terminator::Return { .. }))
            .count();

        if function.basic_blocks.is_empty() {
            0.0
        } else {
            (early_return_blocks as f64 / function.basic_blocks.len() as f64) * 100.0
        }
    }

    /// Calculates inlining potential
    fn calculate_inlining_potential(&self, function: &WasmIR) -> f64 {
        let instruction_count = function.instruction_count();
        let basic_block_count = function.basic_blocks.len();

        // Small functions with simple control flow are good inlining candidates
        let size_factor = if instruction_count < 20 { 1.0 } else { 0.5 };
        let complexity_factor = if basic_block_count < 5 { 1.0 } else { 0.3 };

        size_factor * complexity_factor
    }

    /// Calculates thinning suitability
    fn calculate_thinning_suitability(&self, function: &WasmIR, used_types: &[String]) -> f64 {
        let instruction_count = function.instruction_count();
        let generic_type_count = used_types.iter()
            .filter(|ty| ty.starts_with("T") || ty.contains('<'))
            .count();

        // Functions with moderate size and generic usage are good candidates
        let size_factor = if instruction_count > 10 && instruction_count < 100 { 1.0 } else { 0.3 };
        let generic_factor = if generic_type_count > 0 { 1.0 } else { 0.0 };

        size_factor * generic_factor
    }

    /// Analyzes type impact across all functions
    fn analyze_type_impact(
        &self,
        function_metrics: &HashMap<String, FunctionMetrics>,
    ) -> Result<HashMap<String, TypeImpact>, AnalysisError> {
        let mut type_impact = HashMap::new();
        let mut type_usage: HashMap<String, Vec<usize>> = HashMap::new();
        let mut type_sizes: HashMap<String, Vec<usize>> = HashMap::new();

        // Collect usage and size data for each type
        for (function_name, metrics) in function_metrics {
            for type_name in &metrics.used_types {
                type_usage.entry(type_name.clone()).or_insert_with(Vec::new).push(metrics.code_size);
                type_sizes.entry(type_name.clone()).or_insert_with(Vec::new).push(metrics.code_size);
            }
        }

        // Calculate impact metrics for each type
        for (type_name, usage_sizes) in type_usage {
            let usage_count = usage_sizes.len();
            let total_size_impact: usize = usage_sizes.iter().sum();
            let average_function_size = usage_sizes.iter().sum::<usize>() as f64 / usage_count as f64;
            let generic_complexity_contribution = self.calculate_type_generic_complexity(type_name, function_metrics);

            let optimization_potential = OptimizationPotential {
                thinning_potential: self.calculate_type_thinning_potential(type_name, &usage_sizes),
                monomorphization_potential: self.calculate_type_monomorphization_potential(type_name, &usage_sizes),
                inlining_potential: self.calculate_type_inlining_potential(type_name, &usage_sizes),
                streaming_potential: self.calculate_type_streaming_potential(type_name, &usage_sizes),
            };

            type_impact.insert(type_name.clone(), TypeImpact {
                type_name: type_name.clone(),
                code_size_impact: total_size_impact,
                usage_count,
                average_function_size,
                generic_complexity_contribution,
                optimization_potential,
            });
        }

        Ok(type_impact)
    }

    /// Calculates generic complexity contribution for a type
    fn calculate_type_generic_complexity(&self, type_name: &str, function_metrics: &HashMap<String, FunctionMetrics>) -> f64 {
        function_metrics
            .values()
            .filter(|metrics| metrics.used_types.contains(&type_name.to_string()))
            .map(|metrics| metrics.generic_complexity)
            .sum::<f64>() / function_metrics.len() as f64
    }

    /// Calculates thinning potential for a type
    fn calculate_type_thinning_potential(&self, _type_name: &str, usage_sizes: &[usize]) -> f64 {
        // Types used in multiple medium-sized functions are good candidates
        if usage_sizes.len() > 1 && usage_sizes.iter().any(|&size| size > 50 && size < 200) {
            0.8
        } else if usage_sizes.len() > 1 {
            0.5
        } else {
            0.1
        }
    }

    /// Calculates monomorphization potential for a type
    fn calculate_type_monomorphization_potential(&self, _type_name: &str, usage_sizes: &[usize]) -> f64 {
        // Types with consistent usage patterns benefit from monomorphization
        let avg_size = usage_sizes.iter().sum::<usize>() as f64 / usage_sizes.len() as f64;
        if avg_size > 30.0 {
            0.7
        } else {
            0.3
        }
    }

    /// Calculates inlining potential for a type
    fn calculate_type_inlining_potential(&self, _type_name: &str, usage_sizes: &[usize]) -> f64 {
        // Types used in small functions are good for inlining
        if usage_sizes.iter().any(|&size| size < 20) {
            0.6
        } else {
            0.2
        }
    }

    /// Calculates streaming potential for a type
    fn calculate_type_streaming_potential(&self, _type_name: &str, usage_sizes: &[usize]) -> f64 {
        // Types with high usage count benefit from streaming
        if usage_sizes.len() > 5 {
            0.7
        } else {
            0.3
        }
    }

    /// Generates optimization suggestions
    fn generate_optimization_suggestions(
        &self,
        function_metrics: &HashMap<String, FunctionMetrics>,
        type_impact: &HashMap<String, TypeImpact>,
    ) -> Result<Vec<OptimizationSuggestion>, AnalysisError> {
        let mut suggestions = Vec::new();

        // Suggestion 1: Thinning for suitable functions
        for (function_name, metrics) in function_metrics {
            if metrics.thinning_suitability > 0.7 {
                suggestions.push(OptimizationSuggestion {
                    suggestion_type: SuggestionType::Thinning,
                    target: function_name.clone(),
                    expected_reduction: (metrics.code_size as f64 * 0.4) as usize, // 40% reduction
                    confidence: metrics.thinning_suitability,
                    description: format!(
                        "Apply thin monomorphization to {} (suitability: {:.2})",
                        function_name, metrics.thinning_suitability
                    ),
                    implementation_complexity: ImplementationComplexity::Moderate,
                });
            }
        }

        // Suggestion 2: Deduplication for similar functions
        let similar_functions = self.find_similar_functions(function_metrics);
        for (func1, func2, similarity) in similar_functions {
            if similarity > 0.8 {
                let metrics1 = function_metrics.get(func1).unwrap();
                let expected_reduction = (metrics1.code_size as f64 * 0.3) as usize;
                suggestions.push(OptimizationSuggestion {
                    suggestion_type: SuggestionType::Deduplication,
                    target: format!("{} and {}", func1, func2),
                    expected_reduction,
                    confidence: similarity,
                    description: format!(
                        "Deduplicate similar functions {} and {} (similarity: {:.2})",
                        func1, func2, similarity
                    ),
                    implementation_complexity: ImplementationComplexity::Easy,
                });
            }
        }

        // Suggestion 3: Type-specific optimizations
        for (type_name, impact) in type_impact {
            if impact.optimization_potential.thinning_potential > 0.7 {
                suggestions.push(OptimizationSuggestion {
                    suggestion_type: SuggestionType::TypeSpecialization,
                    target: type_name.clone(),
                    expected_reduction: (impact.code_size_impact as f64 * 0.25) as usize,
                    confidence: impact.optimization_potential.thinning_potential,
                    description: format!(
                        "Apply type specialization for {} (potential: {:.2})",
                        type_name, impact.optimization_potential.thinning_potential
                    ),
                    implementation_complexity: ImplementationComplexity::Moderate,
                });
            }
        }

        // Suggestion 4: Streaming layout optimization
        let total_size: usize = function_metrics.values().map(|m| m.code_size).sum();
        if total_size > 1000 {
            suggestions.push(OptimizationSuggestion {
                suggestion_type: SuggestionType::StreamingLayout,
                target: "all_functions".to_string(),
                expected_reduction: (total_size as f64 * 0.05) as usize, // 5% reduction
                confidence: 0.8,
                description: "Apply streaming layout optimization for large codebases".to_string(),
                implementation_complexity: ImplementationComplexity::Easy,
            });
        }

        // Sort suggestions by expected reduction
        suggestions.sort_by(|a, b| b.expected_reduction.cmp(&a.expected_reduction));

        Ok(suggestions)
    }

    /// Finds similar functions for deduplication
    fn find_similar_functions(&self, function_metrics: &HashMap<String, FunctionMetrics>) -> Vec<(String, String, f64)> {
        let mut similar_pairs = Vec::new();
        let functions: Vec<_> = function_metrics.keys().cloned().collect();

        for (i, func1) in functions.iter().enumerate() {
            for func2 in functions.iter().skip(i + 1) {
                let similarity = self.calculate_function_similarity(
                    function_metrics.get(func1).unwrap(),
                    function_metrics.get(func2).unwrap(),
                );
                
                if similarity > 0.6 {
                    similar_pairs.push((func1.clone(), func2.clone(), similarity));
                }
            }
        }

        similar_pairs
    }

    /// Calculates similarity between two functions
    fn calculate_function_similarity(&self, metrics1: &FunctionMetrics, metrics2: &FunctionMetrics) -> f64 {
        // Size similarity
        let size_similarity = 1.0 - (metrics1.code_size as f64 - metrics2.code_size as f64).abs() / 
                               (metrics1.code_size + metrics2.code_size) as f64;

        // Instruction count similarity
        let instruction_similarity = 1.0 - (metrics1.instruction_count as f64 - metrics2.instruction_count as f64).abs() / 
                                        (metrics1.instruction_count + metrics2.instruction_count) as f64;

        // Complexity similarity
        let complexity_similarity = 1.0 - (metrics1.generic_complexity - metrics2.generic_complexity).abs() / 
                                      (metrics1.generic_complexity + metrics2.generic_complexity);

        // Type usage similarity
        let type_similarity = self.calculate_type_set_similarity(&metrics1.used_types, &metrics2.used_types);

        (size_similarity * 0.3) + 
        (instruction_similarity * 0.2) + 
        (complexity_similarity * 0.3) + 
        (type_similarity * 0.2)
    }

    /// Calculates similarity between type sets
    fn calculate_type_set_similarity(&self, types1: &[String], types2: &[String]) -> f64 {
        let set1: std::collections::HashSet<_> = types1.iter().cloned().collect();
        let set2: std::collections::HashSet<_> = types2.iter().cloned().collect();
        
        let intersection = set1.intersection(&set2).count();
        let union = set1.union(&set2).count();
        
        if union == 0 {
            1.0
        } else {
            intersection as f64 / union as f64
        }
    }

    /// Estimates optimized code size after applying suggestions
    fn estimate_optimized_size(
        &self,
        function_metrics: &HashMap<String, FunctionMetrics>,
        suggestions: &[OptimizationSuggestion],
    ) -> Result<usize, AnalysisError> {
        let mut total_size: usize = function_metrics.values().map(|m| m.code_size).sum();
        let mut functions_optimized: std::collections::HashSet<String> = std::collections::HashSet::new();

        for suggestion in suggestions {
            if suggestion.suggestion_type == SuggestionType::Thinning {
                if let Some(metrics) = function_metrics.get(&suggestion.target) {
                    if !functions_optimized.contains(&suggestion.target) {
                        total_size -= (metrics.code_size as f64 * suggestion.confidence) as usize;
                        functions_optimized.insert(suggestion.target.clone());
                    }
                }
            } else if suggestion.suggestion_type == SuggestionType::Deduplication {
                // Simplified - split the reduction evenly
                total_size = total_size.saturating_sub(suggestion.expected_reduction);
            } else {
                total_size = total_size.saturating_sub(suggestion.expected_reduction);
            }
        }

        Ok(total_size)
    }

    /// Gets the last analysis report
    pub fn get_last_report(&self) -> Option<&SizeAnalysisReport> {
        self.last_report.as_ref()
    }
}

impl Default for PerformanceProfile {
    fn default() -> Self {
        Self {
            call_frequency: CallFrequency::Unknown,
            memory_pattern: MemoryPattern::Unknown,
            control_flow_complexity: ControlFlowComplexity::Linear,
            hot_path_percentage: 0.0,
        }
    }
}

/// Errors that can occur during size analysis
#[derive(Debug, Clone)]
pub enum AnalysisError {
    /// Invalid instruction for size estimation
    InvalidInstruction(String),
    /// Invalid type for size estimation
    InvalidType(String),
    /// Analysis configuration error
    ConfigurationError(String),
    /// Insufficient data for analysis
    InsufficientData(String),
}

impl std::fmt::Display for AnalysisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnalysisError::InvalidInstruction(msg) => write!(f, "Invalid instruction: {}", msg),
            AnalysisError::InvalidType(msg) => write!(f, "Invalid type: {}", msg),
            AnalysisError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            AnalysisError::InsufficientData(msg) => write!(f, "Insufficient data: {}", msg),
        }
    }
}

impl std::error::Error for AnalysisError {}

#[cfg(test)]
mod tests {
    use super::*;
    use rustc_target::spec::Target;

    #[test]
    fn test_size_analyzer_creation() {
        let target = rustc_target::spec::Target {
            arch: "wasm32".to_string(),
            ..Default::default()
        };
        
        let analyzer = SizeAnalyzer::new(target);
        assert!(analyzer.get_last_report().is_none());
    }

    #[test]
    fn test_function_size_estimation() {
        let target = rustc_target::spec::Target {
            arch: "wasm32".to_string(),
            ..Default::default()
        };
        
        let analyzer = SizeAnalyzer::new(target);
        
        let mut function = WasmIR::new(
            "test_function".to_string(),
            Signature {
                params: vec![Type::I32, Type::I32],
                returns: Some(Type::I32),
            },
        );
        
        function.add_local(Type::I32);
        
        let size = analyzer.estimate_function_size(&function).unwrap();
        assert!(size > 0);
        assert!(size < 100); // Should be reasonable
    }

    #[test]
    fn test_instruction_size_estimation() {
        let target = rustc_target::spec::Target {
            arch: "wasm32".to_string(),
            ..Default::default()
        };
        
        let analyzer = SizeAnalyzer::new(target);
        
        let simple_add = Instruction::BinaryOp {
            op: BinaryOp::Add,
            left: Operand::Local(0),
            right: Operand::Constant(Constant::I32(42)),
        };
        
        let size = analyzer.estimate_instruction_size(&simple_add).unwrap();
        assert_eq!(size, 2 + 1 + 4); // base_size + local + constant
    }

    #[test]
    fn test_type_extraction() {
        let target = rustc_target::spec::Target {
            arch: "wasm32".to_string(),
            ..Default::default()
        };
        
        let analyzer = SizeAnalyzer::new(target);
        
        let mut function = WasmIR::new(
            "test_function".to_string(),
            Signature {
                params: vec![Type::I32, Type::Array { 
                    element_type: Box::new(Type::I32),
                    size: Some(10),
                }],
                returns: Some(Type::I32),
            },
        );
        
        function.add_local(Type::F64);
        
        let used_types = analyzer.extract_used_types(&function);
        assert!(used_types.contains(&"i32".to_string()));
        assert!(used_types.iter().any(|ty| ty.contains("Array")));
        assert!(used_types.contains(&"f64".to_string()));
    }

    #[test]
    fn test_generic_complexity_calculation() {
        let target = rustc_target::spec::Target {
            arch: "wasm32".to_string(),
            ..Default::default()
        };
        
        let analyzer = SizeAnalyzer::new(target);
        
        let simple_function = WasmIR::new(
            "simple".to_string(),
            Signature { params: vec![], returns: None },
        );
        
        let complex_function = WasmIR::new(
            "complex<T>".to_string(),
            Signature {
                params: vec![Type::Ref("T".to_string())],
                returns: Some(Type::Ref("T".to_string())),
            },
        );
        
        let simple_complexity = analyzer.calculate_generic_complexity(&simple_function, &[]);
        let complex_complexity = analyzer.calculate_generic_complexity(&complex_function, &["T".to_string()]);
        
        assert!(complex_complexity > simple_complexity);
    }

    #[test]
    fn test_optimization_suggestion_generation() {
        let target = rustc_target::spec::Target {
            arch: "wasm32".to_string(),
            ..Default::default()
        };
        
        let analyzer = SizeAnalyzer::new(target);
        
        let mut function_metrics = HashMap::new();
        
        // Add a function that's good for thinning
        let good_function = FunctionMetrics {
            name: "good_candidate".to_string(),
            code_size: 100,
            instruction_count: 50,
            basic_block_count: 10,
            generic_complexity: 25.0,
            inlining_potential: 0.3,
            thinning_suitability: 0.8,
            size_contribution: 0.0,
            used_types: vec!["T".to_string()],
            performance_profile: PerformanceProfile::default(),
        };
        
        function_metrics.insert("good_candidate".to_string(), good_function);
        
        let suggestions = analyzer.generate_optimization_suggestions(&function_metrics, &HashMap::new()).unwrap();
        
        assert!(!suggestions.is_empty());
        assert!(suggestions.iter().any(|s| s.suggestion_type == SuggestionType::Thinning));
    }

    #[test]
    fn test_function_similarity() {
        let target = rustc_target::spec::Target {
            arch: "wasm32".to_string(),
            ..Default::default()
        };
        
        let analyzer = SizeAnalyzer::new(target);
        
        let func1 = FunctionMetrics {
            name: "function1".to_string(),
            code_size: 100,
            instruction_count: 25,
            basic_block_count: 5,
            generic_complexity: 10.0,
            inlining_potential: 0.5,
            thinning_suitability: 0.7,
            size_contribution: 0.0,
            used_types: vec!["i32".to_string()],
            performance_profile: PerformanceProfile::default(),
        };
        
        let func2 = FunctionMetrics {
            name: "function2".to_string(),
            code_size: 110,
            instruction_count: 27,
            basic_block_count: 6,
            generic_complexity: 11.0,
            inlining_potential: 0.4,
            thinning_suitability: 0.6,
            size_contribution: 0.0,
            used_types: vec!["i32".to_string()],
            performance_profile: PerformanceProfile::default(),
        };
        
        let similarity = analyzer.calculate_function_similarity(&func1, &func2);
        assert!(similarity > 0.8); // Should be quite similar
    }

    #[test]
    fn test_type_impact_analysis() {
        let target = rustc_target::spec::Target {
            arch: "wasm32".to_string(),
            ..Default::default()
        };
        
        let analyzer = SizeAnalyzer::new(target);
        
        let mut function_metrics = HashMap::new();
        
        let func1 = FunctionMetrics {
            name: "func1".to_string(),
            code_size: 50,
            instruction_count: 12,
            basic_block_count: 3,
            generic_complexity: 5.0,
            inlining_potential: 0.8,
            thinning_suitability: 0.3,
            size_contribution: 0.0,
            used_types: vec!["i32".to_string()],
            performance_profile: PerformanceProfile::default(),
        };
        
        let func2 = FunctionMetrics {
            name: "func2".to_string(),
            code_size: 75,
            instruction_count: 18,
            basic_block_count: 4,
            generic_complexity: 7.5,
            inlining_potential: 0.6,
            thinning_suitability: 0.4,
            size_contribution: 0.0,
            used_types: vec!["i32".to_string()],
            performance_profile: PerformanceProfile::default(),
        };
        
        function_metrics.insert("func1".to_string(), func1);
        function_metrics.insert("func2".to_string(), func2);
        
        let type_impact = analyzer.analyze_type_impact(&function_metrics).unwrap();
        
        assert!(type_impact.contains_key("i32"));
        let impact = type_impact.get("i32").unwrap();
        assert_eq!(impact.usage_count, 2);
        assert_eq!(impact.code_size_impact, 125); // 50 + 75
        assert_eq!(impact.average_function_size, 62.5); // (50 + 75) / 2
    }
}