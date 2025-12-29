//! MIR Complexity Analysis for Thin Monomorphization
//! 
//! This module implements analysis of Rust MIR to identify functions
//! that are good candidates for thin monomorphization based on
//! their complexity, size, and usage patterns.

use rustc_middle::mir::{Body, BasicBlock, Terminator, TerminatorKind, StatementKind};
use rustc_middle::ty::{TyS, TyKind, Instance};
use rustc_target::spec::Target;
use std::collections::{HashMap, HashSet};
use crate::wasmir::WasmIR;

/// Complexity analysis result for a function
#[derive(Debug, Clone)]
pub struct FunctionComplexity {
    /// Function name/identifier
    pub function_name: String,
    /// Instance identifier
    pub instance: Instance,
    /// Number of basic blocks
    pub basic_block_count: usize,
    /// Number of instructions total
    pub instruction_count: usize,
    /// Maximum nesting depth
    pub max_nesting_depth: usize,
    /// Number of generic type usages
    pub generic_type_usages: usize,
    /// Number of calls to other functions
    pub call_count: usize,
    /// Whether function contains loops
    pub has_loops: bool,
    /// Whether function contains recursive calls
    pub is_recursive: bool,
    /// Estimated complexity score (higher = more complex)
    pub complexity_score: f64,
    /// Whether function is a good candidate for thinning
    pub is_thinning_candidate: bool,
    /// Reason for candidacy decision
    pub candidacy_reason: String,
}

/// MIR complexity analyzer
pub struct MirComplexityAnalyzer {
    /// Target architecture
    target: Target,
    /// Threshold for basic block count (default: 20)
    basic_block_threshold: usize,
    /// Threshold for instruction count (default: 50)
    instruction_threshold: usize,
    /// Weight factors for complexity calculation
    weights: ComplexityWeights,
}

/// Weight factors for complexity calculation
#[derive(Debug, Clone)]
pub struct ComplexityWeights {
    /// Weight for each basic block
    pub basic_block_weight: f64,
    /// Weight for each instruction
    pub instruction_weight: f64,
    /// Weight for nesting depth
    pub nesting_depth_weight: f64,
    /// Weight for generic type usage
    pub generic_type_weight: f64,
    /// Weight for function calls
    pub call_weight: f64,
    /// Weight for loops
    pub loop_weight: f64,
    /// Weight for recursion
    pub recursion_weight: f64,
}

impl Default for ComplexityWeights {
    fn default() -> Self {
        Self {
            basic_block_weight: 1.0,
            instruction_weight: 0.1,
            nesting_depth_weight: 2.0,
            generic_type_weight: 3.0,
            call_weight: 1.5,
            loop_weight: 5.0,
            recursion_weight: 10.0,
        }
    }
}

impl MirComplexityAnalyzer {
    /// Creates a new MIR complexity analyzer
    pub fn new(target: Target) -> Self {
        Self {
            target,
            basic_block_threshold: 20,
            instruction_threshold: 50,
            weights: ComplexityWeights::default(),
        }
    }

    /// Sets custom thresholds
    pub fn with_thresholds(mut self, basic_block_threshold: usize, instruction_threshold: usize) -> Self {
        self.basic_block_threshold = basic_block_threshold;
        self.instruction_threshold = instruction_threshold;
        self
    }

    /// Sets custom weight factors
    pub fn with_weights(mut self, weights: ComplexityWeights) -> Self {
        self.weights = weights;
        self
    }

    /// Analyzes a MIR body for complexity
    pub fn analyze_function(
        &self,
        body: &Body<'_>,
        instance: Instance,
        function_name: &str,
    ) -> FunctionComplexity {
        let mut analysis = FunctionComplexity {
            function_name: function_name.to_string(),
            instance,
            basic_block_count: body.basic_blocks().len(),
            instruction_count: 0,
            max_nesting_depth: 0,
            generic_type_usages: 0,
            call_count: 0,
            has_loops: false,
            is_recursive: false,
            complexity_score: 0.0,
            is_thinning_candidate: false,
            candidacy_reason: String::new(),
        };

        // Analyze each basic block
        let mut current_nesting_depth = 0;
        let mut loop_headers: HashSet<_> = HashSet::new();

        for (bb_id, basic_block) in body.basic_blocks().iter_enumerated() {
            // Count instructions
            analysis.instruction_count += basic_block.statements.len();
            if let Some(_) = &basic_block.terminator {
                analysis.instruction_count += 1;
            }

            // Analyze statements
            for statement in &basic_block.statements {
                self.analyze_statement(statement, &mut analysis, &mut current_nesting_depth);
            }

            // Analyze terminator
            if let Some(terminator) = &basic_block.terminator {
                self.analyze_terminator(
                    terminator,
                    &mut analysis,
                    bb_id,
                    &mut loop_headers,
                    &mut current_nesting_depth,
                );
            }
        }

        // Calculate complexity score
        analysis.complexity_score = self.calculate_complexity_score(&analysis);

        // Determine candidacy
        self.determine_thinning_candidacy(&mut analysis);

        analysis
    }

    /// Analyzes a MIR statement
    fn analyze_statement(
        &self,
        statement: &StatementKind<'_>,
        analysis: &mut FunctionComplexity,
        nesting_depth: &mut usize,
    ) {
        match statement {
            StatementKind::Assign(box (place, rvalue)) => {
                // Analyze place for generic types
                self.analyze_place(place, analysis);
                
                // Analyze rvalue
                self.analyze_rvalue(rvalue, analysis, nesting_depth);
            }
            StatementKind::StorageLive(_) | StatementKind::StorageDead(_) => {
                // Storage statements don't contribute significantly to complexity
            }
            StatementKind::Nop => {
                // No-op statements
            }
            StatementKind::FakeRead(_) => {
                // Fake reads for borrow checking
            }
            StatementKind::Retag(_, _) => {
                // Retagging for Stack Borrows
            }
            StatementKind::AscribeUserType(_, _) => {
                // User type ascription
            }
            StatementKind::Coverage(_) => {
                // Coverage instrumentation
            }
            StatementKind::Intrinsic(box intrinsic) => {
                // Intrinsic calls
                analysis.call_count += 1;
                self.analyze_intrinsic(intrinsic, analysis);
            }
        }
    }

    /// Analyzes a MIR terminator
    fn analyze_terminator(
        &self,
        terminator: &Terminator<'_>,
        analysis: &mut FunctionComplexity,
        bb_id: rustc_middle::mir::BasicBlock,
        loop_headers: &mut HashSet<rustc_middle::mir::BasicBlock>,
        nesting_depth: &mut usize,
    ) {
        match &terminator.kind {
            TerminatorKind::Return => {
                // Function return
            }
            TerminatorKind::Goto { .. } => {
                // Unconditional branch
            }
            TerminatorKind::SwitchInt { .. } => {
                // Switch statement - increases complexity
                *nesting_depth += 1;
            }
            TerminatorKind::Call { func, .. } => {
                // Function call
                analysis.call_count += 1;
                self.analyze_operand(func, analysis);
            }
            TerminatorKind::Assert { .. } => {
                // Assertion
            }
            TerminatorKind::Drop { place, .. } => {
                // Drop statement
                self.analyze_place(place, analysis);
            }
            TerminatorKind::Yield { .. } => {
                // Generator yield
            }
            TerminatorKind::GeneratorDrop => {
                // Generator cleanup
            }
            TerminatorKind::Resume => {
                // Generator resume
            }
            TerminatorKind::Abort => {
                // Program abort
            }
            TerminatorKind::FalseEdge { .. } => {
                // False edge for borrow checking
            }
            TerminatorKind::FalseUnwind { .. } => {
                // False unwind for borrow checking
            }

            // Loop detection
            TerminatorKind::Goto { target } => {
                if *target <= bb_id {
                    // This is a back edge - indicates a loop
                    analysis.has_loops = true;
                    loop_headers.insert(bb_id);
                }
            }
        }
    }

    /// Analyzes an MIR place
    fn analyze_place(&self, place: &rustc_middle::mir::Place<'_>, analysis: &mut FunctionComplexity) {
        // Check for generic type usage in place
        self.analyze_ty(place.ty(), analysis);
    }

    /// Analyzes an MIR rvalue
    fn analyze_rvalue(
        &self,
        rvalue: &rustc_middle::mir::Rvalue<'_>,
        analysis: &mut FunctionComplexity,
        nesting_depth: &mut usize,
    ) {
        use rustc_middle::mir::Rvalue::*;
        
        match rvalue {
            Use(operand) | Repeat(operand, _) => {
                self.analyze_operand(operand, analysis);
            }
            Ref(_, _, _) => {
                // Reference creation
            }
            ThreadLocalRef(_) => {
                // Thread local access
            }
            AddressOf(_) => {
                // Taking address
            }
            Len(_) | Discriminant(_) | Index(_, _) | Field(_, _) => {
                // Projective operations
            }
            Cast(kind, operand, _) => {
                self.analyze_operand(operand, analysis);
                match kind {
                    rustc_middle::mir::CastKind::IntToInt |
                    rustc_middle::mir::CastKind::IntToFloat => {
                        // Simple casts
                    }
                    _ => {
                        // Complex casts increase complexity
                        analysis.complexity_score += self.weights.call_weight;
                    }
                }
            }
            BinaryOp(_, left, right) => {
                self.analyze_operand(left, analysis);
                self.analyze_operand(right, analysis);
            }
            CheckedBinaryOp(_, left, right) => {
                self.analyze_operand(left, analysis);
                self.analyze_operand(right, analysis);
                analysis.complexity_score += self.weights.call_weight; // Checked ops are more complex
            }
            UnaryOp(_, operand) => {
                self.analyze_operand(operand, analysis);
            }
            Aggregate(aggregate_kind, operands) => {
                for operand in operands {
                    self.analyze_operand(operand, analysis);
                }
                match aggregate_kind {
                    rustc_middle::mir::AggregateKind::Adt(_, _, _) => {
                        // ADT construction can be complex
                        analysis.complexity_score += self.weights.generic_type_weight;
                    }
                    _ => {}
                }
            }
            ShallowInitBox(_, _) => {
                // Box allocation
                analysis.call_count += 1;
            }
            NullaryOp(null_op) => {
                match null_op {
                    rustc_middle::mir::NullOp::SizeOf | 
                    rustc_middle::mir::NullOp::AlignOf => {
                        // Size/align operations
                    }
                    rustc_middle::mir::NullOp::BoxMetadata => {
                        // Box metadata
                        analysis.complexity_score += self.weights.call_weight;
                    }
                }
            }
        }
    }

    /// Analyzes an MIR operand
    fn analyze_operand(&self, operand: &rustc_middle::mir::Operand<'_>, analysis: &mut FunctionComplexity) {
        use rustc_middle::mir::Operand::*;
        
        match operand {
            Copy(place) | Move(place) => {
                self.analyze_place(place, analysis);
            }
            Constant(constant) => {
                self.analyze_constant(constant, analysis);
            }
        }
    }

    /// Analyzes a MIR constant
    fn analyze_constant(&self, constant: &rustc_middle::mir::Const<'_>, analysis: &mut FunctionComplexity) {
        match constant.literal() {
            rustc_middle::mir::ConstLiteral::Value { .. } => {
                // Simple literal values
            }
            rustc_middle::mir::ConstLiteral::Ty(ty, _) => {
                // Typed constants
                self.analyze_ty(ty, analysis);
            }
            rustc_middle::mir::ConstLiteral::Unevaluated(_, _, _) => {
                // Unevaluated constants
                analysis.complexity_score += self.weights.call_weight;
            }
        }
    }

    /// Analyzes an intrinsic
    fn analyze_intrinsic(&self, intrinsic: &rustc_middle::mir::Intrinsic<'_>, analysis: &mut FunctionComplexity) {
        match intrinsic {
            rustc_middle::mir::Intrinsic::Assume |
            rustc_middle::mir::Intrinsic::CopyNonOverlapping |
            rustc_middle::mir::Intrinsic::CopyWithin |
            rustc_middle::mir::Intrinsic::DiscriminantValue => {
                // Simple intrinsics
            }
            rustc_middle::mir::Intrinsic::Transmute |
            rustc_middle::mir::Intrinsic::WrappingAdd |
            rustc_middle::mir::Intrinsic::WrappingSub |
            rustc_middle::mir::Intrinsic::WrappingMul => {
                // Complex intrinsics
                analysis.complexity_score += self.weights.call_weight;
            }
            _ => {
                // Unknown or complex intrinsics
                analysis.complexity_score += self.weights.call_weight * 2.0;
            }
        }
    }

    /// Analyzes a type for generic usage
    fn analyze_ty(&self, ty: &TyS<'_>, analysis: &mut FunctionComplexity) {
        match ty.kind() {
            TyKind::Bool |
            TyKind::Int(_) |
            TyKind::Uint(_) |
            TyKind::Float(_) => {
                // Primitive types - not generic
            }
            TyKind::Adt(_, substs) => {
                if !substs.is_empty() {
                    // Generic ADT
                    analysis.generic_type_usages += 1;
                    analysis.complexity_score += self.weights.generic_type_weight;
                }
            }
            TyKind::Array(inner_ty, _) | TyKind::Slice(inner_ty) => {
                self.analyze_ty(inner_ty, analysis);
            }
            TyKind::Ref(_, inner_ty, _) => {
                self.analyze_ty(inner_ty, analysis);
            }
            TyKind::Tuple(tys) => {
                for ty in tys {
                    self.analyze_ty(ty, analysis);
                }
            }
            TyKind::Param(_) => {
                // Type parameter - this is a generic function
                analysis.generic_type_usages += 1;
                analysis.complexity_score += self.weights.generic_type_weight * 2.0;
            }
            _ => {
                // Other types
                analysis.generic_type_usages += 1;
            }
        }
    }

    /// Calculates overall complexity score
    fn calculate_complexity_score(&self, analysis: &FunctionComplexity) -> f64 {
        let mut score = 0.0;
        
        // Basic block contribution
        score += analysis.basic_block_count as f64 * self.weights.basic_block_weight;
        
        // Instruction contribution
        score += analysis.instruction_count as f64 * self.weights.instruction_weight;
        
        // Nesting depth contribution
        score += analysis.max_nesting_depth as f64 * self.weights.nesting_depth_weight;
        
        // Generic type usage contribution
        score += analysis.generic_type_usages as f64 * self.weights.generic_type_weight;
        
        // Function call contribution
        score += analysis.call_count as f64 * self.weights.call_weight;
        
        // Loop contribution
        if analysis.has_loops {
            score += self.weights.loop_weight;
        }
        
        // Recursion contribution
        if analysis.is_recursive {
            score += self.weights.recursion_weight;
        }
        
        score
    }

    /// Determines if a function is a good candidate for thinning
    fn determine_thinning_candidacy(&self, analysis: &mut FunctionComplexity) {
        // Basic thresholds
        let meets_size_threshold = analysis.basic_block_count >= self.basic_block_threshold ||
                                analysis.instruction_count >= self.instruction_threshold;
        
        // Has sufficient generic complexity
        let has_generic_complexity = analysis.generic_type_usages > 0 &&
                                  analysis.generic_type_usages * 2 < analysis.instruction_count;
        
        // Not too simple (small functions should stay monomorphized for inlining)
        let not_too_simple = analysis.basic_block_count > 3 && analysis.instruction_count > 10;
        
        // Not too complex (very complex functions might benefit from full specialization)
        let not_too_complex = analysis.basic_block_count < 100 && analysis.complexity_score < 1000.0;
        
        // High complexity score indicates good candidate
        let high_complexity = analysis.complexity_score > 10.0;
        
        // Make decision
        analysis.is_thinning_candidate = meets_size_threshold &&
                                      has_generic_complexity &&
                                      not_too_simple &&
                                      not_too_complex &&
                                      high_complexity;
        
        // Generate reason
        if analysis.is_thinning_candidate {
            let mut reasons = Vec::new();
            
            if meets_size_threshold {
                reasons.push(format!("sufficient size ({} blocks, {} instructions)", 
                    analysis.basic_block_count, analysis.instruction_count));
            }
            
            if has_generic_complexity {
                reasons.push(format!("good generic usage ratio ({} generic uses)", 
                    analysis.generic_type_usages));
            }
            
            if high_complexity {
                reasons.push(format!("high complexity score ({:.1})", 
                    analysis.complexity_score));
            }
            
            analysis.candidacy_reason = reasons.join(", ");
        } else {
            let mut reasons = Vec::new();
            
            if !meets_size_threshold {
                reasons.push("too small for thinning".to_string());
            }
            
            if !has_generic_complexity {
                reasons.push("insufficient generic usage".to_string());
            }
            
            if !not_too_simple {
                reasons.push("too simple (better for inlining)".to_string());
            }
            
            if !not_too_complex {
                reasons.push("too complex (needs full specialization)".to_string());
            }
            
            analysis.candidacy_reason = if reasons.is_empty() {
                "not a good candidate".to_string()
            } else {
                reasons.join(", ")
            };
        }
    }
}

/// Function similarity analyzer for identifying function shapes
pub struct FunctionSimilarityAnalyzer {
    /// Threshold for similarity (0.0 to 1.0)
    similarity_threshold: f64,
}

impl FunctionSimilarityAnalyzer {
    /// Creates a new function similarity analyzer
    pub fn new(similarity_threshold: f64) -> Self {
        Self {
            similarity_threshold: similarity_threshold.clamp(0.0, 1.0),
        }
    }

    /// Finds similar functions based on their analysis
    pub fn find_similar_functions(
        &self,
        functions: &[FunctionComplexity],
    ) -> Vec<(usize, usize, f64)> {
        let mut similar_pairs = Vec::new();
        
        for (i, func1) in functions.iter().enumerate() {
            for (j, func2) in functions.iter().enumerate().skip(i + 1) {
                let similarity = self.calculate_similarity(func1, func2);
                if similarity >= self.similarity_threshold {
                    similar_pairs.push((i, j, similarity));
                }
            }
        }
        
        // Sort by similarity descending
        similar_pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        similar_pairs
    }

    /// Calculates similarity between two functions
    fn calculate_similarity(&self, func1: &FunctionComplexity, func2: &FunctionComplexity) -> f64 {
        // Size similarity (40% weight)
        let size_similarity = self.calculate_size_similarity(func1, func2);
        
        // Complexity similarity (30% weight)
        let complexity_similarity = self.calculate_complexity_similarity(func1, func2);
        
        // Structure similarity (20% weight)
        let structure_similarity = self.calculate_structure_similarity(func1, func2);
        
        // Control flow similarity (10% weight)
        let control_flow_similarity = self.calculate_control_flow_similarity(func1, func2);
        
        size_similarity * 0.4 + 
        complexity_similarity * 0.3 + 
        structure_similarity * 0.2 + 
        control_flow_similarity * 0.1
    }

    /// Calculates size similarity
    fn calculate_size_similarity(&self, func1: &FunctionComplexity, func2: &FunctionComplexity) -> f64 {
        let bb_similarity = 1.0 - (func1.basic_block_count as f64 - func2.basic_block_count as f64).abs() / 
                           (func1.basic_block_count + func2.basic_block_count) as f64;
        
        let inst_similarity = 1.0 - (func1.instruction_count as f64 - func2.instruction_count as f64).abs() / 
                            (func1.instruction_count + func2.instruction_count) as f64;
        
        (bb_similarity + inst_similarity) / 2.0
    }

    /// Calculates complexity similarity
    fn calculate_complexity_similarity(&self, func1: &FunctionComplexity, func2: &FunctionComplexity) -> f64 {
        1.0 - (func1.complexity_score - func2.complexity_score).abs() / 
              (func1.complexity_score + func2.complexity_score)
    }

    /// Calculates structure similarity
    fn calculate_structure_similarity(&self, func1: &FunctionComplexity, func2: &FunctionComplexity) -> f64 {
        let generic_similarity = if func1.generic_type_usages > 0 && func2.generic_type_usages > 0 {
            let min_generic = func1.generic_type_usages.min(func2.generic_type_usages);
            let max_generic = func1.generic_type_usages.max(func2.generic_type_usages);
            min_generic as f64 / max_generic as f64
        } else if func1.generic_type_usages == 0 && func2.generic_type_usages == 0 {
            1.0
        } else {
            0.0
        };
        
        let call_similarity = if func1.call_count > 0 && func2.call_count > 0 {
            let min_calls = func1.call_count.min(func2.call_count);
            let max_calls = func1.call_count.max(func2.call_count);
            min_calls as f64 / max_calls as f64
        } else if func1.call_count == 0 && func2.call_count == 0 {
            1.0
        } else {
            0.0
        };
        
        (generic_similarity + call_similarity) / 2.0
    }

    /// Calculates control flow similarity
    fn calculate_control_flow_similarity(&self, func1: &FunctionComplexity, func2: &FunctionComplexity) -> f64 {
        let loops_match = func1.has_loops == func2.has_loops;
        let recursion_match = func1.is_recursive == func2.is_recursive;
        
        if loops_match && recursion_match {
            1.0
        } else if loops_match || recursion_match {
            0.5
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustc_middle::mir::{Body, BasicBlockData, TerminatorKind};
    use rustc_middle::ty::TyS;
    use rustc_target::spec::Target;

    #[test]
    fn test_complexity_analyzer_creation() {
        let target = rustc_target::spec::Target {
            arch: "wasm32".to_string(),
            ..Default::default()
        };
        
        let analyzer = MirComplexityAnalyzer::new(target);
        assert_eq!(analyzer.basic_block_threshold, 20);
        assert_eq!(analyzer.instruction_threshold, 50);
    }

    #[test]
    fn test_custom_thresholds() {
        let target = rustc_target::spec::Target {
            arch: "wasm32".to_string(),
            ..Default::default()
        };
        
        let analyzer = MirComplexityAnalyzer::new(target)
            .with_thresholds(10, 25);
        
        assert_eq!(analyzer.basic_block_threshold, 10);
        assert_eq!(analyzer.instruction_threshold, 25);
    }

    #[test]
    fn test_complexity_weights() {
        let weights = ComplexityWeights::default();
        assert_eq!(weights.basic_block_weight, 1.0);
        assert_eq!(weights.instruction_weight, 0.1);
        assert_eq!(weights.nesting_depth_weight, 2.0);
        assert_eq!(weights.generic_type_weight, 3.0);
        assert_eq!(weights.call_weight, 1.5);
        assert_eq!(weights.loop_weight, 5.0);
        assert_eq!(weights.recursion_weight, 10.0);
    }

    #[test]
    fn test_thinning_candidacy_simple_function() {
        let target = rustc_target::spec::Target {
            arch: "wasm32".to_string(),
            ..Default::default()
        };
        
        let analyzer = MirComplexityAnalyzer::new(target);
        
        // Create a simple function that should NOT be a candidate
        let mut simple_analysis = FunctionComplexity {
            function_name: "simple_function".to_string(),
            instance: /* dummy instance */,
            basic_block_count: 2,
            instruction_count: 5,
            max_nesting_depth: 0,
            generic_type_usages: 0,
            call_count: 0,
            has_loops: false,
            is_recursive: false,
            complexity_score: 0.0,
            is_thinning_candidate: false,
            candidacy_reason: String::new(),
        };
        
        analyzer.determine_thinning_candidacy(&mut simple_analysis);
        
        assert!(!simple_analysis.is_thinning_candidate);
        assert!(simple_analysis.candidacy_reason.contains("too small") || 
                 simple_analysis.candidacy_reason.contains("simple"));
    }

    #[test]
    fn test_thinning_candidacy_complex_function() {
        let target = rustc_target::spec::Target {
            arch: "wasm32".to_string(),
            ..Default::default()
        };
        
        let analyzer = MirComplexityAnalyzer::new(target);
        
        // Create a complex function that SHOULD be a candidate
        let mut complex_analysis = FunctionComplexity {
            function_name: "complex_generic_function".to_string(),
            instance: /* dummy instance */,
            basic_block_count: 25,
            instruction_count: 80,
            max_nesting_depth: 3,
            generic_type_usages: 15,
            call_count: 8,
            has_loops: true,
            is_recursive: false,
            complexity_score: 0.0,
            is_thinning_candidate: false,
            candidacy_reason: String::new(),
        };
        
        analyzer.determine_thinning_candidacy(&mut complex_analysis);
        
        assert!(complex_analysis.is_thinning_candidate);
        assert!(complex_analysis.candidacy_reason.contains("sufficient size"));
        assert!(complex_analysis.candidacy_reason.contains("generic usage"));
    }

    #[test]
    fn test_function_similarity() {
        let analyzer = FunctionSimilarityAnalyzer::new(0.7);
        
        let func1 = FunctionComplexity {
            function_name: "function1".to_string(),
            instance: /* dummy */,
            basic_block_count: 20,
            instruction_count: 60,
            max_nesting_depth: 2,
            generic_type_usages: 10,
            call_count: 5,
            has_loops: true,
            is_recursive: false,
            complexity_score: 50.0,
            is_thinning_candidate: true,
            candidacy_reason: "good candidate".to_string(),
        };
        
        let func2 = FunctionComplexity {
            function_name: "function2".to_string(),
            instance: /* dummy */,
            basic_block_count: 22,
            instruction_count: 65,
            max_nesting_depth: 2,
            generic_type_usages: 12,
            call_count: 6,
            has_loops: true,
            is_recursive: false,
            complexity_score: 55.0,
            is_thinning_candidate: true,
            candidacy_reason: "good candidate".to_string(),
        };
        
        let similarity = analyzer.calculate_similarity(&func1, &func2);
        assert!(similarity > 0.7); // Should be similar
        
        let similar_pairs = analyzer.find_similar_functions(&[func1, func2]);
        assert_eq!(similar_pairs.len(), 1);
        assert_eq!(similar_pairs[0].2, similarity); // The similarity score
    }

    #[test]
    fn test_similarity_analyzer() {
        let analyzer = FunctionSimilarityAnalyzer::new(0.8);
        
        let func1 = FunctionComplexity {
            function_name: "func1".to_string(),
            instance: /* dummy */,
            basic_block_count: 10,
            instruction_count: 30,
            max_nesting_depth: 1,
            generic_type_usages: 5,
            call_count: 2,
            has_loops: false,
            is_recursive: false,
            complexity_score: 25.0,
            is_thinning_candidate: false,
            candidacy_reason: "too simple".to_string(),
        };
        
        let func2 = FunctionComplexity {
            function_name: "func2".to_string(),
            instance: /* dummy */,
            basic_block_count: 50,
            instruction_count: 150,
            max_nesting_depth: 5,
            generic_type_usages: 25,
            call_count: 20,
            has_loops: true,
            is_recursive: true,
            complexity_score: 200.0,
            is_thinning_candidate: false,
            candidacy_reason: "too complex".to_string(),
        };
        
        let similar_pairs = analyzer.find_similar_functions(&[func1, func2]);
        
        // Should not be similar with high threshold
        assert_eq!(similar_pairs.len(), 0);
        
        // Should be similar with low threshold
        let low_threshold_analyzer = FunctionSimilarityAnalyzer::new(0.3);
        let low_threshold_pairs = low_threshold_analyzer.find_similar_functions(&[func1, func2]);
        assert_eq!(low_threshold_pairs.len(), 0); // Still too different
    }
}