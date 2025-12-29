//! Symbol Similarity Checker for Thin Monomorphization
//! 
//! This module implements comprehensive symbol similarity checking using
//! wasmparser and binary analysis tools to verify that thinned functions
//! correctly share common code paths.

use std::collections::{HashMap, HashSet};
use std::process::{Command, Stdio};
use std::io::{self, Read, BufRead, Write};
use serde::{Deserialize, Serialize};
use anyhow::{Result, Context};

/// Symbol analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolAnalysis {
    /// Function name
    pub name: String,
    /// Symbol address in binary
    pub address: u64,
    /// Symbol size in bytes
    pub size: usize,
    /// Function hash (for similarity comparison)
    pub hash: u64,
    /// Instruction sequences
    pub instructions: Vec<InstructionInfo>,
    /// Basic block information
    pub basic_blocks: Vec<BasicBlockInfo>,
    /// Call graph information
    pub calls: Vec<CallInfo>,
}

/// Individual instruction information
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstructionInfo {
    /// Opcode as bytes
    pub opcode: Vec<u8>,
    /// Operands as bytes
    pub operands: Vec<u8>,
    /// Instruction address
    pub address: u64,
    /// Instruction length
    pub length: usize,
}

/// Basic block information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicBlockInfo {
    /// Block start address
    pub start_address: u64,
    /// Block end address
    pub end_address: usize,
    /// Block size in bytes
    pub size: usize,
    /// Number of instructions
    pub instruction_count: usize,
    /// Successor blocks
    pub successors: Vec<u64>,
    /// Predecessor blocks
    pub predecessors: Vec<u64>,
}

/// Function call information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallInfo {
    /// Call instruction address
    pub call_address: u64,
    /// Target function name (if direct call)
    pub target_function: Option<String>,
    /// Call type (direct, indirect, tail call)
    pub call_type: CallType,
}

/// Types of function calls
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CallType {
    /// Direct function call
    Direct,
    /// Indirect function call through register or table
    Indirect,
    /// Tail call (optimization)
    TailCall,
}

/// Similarity analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityAnalysis {
    /// Original functions
    pub original_functions: Vec<SymbolAnalysis>,
    /// Thinned functions
    pub thinned_functions: Vec<SymbolAnalysis>,
    /// Shim functions
    pub shim_functions: Vec<SymbolAnalysis>,
    /// Similarity scores between functions
    pub similarity_scores: Vec<FunctionSimilarity>,
    /// Shared code regions
    pub shared_regions: Vec<SharedRegion>,
    /// Overall similarity metrics
    pub metrics: SimilarityMetrics,
}

/// Function similarity information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionSimilarity {
    /// First function name
    pub func1: String,
    /// Second function name
    pub func2: String,
    /// Similarity score (0.0 to 1.0)
    pub similarity_score: f64,
    /// Shared instruction count
    pub shared_instructions: usize,
    /// Total instruction count in larger function
    pub total_instructions: usize,
    /// Similarity type
    pub similarity_type: SimilarityType,
}

/// Types of similarity
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimilarityType {
    /// Identical functions
    Identical,
    /// Functions with common core and small differences
    ThinnedShared,
    /// Functions with similar patterns
    PatternSimilar,
    /// Unrelated functions
    Different,
}

/// Shared code region between functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedRegion {
    /// Region hash
    pub hash: u64,
    /// Region size in bytes
    pub size: usize,
    /// Functions that contain this region
    pub functions: Vec<String>,
    /// Region start addresses in each function
    pub addresses: HashMap<String, u64>,
}

/// Overall similarity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityMetrics {
    /// Total number of functions analyzed
    pub total_functions: usize,
    /// Number of thinned functions
    pub thinned_functions: usize,
    /// Number of shim functions
    pub shim_functions: usize,
    /// Average similarity between thinned and original functions
    pub average_similarity: f64,
    /// Total shared code size
    pub shared_code_size: usize,
    /// Code sharing efficiency
    pub sharing_efficiency: f64,
    /// Expected size reduction
    pub expected_size_reduction: f64,
}

/// Symbol similarity checker configuration
#[derive(Debug, Clone)]
pub struct SymbolCheckerConfig {
    /// Path to wasm-objdump
    pub wasm_objdump_path: String,
    /// Path to wasm-opt (if available)
    pub wasm_opt_path: Option<String>,
    /// Minimum similarity threshold for reporting
    pub similarity_threshold: f64,
    /// Enable instruction-level analysis
    pub enable_instruction_analysis: bool,
    /// Enable control flow analysis
    pub enable_control_flow_analysis: bool,
    /// Enable call graph analysis
    pub enable_call_graph_analysis: bool,
    /// Output detailed analysis
    pub verbose: bool,
}

impl Default for SymbolCheckerConfig {
    fn default() -> Self {
        Self {
            wasm_objdump_path: "wasm-objdump".to_string(),
            wasm_opt_path: None,
            similarity_threshold: 0.7,
            enable_instruction_analysis: true,
            enable_control_flow_analysis: true,
            enable_call_graph_analysis: true,
            verbose: false,
        }
    }
}

/// Main symbol similarity checker
pub struct SymbolSimilarityChecker {
    config: SymbolCheckerConfig,
}

impl SymbolSimilarityChecker {
    /// Creates a new symbol similarity checker
    pub fn new(config: SymbolCheckerConfig) -> Self {
        Self { config }
    }

    /// Analyzes Wasm binaries for symbol similarity
    pub fn analyze_similarity(
        &self,
        original_binary: &str,
        optimized_binary: &str,
    ) -> Result<SimilarityAnalysis> {
        // Phase 1: Extract symbols from both binaries
        let original_symbols = self.extract_symbols(original_binary)
            .context("Failed to extract symbols from original binary")?;
        
        let optimized_symbols = self.extract_symbols(optimized_binary)
            .context("Failed to extract symbols from optimized binary")?;

        // Phase 2: Categorize functions
        let (thinned_functions, shim_functions, remaining_functions) = 
            self.categorize_functions(&optimized_symbols, &original_symbols)?;

        // Phase 3: Analyze similarity
        let similarity_scores = self.analyze_function_similarity(
            &original_symbols,
            &thinned_functions,
            &shim_functions,
        )?;

        // Phase 4: Identify shared code regions
        let shared_regions = self.identify_shared_regions(
            &original_symbols,
            &thinned_functions,
            &shim_functions,
        )?;

        // Phase 5: Calculate overall metrics
        let metrics = self.calculate_metrics(
            &original_symbols,
            &thinned_functions,
            &shim_functions,
            &shared_regions,
        )?;

        Ok(SimilarityAnalysis {
            original_functions: original_symbols,
            thinned_functions,
            shim_functions,
            similarity_scores,
            shared_regions,
            metrics,
        })
    }

    /// Extracts symbols from a Wasm binary
    fn extract_symbols(&self, binary_path: &str) -> Result<Vec<SymbolAnalysis>> {
        let mut symbols = Vec::new();

        // Use wasm-objdump to disassemble the binary
        let output = Command::new(&self.config.wasm_objdump_path)
            .args(&["-d", "-r", binary_path])
            .output()
            .context("Failed to run wasm-objdump")?;

        if !output.status.success() {
            anyhow::bail!("wasm-objdump failed: {}", String::from_utf8_lossy(&output.stderr));
        }

        let output_str = String::from_utf8(output.stdout)?;
        
        // Parse the disassembly output
        let mut current_function: Option<SymbolAnalysis> = None;
        
        for line in output_str.lines() {
            if let Some(symbol) = self.parse_symbol_line(line)? {
                if let Some(mut current) = current_function {
                    // Finalize previous function
                    current.hash = self.calculate_function_hash(&current);
                    symbols.push(current);
                }
                current_function = Some(symbol);
            } else if let Some(ref mut current) = current_function {
                if let Some(instruction) = self.parse_instruction_line(line)? {
                    current.instructions.push(instruction);
                }
            }
        }

        // Add the last function
        if let Some(mut current) = current_function {
            current.hash = self.calculate_function_hash(&current);
            symbols.push(current);
        }

        // Parse control flow and call information
        for symbol in &mut symbols {
            self.analyze_control_flow(symbol)?;
            self.analyze_calls(symbol)?;
        }

        Ok(symbols)
    }

    /// Parses a symbol definition line
    fn parse_symbol_line(&self, line: &str) -> Result<Option<SymbolAnalysis>> {
        // Expected format: "<function_name>:" or "00000123 <function_name>:"
        let trimmed = line.trim();
        
        if !trimmed.ends_with(':') {
            return Ok(None);
        }

        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() < 1 {
            return Ok(None);
        }

        let name_part = parts.last().unwrap();
        let name = name_part.trim_end_matches(':');

        // Extract address if present
        let address = if parts.len() >= 2 {
            parts[0].parse::<u64>()
                .with_context(|| format!("Failed to parse address: {}", parts[0]))?
        } else {
            0
        };

        Ok(Some(SymbolAnalysis {
            name: name.to_string(),
            address,
            size: 0, // Will be calculated later
            hash: 0, // Will be calculated later
            instructions: Vec::new(),
            basic_blocks: Vec::new(),
            calls: Vec::new(),
        }))
    }

    /// Parses an instruction line
    fn parse_instruction_line(&self, line: &str) -> Result<Option<InstructionInfo>> {
        // Expected format: "12345678: opcode operands..."
        let trimmed = line.trim();
        
        if !trimmed.contains(':') {
            return Ok(None);
        }

        let parts: Vec<&str> = trimmed.splitn(2, ':').collect();
        if parts.len() != 2 {
            return Ok(None);
        }

        let address = parts[0].parse::<u64>()
            .with_context(|| format!("Failed to parse instruction address: {}", parts[0]))?;
        
        let instruction_part = parts[1].trim();
        if instruction_part.is_empty() {
            return Ok(None);
        }

        let instruction_parts: Vec<&str> = instruction_part.split_whitespace().collect();
        if instruction_parts.is_empty() {
            return Ok(None);
        }

        // Parse opcode and operands (simplified)
        let opcode = instruction_parts[0].as_bytes().to_vec();
        let mut operands = Vec::new();
        
        for part in instruction_parts.iter().skip(1) {
            operands.extend_from_slice(part.as_bytes());
            operands.push(b' '); // separator
        }
        
        let length = opcode.len() + operands.len();

        Ok(Some(InstructionInfo {
            opcode,
            operands,
            address,
            length,
        }))
    }

    /// Analyzes control flow for a function
    fn analyze_control_flow(&self, symbol: &mut SymbolAnalysis) -> Result<()> {
        if !self.config.enable_control_flow_analysis {
            return Ok(());
        }

        // Build basic blocks from instructions
        let mut basic_blocks = Vec::new();
        let mut current_block_start = 0;
        let mut block_instructions = Vec::new();
        let mut block_addresses = Vec::new();

        // Find branch targets and block boundaries
        let mut branch_targets = HashSet::new();
        let mut block_boundaries = HashSet::new();

        for (i, instruction) in symbol.instructions.iter().enumerate() {
            block_instructions.push(i);
            block_addresses.push(instruction.address);

            // Check if instruction is a branch
            if self.is_branch_instruction(&instruction.opcode) {
                branch_targets.insert(self.get_branch_target(instruction)?);
                block_boundaries.insert(i + 1);
            }
        }

        // Create basic blocks
        for (i, &instruction_idx) in block_instructions.iter().enumerate() {
            if i == 0 || block_boundaries.contains(&instruction_idx) {
                // Start new block
                if i > 0 {
                    // Complete previous block
                    let start_addr = block_addresses[current_block_start];
                    let end_addr = block_addresses[i - 1];
                    let size = end_addr as usize - start_addr as usize + 1;
                    let instruction_count = i - current_block_start;

                    basic_blocks.push(BasicBlockInfo {
                        start_address: start_addr,
                        end_address: i - 1,
                        size,
                        instruction_count,
                        successors: Vec::new(),
                        predecessors: Vec::new(),
                    });
                }

                current_block_start = i;
            }
        }

        // Add the final block
        if !block_instructions.is_empty() {
            let start_addr = block_addresses[current_block_start];
            let end_addr = block_addresses[block_instructions.len() - 1];
            let size = end_addr as usize - start_addr as usize + 1;
            let instruction_count = block_instructions.len() - current_block_start;

            basic_blocks.push(BasicBlockInfo {
                start_address: start_addr,
                end_address: block_instructions.len() - 1,
                size,
                instruction_count,
                successors: Vec::new(),
                predecessors: Vec::new(),
            });
        }

        symbol.basic_blocks = basic_blocks;
        symbol.size = symbol.instructions.iter().map(|instr| instr.length).sum();

        Ok(())
    }

    /// Analyzes function calls within a symbol
    fn analyze_calls(&self, symbol: &mut SymbolAnalysis) -> Result<()> {
        if !self.config.enable_call_graph_analysis {
            return Ok(());
        }

        for instruction in &symbol.instructions {
            if let Some(call_info) = self.analyze_call_instruction(instruction)? {
                symbol.calls.push(call_info);
            }
        }

        Ok(())
    }

    /// Checks if an instruction is a branch
    fn is_branch_instruction(&self, opcode: &[u8]) -> bool {
        // Simplified check - in practice would decode WASM opcodes
        opcode.len() >= 1 && (
            opcode[0] == 0x04 || // if
            opcode[0] == 0x0B || // block
            opcode[0] == 0x0C || // loop
            opcode[0] == 0x0D    // br
        )
    }

    /// Gets branch target from instruction
    fn get_branch_target(&self, instruction: &InstructionInfo) -> Result<u64> {
        // Simplified target extraction
        // In practice, would decode WASM branch targets
        Ok(instruction.address + instruction.length as u64)
    }

    /// Analyzes a call instruction
    fn analyze_call_instruction(&self, instruction: &InstructionInfo) -> Result<Option<CallInfo>> {
        // Check if instruction is a call
        if !self.is_call_instruction(&instruction.opcode) {
            return Ok(None);
        }

        let call_type = self.determine_call_type(instruction)?;
        let target_function = self.extract_call_target(instruction)?;

        Ok(Some(CallInfo {
            call_address: instruction.address,
            target_function,
            call_type,
        }))
    }

    /// Checks if instruction is a call
    fn is_call_instruction(&self, opcode: &[u8]) -> bool {
        opcode.len() >= 1 && (
            opcode[0] == 0x10 || // call
            opcode[0] == 0x11    // call_indirect
        )
    }

    /// Determines call type
    fn determine_call_type(&self, instruction: &InstructionInfo) -> Result<CallType> {
        if instruction.opcode[0] == 0x10 {
            Ok(CallType::Direct)
        } else if instruction.opcode[0] == 0x11 {
            Ok(CallType::Indirect)
        } else {
            anyhow::bail!("Unknown call instruction");
        }
    }

    /// Extracts call target from instruction
    fn extract_call_target(&self, instruction: &InstructionInfo) -> Result<Option<String>> {
        // Simplified target extraction
        // In practice, would decode WASM call targets
        if instruction.opcode[0] == 0x10 && instruction.operands.len() >= 4 {
            // Direct call - extract function index
            let target_bytes = &instruction.operands[0..4];
            let target_index = u32::from_le_bytes([
                target_bytes[0], target_bytes[1], target_bytes[2], target_bytes[3],
            ]);
            Ok(Some(format!("function_{}", target_index)))
        } else {
            Ok(None)
        }
    }

    /// Calculates hash for a function
    fn calculate_function_hash(&self, symbol: &SymbolAnalysis) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash instruction sequences (excluding addresses)
        for instruction in &symbol.instructions {
            instruction.opcode.hash(&mut hasher);
            instruction.operands.hash(&mut hasher);
        }

        hasher.finish()
    }

    /// Categorizes functions by type
    fn categorize_functions(
        &self,
        optimized_symbols: &[SymbolAnalysis],
        original_symbols: &[SymbolAnalysis],
    ) -> Result<(Vec<SymbolAnalysis>, Vec<SymbolAnalysis>, Vec<SymbolAnalysis>)> {
        let mut thinned_functions = Vec::new();
        let mut shim_functions = Vec::new();
        let mut remaining_functions = Vec::new();

        let original_names: HashSet<_> = original_symbols.iter()
            .map(|s| &s.name)
            .collect();

        for symbol in optimized_symbols {
            if symbol.name.contains("_thinned") {
                thinned_functions.push(symbol.clone());
            } else if symbol.name.contains("_shim") {
                shim_functions.push(symbol.clone());
            } else if original_names.contains(&symbol.name) {
                remaining_functions.push(symbol.clone());
            }
        }

        Ok((thinned_functions, shim_functions, remaining_functions))
    }

    /// Analyzes similarity between functions
    fn analyze_function_similarity(
        &self,
        original_symbols: &[SymbolAnalysis],
        thinned_functions: &[SymbolAnalysis],
        shim_functions: &[SymbolAnalysis],
    ) -> Result<Vec<FunctionSimilarity>> {
        let mut similarities = Vec::new();

        // Compare original functions with thinned functions
        for original in original_symbols {
            for thinned in thinned_functions {
                if self.are_related_functions(original, thinned) {
                    let similarity = self.calculate_function_similarity(original, thinned)?;
                    similarities.push(similarity);
                }
            }
        }

        // Compare shim functions with thinned functions
        for shim in shim_functions {
            for thinned in thinned_functions {
                if self.are_related_functions(shim, thinned) {
                    let similarity = self.calculate_function_similarity(shim, thinned)?;
                    similarities.push(similarity);
                }
            }
        }

        // Compare original functions with each other (for deduplication)
        for (i, func1) in original_symbols.iter().enumerate() {
            for func2 in original_symbols.iter().skip(i + 1) {
                let similarity = self.calculate_function_similarity(func1, func2)?;
                if similarity.similarity_score >= self.config.similarity_threshold {
                    similarities.push(similarity);
                }
            }
        }

        Ok(similarities)
    }

    /// Checks if two functions are related (thinned/original relationship)
    fn are_related_functions(&self, func1: &SymbolAnalysis, func2: &SymbolAnalysis) -> bool {
        let name1 = &func1.name;
        let name2 = &func2.name;

        // Remove suffixes to get base names
        let base1 = name1.trim_end_matches("_thinned")
            .trim_end_matches("_shim");
        let base2 = name2.trim_end_matches("_thinned")
            .trim_end_matches("_shim");

        base1 == base2
    }

    /// Calculates similarity between two functions
    fn calculate_function_similarity(
        &self,
        func1: &SymbolAnalysis,
        func2: &SymbolAnalysis,
    ) -> Result<FunctionSimilarity> {
        // Calculate instruction-level similarity
        let instruction_similarity = self.calculate_instruction_similarity(func1, func2)?;
        
        // Calculate control flow similarity
        let control_flow_similarity = self.calculate_control_flow_similarity(func1, func2)?;
        
        // Calculate call graph similarity
        let call_graph_similarity = self.calculate_call_graph_similarity(func1, func2)?;

        // Combined similarity score
        let overall_similarity = (
            instruction_similarity * 0.5 +
            control_flow_similarity * 0.3 +
            call_graph_similarity * 0.2
        ).max(0.0).min(1.0);

        // Determine similarity type
        let similarity_type = self.determine_similarity_type(func1, func2, overall_similarity)?;

        // Calculate shared instructions
        let shared_instructions = self.count_shared_instructions(func1, func2)?;
        let total_instructions = func1.instructions.len().max(func2.instructions.len());

        Ok(FunctionSimilarity {
            func1: func1.name.clone(),
            func2: func2.name.clone(),
            similarity_score: overall_similarity,
            shared_instructions,
            total_instructions,
            similarity_type,
        })
    }

    /// Calculates instruction-level similarity
    fn calculate_instruction_similarity(
        &self,
        func1: &SymbolAnalysis,
        func2: &SymbolAnalysis,
    ) -> Result<f64> {
        let total_instructions = func1.instructions.len().max(func2.instructions.len());
        if total_instructions == 0 {
            return Ok(1.0);
        }

        // Find longest common subsequence of instructions
        let lcs_length = self.find_lcs_length(&func1.instructions, &func2.instructions);
        
        Ok(lcs_length as f64 / total_instructions as f64)
    }

    /// Calculates control flow similarity
    fn calculate_control_flow_similarity(
        &self,
        func1: &SymbolAnalysis,
        func2: &SymbolAnalysis,
    ) -> Result<f64> {
        // Compare basic block structure
        let bb1_count = func1.basic_blocks.len();
        let bb2_count = func2.basic_blocks.len();
        
        if bb1_count == 0 && bb2_count == 0 {
            return Ok(1.0);
        }

        // Simple similarity based on basic block count
        let count_similarity = if bb1_count == bb2_count { 1.0 } else {
            (bb1_count.min(bb2_count) as f64 / bb1_count.max(bb2_count) as f64)
        };

        // Compare total control flow size
        let size1 = func1.basic_blocks.iter().map(|bb| bb.size).sum::<usize>();
        let size2 = func2.basic_blocks.iter().map(|bb| bb.size).sum::<usize>();
        let max_size = size1.max(size2);
        
        let size_similarity = if max_size == 0 { 1.0 } else {
            (size1.min(size2) as f64 / max_size as f64)
        };

        Ok((count_similarity + size_similarity) / 2.0)
    }

    /// Calculates call graph similarity
    fn calculate_call_graph_similarity(
        &self,
        func1: &SymbolAnalysis,
        func2: &SymbolAnalysis,
    ) -> Result<f64> {
        let calls1: HashSet<_> = func1.calls.iter()
            .filter_map(|c| c.target_function.clone())
            .collect();
        let calls2: HashSet<_> = func2.calls.iter()
            .filter_map(|c| c.target_function.clone())
            .collect();

        let union_size = calls1.union(&calls2).count();
        let intersection_size = calls1.intersection(&calls2).count();

        if union_size == 0 {
            return Ok(1.0);
        }

        Ok(intersection_size as f64 / union_size as f64)
    }

    /// Finds longest common subsequence length
    fn find_lcs_length(&self, seq1: &[InstructionInfo], seq2: &[InstructionInfo]) -> usize {
        // Dynamic programming approach for LCS
        let m = seq1.len();
        let n = seq2.len();
        let mut dp = vec![vec![0; n + 1]; m + 1];

        for i in 1..=m {
            for j in 1..=n {
                if self.instructions_equal(&seq1[i - 1], &seq2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
                }
            }
        }

        dp[m][n]
    }

    /// Checks if two instructions are equal for comparison purposes
    fn instructions_equal(&self, instr1: &InstructionInfo, instr2: &InstructionInfo) -> bool {
        // Compare opcodes
        if instr1.opcode != instr2.opcode {
            return false;
        }

        // Compare operands (ignoring address-dependent parts)
        self.normalize_operands(&instr1.operands) == self.normalize_operands(&instr2.operands)
    }

    /// Normalizes operands for comparison
    fn normalize_operands(&self, operands: &[u8]) -> Vec<u8> {
        // Remove address-specific bytes for comparison
        operands.iter()
            .filter(|&&b| b != 0x00) // Remove null bytes (simplified)
            .cloned()
            .collect()
    }

    /// Determines similarity type based on characteristics
    fn determine_similarity_type(
        &self,
        func1: &SymbolAnalysis,
        func2: &SymbolAnalysis,
        similarity_score: f64,
    ) -> Result<SimilarityType> {
        if similarity_score >= 0.95 {
            return Ok(SimilarityType::Identical);
        }

        let name1_contains_thinned = func1.name.contains("_thinned");
        let name2_contains_thinned = func2.name.contains("_thinned");
        let name1_contains_shim = func1.name.contains("_shim");
        let name2_contains_shim = func2.name.contains("_shim");

        if (name1_contains_thinned || name2_contains_thinned) && similarity_score >= 0.7 {
            Ok(SimilarityType::ThinnedShared)
        } else if similarity_score >= 0.5 {
            Ok(SimilarityType::PatternSimilar)
        } else {
            Ok(SimilarityType::Different)
        }
    }

    /// Counts shared instructions between functions
    fn count_shared_instructions(
        &self,
        func1: &SymbolAnalysis,
        func2: &SymbolAnalysis,
    ) -> Result<usize> {
        self.find_lcs_length(&func1.instructions, &func2.instructions)
    }

    /// Identifies shared code regions across functions
    fn identify_shared_regions(
        &self,
        original_symbols: &[SymbolAnalysis],
        thinned_functions: &[SymbolAnalysis],
        shim_functions: &[SymbolAnalysis],
    ) -> Result<Vec<SharedRegion>> {
        let mut shared_regions = Vec::new();
        let mut region_hash_map = HashMap::new();

        // Process all functions to find common instruction sequences
        let all_functions: Vec<&SymbolAnalysis> = original_symbols
            .iter()
            .chain(thinned_functions.iter())
            .chain(shim_functions.iter())
            .collect();

        // Find common instruction sequences of minimum length
        let min_sequence_length = 5;
        
        for function in all_functions {
            for start_idx in 0..function.instructions.len().saturating_sub(min_sequence_length) {
                for length in min_sequence_length..=function.instructions.len() - start_idx {
                    let sequence = &function.instructions[start_idx..start_idx + length];
                    let hash = self.calculate_sequence_hash(sequence);

                    let region = region_hash_map.entry(hash).or_insert_with(|| SharedRegion {
                        hash,
                        size: length,
                        functions: Vec::new(),
                        addresses: HashMap::new(),
                    });

                    if !region.functions.contains(&function.name) {
                        region.functions.push(function.name.clone());
                    }
                    
                    region.addresses.insert(
                        function.name.clone(),
                        sequence[0].address,
                    );
                }
            }
        }

        // Filter regions shared by multiple functions
        for region in region_hash_map.into_values() {
            if region.functions.len() >= 2 {
                shared_regions.push(region);
            }
        }

        // Sort by size (largest first)
        shared_regions.sort_by(|a, b| b.size.cmp(&a.size));

        Ok(shared_regions)
    }

    /// Calculates hash for an instruction sequence
    fn calculate_sequence_hash(&self, sequence: &[InstructionInfo]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        for instruction in sequence {
            instruction.opcode.hash(&mut hasher);
            self.normalize_operands(&instruction.operands).hash(&mut hasher);
        }

        hasher.finish()
    }

    /// Calculates overall similarity metrics
    fn calculate_metrics(
        &self,
        original_symbols: &[SymbolAnalysis],
        thinned_functions: &[SymbolAnalysis],
        shim_functions: &[SymbolAnalysis],
        shared_regions: &[SharedRegion],
    ) -> Result<SimilarityMetrics> {
        let total_functions = original_symbols.len() + thinned_functions.len() + shim_functions.len();
        
        // Calculate average similarity
        let mut total_similarity = 0.0;
        let mut similarity_count = 0;

        // Find similarities between related functions
        for original in original_symbols {
            for thinned in thinned_functions {
                if self.are_related_functions(original, thinned) {
                    let similarity = self.calculate_function_similarity(original, thinned)?;
                    total_similarity += similarity.similarity_score;
                    similarity_count += 1;
                }
            }
        }

        let average_similarity = if similarity_count > 0 {
            total_similarity / similarity_count as f64
        } else {
            0.0
        };

        // Calculate shared code size
        let shared_code_size: usize = shared_regions.iter()
            .map(|r| r.size)
            .sum();

        // Calculate sharing efficiency
        let total_code_size = original_symbols.iter()
            .chain(thinned_functions.iter())
            .chain(shim_functions.iter())
            .map(|s| s.size)
            .sum::<usize>();

        let sharing_efficiency = if total_code_size > 0 {
            shared_code_size as f64 / total_code_size as f64
        } else {
            0.0
        };

        // Estimate expected size reduction
        let original_total_size: usize = original_symbols.iter()
            .map(|s| s.size)
            .sum();
        
        let optimized_total_size: usize = thinned_functions.iter()
            .chain(shim_functions.iter())
            .map(|s| s.size)
            .sum();

        let expected_size_reduction = if original_total_size > 0 {
            ((original_total_size - optimized_total_size) as f64 / original_total_size as f64) * 100.0
        } else {
            0.0
        };

        Ok(SimilarityMetrics {
            total_functions,
            thinned_functions: thinned_functions.len(),
            shim_functions: shim_functions.len(),
            average_similarity,
            shared_code_size,
            sharing_efficiency,
            expected_size_reduction,
        })
    }

    /// Generates a detailed similarity report
    pub fn generate_report(&self, analysis: &SimilarityAnalysis) -> Result<String> {
        let mut report = String::new();
        
        report.push_str("# Symbol Similarity Analysis Report\n\n");
        
        // Executive summary
        report.push_str("## Executive Summary\n\n");
        report.push_str(&format!("- Total functions analyzed: {}\n", analysis.metrics.total_functions));
        report.push_str(&format!("- Thinned functions: {}\n", analysis.metrics.thinned_functions));
        report.push_str(&format!("- Shim functions: {}\n", analysis.metrics.shim_functions));
        report.push_str(&format!("- Average similarity: {:.2}%\n", analysis.metrics.average_similarity * 100.0));
        report.push_str(&format!("- Shared code size: {} bytes\n", analysis.metrics.shared_code_size));
        report.push_str(&format!("- Sharing efficiency: {:.2}%\n", analysis.metrics.sharing_efficiency * 100.0));
        report.push_str(&format!("- Expected size reduction: {:.1}%\n\n", analysis.metrics.expected_size_reduction));
        
        // Similarity scores
        report.push_str("## Function Similarities\n\n");
        for similarity in &analysis.similarity_scores {
            report.push_str(&format!(
                "### {} <-> {}\n",
                similarity.func1, similarity.func2
            ));
            report.push_str(&format!("- Similarity: {:.2}%\n", similarity.similarity_score * 100.0));
            report.push_str(&format!("- Type: {:?}\n", similarity.similarity_type));
            report.push_str(&format!("- Shared instructions: {} / {}\n\n", 
                similarity.shared_instructions, similarity.total_instructions));
        }
        
        // Shared regions
        report.push_str("## Shared Code Regions\n\n");
        for (i, region) in analysis.shared_regions.iter().take(10).enumerate() { // Top 10 regions
            report.push_str(&format!("### Region {} ({} bytes)\n", i + 1, region.size));
            report.push_str(&format!("- Shared by: {}\n", region.functions.join(", ")));
            report.push_str("\n");
        }
        
        // Detailed function information
        if self.config.verbose {
            report.push_str("## Detailed Function Analysis\n\n");
            
            for category in [
                ("Original Functions", &analysis.original_functions),
                ("Thinned Functions", &analysis.thinned_functions),
                ("Shim Functions", &analysis.shim_functions),
            ] {
                report.push_str(&format!("### {}\n\n", category.0));
                
                for function in category.1 {
                    report.push_str(&format!("#### {}\n", function.name));
                    report.push_str(&format!("- Size: {} bytes\n", function.size));
                    report.push_str(&format!("- Instructions: {}\n", function.instructions.len()));
                    report.push_str(&format!("- Basic blocks: {}\n", function.basic_blocks.len()));
                    report.push_str(&format!("- Calls: {}\n", function.calls.len()));
                    report.push_str("\n");
                }
            }
        }
        
        Ok(report)
    }

    /// Verifies that thinning meets requirements
    pub fn verify_thinning_requirements(&self, analysis: &SimilarityAnalysis) -> Result<bool> {
        let mut passed_checks = Vec::new();
        
        // Check 1: Should have thinned functions
        if analysis.metrics.thinned_functions > 0 {
            passed_checks.push("Has thinned functions");
        }
        
        // Check 2: Should have shim functions
        if analysis.metrics.shim_functions > 0 {
            passed_checks.push("Has shim functions");
        }
        
        // Check 3: Should achieve similarity threshold
        if analysis.metrics.average_similarity >= self.config.similarity_threshold {
            passed_checks.push("Meets similarity threshold");
        }
        
        // Check 4: Should achieve size reduction
        if analysis.metrics.expected_size_reduction >= 10.0 { // 10% minimum
            passed_checks.push("Achieves size reduction");
        }
        
        // Check 5: Should have shared code regions
        if !analysis.shared_regions.is_empty() {
            passed_checks.push("Has shared code regions");
        }
        
        let total_checks = 5;
        let passed_count = passed_checks.len();
        let all_passed = passed_count == total_checks;
        
        println!("Verification Results:");
        for check in passed_checks {
            println!("  ✓ {}", check);
        }
        if passed_count < total_checks {
            println!("  {} / {} checks passed", passed_count, total_checks);
        }
        
        Ok(all_passed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_checker_creation() {
        let config = SymbolCheckerConfig::default();
        let checker = SymbolSimilarityChecker::new(config);
        assert_eq!(checker.config.wasm_objdump_path, "wasm-objdump");
    }

    #[test]
    fn test_similarity_threshold_validation() {
        let config = SymbolCheckerConfig {
            similarity_threshold: 0.8,
            ..Default::default()
        };
        
        let checker = SymbolSimilarityChecker::new(config);
        assert_eq!(checker.config.similarity_threshold, 0.8);
    }

    #[test]
    fn test_related_function_detection() {
        let config = SymbolCheckerConfig::default();
        let checker = SymbolSimilarityChecker::new(config);
        
        let func1 = SymbolAnalysis {
            name: "process_i32_thinned".to_string(),
            address: 0,
            size: 100,
            hash: 0,
            instructions: Vec::new(),
            basic_blocks: Vec::new(),
            calls: Vec::new(),
        };
        
        let func2 = SymbolAnalysis {
            name: "process_i32".to_string(),
            address: 0,
            size: 80,
            hash: 0,
            instructions: Vec::new(),
            basic_blocks: Vec::new(),
            calls: Vec::new(),
        };
        
        assert!(checker.are_related_functions(&func1, &func2));
    }
}
