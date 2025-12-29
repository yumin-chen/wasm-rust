//! Cranelift Backend Module
//! 
//! This module provides the Cranelift-based codegen backend for WasmRust,
//! optimized for fast development compilation.

pub mod lib;
pub mod integration;
pub mod mir_lowering;
pub mod thin_monomorphization;
pub mod type_descriptor;
pub mod mir_complexity;
pub mod thinning_pass;
pub mod size_analyzer;
pub mod streaming_optimizer;
pub mod indirect_call_optimizer;

// Re-export main types
pub use lib::*;
pub use thin_monomorphization::*;
pub use type_descriptor::*;
pub use mir_complexity::*;
pub use thinning_pass::*;
pub use size_analyzer::*;
pub use streaming_optimizer::*;
pub use indirect_call_optimizer::*;
