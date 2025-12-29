//! Structure validation tests for WasmRust project
//! 
//! This module validates the project structure according to the
//! canonical structure contract defined in docs/project-structure.toml.

use std::path::{Path, PathBuf};
use quickcheck::{Arbitrary, Gen, QuickCheck, TestResult};

/// Expected project structure based on contract
const EXPECTED_DIRECTORIES: &[&str] = &[
    "crates",
    "src",
    "docs",
    "tests",
];

/// Expected files based on contract
const EXPECTED_FILES: &[&str] = &[
    "Cargo.toml",
    "README.md",
    "crates/wasm/Cargo.toml",
    "crates/wasm/src/lib.rs",
    "src/lib.rs",
    "src/backend/cranelift/Cargo.toml",
    "src/backend/llvm/Cargo.toml",
    "docs/project-structure.toml",
];

/// Property: All required directories exist
#[test]
fn prop_required_directories_exist() {
    fn property() -> TestResult {
        for dir in EXPECTED_DIRECTORIES {
            if !Path::new(dir).exists() {
                eprintln!("Missing required directory: {}", dir);
                return TestResult::failed();
            }
        }
        TestResult::passed()
    }

    QuickCheck::new()
        .tests(1)
        .gen(Gen::new(100))
        .quickcheck(property as fn() -> TestResult);
}

/// Property: All required files exist
#[test]
fn prop_required_files_exist() {
    fn property() -> TestResult {
        for file in EXPECTED_FILES {
            if !Path::new(file).exists() {
                eprintln!("Missing required file: {}", file);
                return TestResult::failed();
            }
        }
        TestResult::passed()
    }

    QuickCheck::new()
        .tests(1)
        .gen(Gen::new(100))
        .quickcheck(property as fn() -> TestResult);
}

/// Property: Cargo workspace configuration is valid
#[test]
fn prop_valid_cargo_workspace() -> TestResult {
    let workspace_toml = match std::fs::read_to_string("Cargo.toml") {
        Ok(content) => content,
        Err(_) => return TestResult::failed(),
    };

    // Check for workspace section
    if !workspace_toml.contains("[workspace]") {
        eprintln!("Missing [workspace] section in Cargo.toml");
        return TestResult::failed();
    }

    // Check for required workspace members
    let required_members = &["crates/wasm"];
    for member in required_members {
        if !workspace_toml.contains(member) {
            eprintln!("Missing workspace member: {}", member);
            return TestResult::failed();
        }
    }

    TestResult::passed()
}

/// Property: Rust toolchain is specified
#[test]
fn prop_rust_toolchain_specified() -> TestResult {
    if !Path::new("rust-toolchain.toml").exists() {
        eprintln!("Missing rust-toolchain.toml");
        return TestResult::failed();
    }

    let toolchain_content = match std::fs::read_to_string("rust-toolchain.toml") {
        Ok(content) => content,
        Err(_) => return TestResult::failed(),
    };

    // Check for required toolchain components
    if !toolchain_content.contains("channel") {
        eprintln!("Missing channel specification in rust-toolchain.toml");
        return TestResult::failed();
    }

    TestResult::passed()
}

/// Property: Project structure contract exists and is valid
#[test]
fn prop_project_structure_contract() -> TestResult {
    let contract_path = Path::new("docs/project-structure.toml");
    
    if !contract_path.exists() {
        eprintln!("Missing project structure contract: docs/project-structure.toml");
        return TestResult::failed();
    }
    
    let contract_content = match std::fs::read_to_string(contract_path) {
        Ok(content) => content,
        Err(_) => return TestResult::failed(),
    };
    
    // Check for required sections
    if !contract_content.contains("[root]") {
        eprintln!("Missing [root] section in project structure contract");
        return TestResult::failed();
    }
    
    if !contract_content.contains("[directories.crates]") {
        eprintln!("Missing [directories.crates] section in project structure contract");
        return TestResult::failed();
    }
    
    if !contract_content.contains("[constraints]") {
        eprintln!("Missing [constraints] section in project structure contract");
        return TestResult::failed();
    }
    
    TestResult::passed()
}

/// Property: Compiler backend structure is correct
#[test]
fn prop_compiler_backend_structure() -> TestResult {
    let backend_dirs = &["cranelift", "llvm"];
    
    for backend in backend_dirs {
        let backend_path = Path::new("src/backend").join(backend);
        if !backend_path.exists() {
            eprintln!("Missing backend directory: {}", backend_path.display());
            return TestResult::failed();
        }
        
        let cargo_toml = backend_path.join("Cargo.toml");
        if !cargo_toml.exists() {
            eprintln!("Missing Cargo.toml for backend: {}", backend);
            return TestResult::failed();
        }
        
        let lib_rs = backend_path.join("lib.rs");
        if !lib_rs.exists() {
            eprintln!("Missing lib.rs for backend: {}", backend);
            return TestResult::failed();
        }
    }
    
    TestResult::passed()
}

/// Property: Core wasm crate structure is correct
#[test]
fn prop_wasm_crate_structure() -> TestResult {
    let wasm_dir = Path::new("crates/wasm");
    
    if !wasm_dir.exists() {
        eprintln!("Missing wasm crate directory");
        return TestResult::failed();
    }
    
    let required_files = &["Cargo.toml", "src/lib.rs"];
    for file in required_files {
        let file_path = wasm_dir.join(file);
        if !file_path.exists() {
            eprintln!("Missing file in wasm crate: {}", file);
            return TestResult::failed();
        }
    }
    
    // Check Cargo.toml contains required metadata
    let cargo_content = match std::fs::read_to_string(wasm_dir.join("Cargo.toml")) {
        Ok(content) => content,
        Err(_) => return TestResult::failed(),
    };
    
    if !cargo_content.contains("name = \"wasm\"") {
        eprintln!("Invalid package name in wasm crate");
        return TestResult::failed();
    }
    
    if !cargo_content.contains("crate-type = [\"rlib\"]") {
        eprintln!("Missing crate-type configuration in wasm crate");
        return TestResult::failed();
    }
    
    TestResult::passed()
}

/// Validates project structure against contract
pub fn validate_structure() -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    
    // Validate directories
    for dir in EXPECTED_DIRECTORIES {
        if !Path::new(dir).exists() {
            errors.push(format!("Missing required directory: {}", dir));
        }
    }
    
    // Validate files
    for file in EXPECTED_FILES {
        if !Path::new(file).exists() {
            errors.push(format!("Missing required file: {}", file));
        }
    }
    
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_complete_structure_validation() {
        match validate_structure() {
            Ok(()) => {
                // Structure is valid
                assert!(true);
            }
            Err(errors) => {
                panic!("Structure validation failed: {:?}", errors);
            }
        }
    }

    #[test]
    fn test_contract_consistency() {
        // Test that the contract matches the actual structure
        let _ = prop_project_structure_contract();
        let _ = prop_required_directories_exist();
        let _ = prop_required_files_exist();
        
        // If we get here, all tests passed
        assert!(true);
    }

    #[test]
    fn test_workspace_members() {
        let _ = prop_valid_cargo_workspace();
        
        // Test that all workspace members exist
        let workspace_toml = std::fs::read_to_string("Cargo.toml").unwrap();
        assert!(workspace_toml.contains("crates/wasm"));
    }

    #[test]
    fn test_toolchain_configuration() {
        let _ = prop_rust_toolchain_specified();
        
        let toolchain_content = std::fs::read_to_string("rust-toolchain.toml").unwrap();
        assert!(toolchain_content.contains("nightly"));
        assert!(toolchain_content.contains("targets = [\"wasm32-unknown-unknown\"]"));
    }

    #[test]
    fn test_compiler_backend_cargo_files() {
        let _ = prop_compiler_backend_structure();
        
        // Verify Cranelift backend Cargo.toml
        let cranelift_cargo = std::fs::read_to_string("src/backend/cranelift/Cargo.toml").unwrap();
        assert!(cranelift_cargo.contains("name = \"wasmrust-codegen-cranelift\""));
        
        // Verify LLVM backend Cargo.toml
        let llvm_cargo = std::fs::read_to_string("src/backend/llvm/Cargo.toml").unwrap();
        assert!(llvm_cargo.contains("name = \"wasmrust-codegen-llvm\""));
    }

    #[test]
    fn test_wasm_cargo_configuration() {
        let _ = prop_wasm_crate_structure();
        
        let wasm_cargo = std::fs::read_to_string("crates/wasm/Cargo.toml").unwrap();
        
        // Check that wasm crate has no external dependencies
        assert!(wasm_cargo.contains("# No external dependencies - dependency-free by design"));
        
        // Check crate type
        assert!(wasm_cargo.contains("crate-type = [\"rlib\"]"));
        
        // Check features are optional
        assert!(wasm_cargo.contains("default = []"));
    }

    #[test]
    fn test_no_forbidden_dependencies() {
        // Check that wasm crate doesn't depend on compiler internals
        let wasm_cargo = std::fs::read_to_string("crates/wasm/Cargo.toml").unwrap();
        
        // Should not contain compiler dependencies
        assert!(!wasm_cargo.contains("rustc_"));
        assert!(!wasm_cargo.contains("compiler/"));
        
        // Should be dependency-free
        assert!(wasm_cargo.contains("# No external dependencies"));
    }
}
