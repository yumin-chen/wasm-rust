//! Property-based tests for WasmRust project structure consistency
//! 
//! This module validates that the WasmRust project follows established
//! structural conventions and maintains consistency across all components.
//! 
//! Property 15: Project Structure Consistency
//! Validates: Requirements 12.3

use std::path::{Path, PathBuf};
use std::collections::HashMap;

mod structure_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use super::structure_tests::validate_structure;
    use quickcheck::{Arbitrary, Gen, QuickCheck, TestResult};
    use std::fs;

    /// Project structure configuration
    const EXPECTED_DIRECTORIES: &[&str] = &[
        "src",
        "src/frontend",
        "src/wasmir", 
        "src/backend",
        "src/backend/cranelift",
        "src/backend/llvm",
        "src/runtime",
        "src/rtypes",
        "crates",
        "crates/wasm",
        "crates/wasm-macros",
        "crates/cargo-wasm",
        "tests",
        "benches",
        "docs",
        ".github/workflows",
    ];

    const EXPECTED_FILES: &[&str] = &[
        "Cargo.toml",
        "Cargo.lock",
        "README.md",
        "LICENSE",
        "src/lib.rs",
        "crates/wasm/Cargo.toml",
        "crates/wasm-macros/Cargo.toml",
        "crates/cargo-wasm/Cargo.toml",
        "rust-toolchain.toml",
    ];

    #[derive(Debug, Clone)]
    struct ProjectPath(PathBuf);

    impl Arbitrary for ProjectPath {
        fn arbitrary(g: &mut Gen) -> Self {
            let depth = g.gen_range(0..5);
            let mut path = PathBuf::new();
            
            for _ in 0..depth {
                let component = if g.gen_bool() {
                    format!("dir{}", g.gen_range(0..10))
                } else {
                    format!("file{}.rs", g.gen_range(0..10))
                };
                path.push(component);
            }
            
            ProjectPath(path)
        }
    }

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
        let workspace_toml = match fs::read_to_string("Cargo.toml") {
            Ok(content) => content,
            Err(_) => return TestResult::failed(),
        };

        // Check for workspace section
        if !workspace_toml.contains("[workspace]") {
            eprintln!("Missing [workspace] section in Cargo.toml");
            return TestResult::failed();
        }

        // Check for required workspace members
        let required_members = &["crates/wasm", "crates/wasm-macros", "crates/cargo-wasm"];
        for member in required_members {
            if !workspace_toml.contains(member) {
                eprintln!("Missing workspace member: {}", member);
                return TestResult::failed();
            }
        }

        // Validate each member has its own Cargo.toml
        for member in required_members {
            let cargo_path = format!("{}/Cargo.toml", member);
            if !Path::new(&cargo_path).exists() {
                eprintln!("Missing Cargo.toml for workspace member: {}", member);
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

        let toolchain_content = match fs::read_to_string("rust-toolchain.toml") {
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

    /// Property: CI/CD workflows are properly configured
    #[test]
    fn prop_ci_workflows_configured() -> TestResult {
        let workflows_dir = Path::new(".github/workflows");
        if !workflows_dir.exists() {
            eprintln!("Missing .github/workflows directory");
            return TestResult::failed();
        }

        let required_workflows = &["ci.yml", "test.yml", "security.yml"];
        for workflow in required_workflows {
            let workflow_path = workflows_dir.join(workflow);
            if !workflow_path.exists() {
                eprintln!("Missing CI workflow: {}", workflow);
                return TestResult::failed();
            }
        }

        TestResult::passed()
    }

    /// Property: Documentation is properly structured
    #[test]
    fn prop_documentation_structure() -> TestResult {
        let docs_dir = Path::new("docs");
        if !docs_dir.exists() {
            eprintln!("Missing docs directory");
            return TestResult::failed();
        }

        let required_docs = &["README.md", "architecture.md", "api.md"];
        for doc in required_docs {
            let doc_path = docs_dir.join(doc);
            if !doc_path.exists() {
                eprintln!("Missing documentation: {}", doc);
                return TestResult::failed();
            }
        }

        // Check main README.md exists and contains required sections
        let readme_content = match fs::read_to_string("README.md") {
            Ok(content) => content,
            Err(_) => {
                eprintln!("Cannot read main README.md");
                return TestResult::failed();
            }
        };

        let required_sections = &["## Overview", "## Installation", "## Usage"];
        for section in required_sections {
            if !readme_content.contains(section) {
                eprintln!("Missing section in README.md: {}", section);
                return TestResult::failed();
            }
        }

        TestResult::passed()
    }

    /// Property: Source code follows module structure conventions
    #[test]
    fn prop_source_module_structure() -> TestResult {
        let src_dir = Path::new("src");
        if !src_dir.exists() {
            eprintln!("Missing src directory");
            return TestResult::failed();
        }

        // Check for lib.rs
        if !src_dir.join("lib.rs").exists() {
            eprintln!("Missing src/lib.rs");
            return TestResult::failed();
        }

        // Validate main modules exist
        let main_modules = &["frontend", "wasmir", "backend", "runtime"];
        for module in main_modules {
            let module_path = src_dir.join(module);
            if !module_path.exists() {
                eprintln!("Missing main module: {}", module);
                return TestResult::failed();
            }

            // Check each module has mod.rs
            let mod_rs = module_path.join("mod.rs");
            if !mod_rs.exists() {
                eprintln!("Missing mod.rs in module: {}", module);
                return TestResult::failed();
            }
        }

        TestResult::passed()
    }

    /// Property: Testing infrastructure is properly set up
    #[test]
    fn prop_testing_infrastructure() -> TestResult {
        // Check for test directories
        if !Path::new("tests").exists() {
            eprintln!("Missing tests directory");
            return TestResult::failed();
        }

        if !Path::new("benches").exists() {
            eprintln!("Missing benches directory");
            return TestResult::failed();
        }

        // Check for main Cargo.toml has dev-dependencies for testing
        let cargo_toml = match fs::read_to_string("Cargo.toml") {
            Ok(content) => content,
            Err(_) => return TestResult::failed(),
        };

        let test_deps = &["quickcheck", "criterion", "proptest"];
        for dep in test_deps {
            if !cargo_toml.contains(dep) {
                eprintln!("Missing test dependency: {}", dep);
                return TestResult::failed();
            }
        }

        TestResult::passed()
    }

    /// Property: License and legal files are present
    #[test]
    fn prop_legal_files_present() -> TestResult {
        let legal_files = &["LICENSE", "LICENSE-MIT", "LICENSE-APACHE"];
        let mut found = false;

        for file in legal_files {
            if Path::new(file).exists() {
                found = true;
                break;
            }
        }

        if !found {
            eprintln!("Missing license file (expected LICENSE, LICENSE-MIT, or LICENSE-APACHE)");
            return TestResult::failed();
        }

        // Check for SECURITY.md
        if !Path::new("SECURITY.md").exists() {
            eprintln!("Missing SECURITY.md file");
            return TestResult::failed();
        }

        TestResult::passed()
    }

    /// Property: Configuration files follow conventions
    #[test]
    fn prop_configuration_files() -> TestResult {
        // Check for .gitignore
        if !Path::new(".gitignore").exists() {
            eprintln!("Missing .gitignore");
            return TestResult::failed();
        }

        // Check for common .gitignore entries
        let gitignore_content = match fs::read_to_string(".gitignore") {
            Ok(content) => content,
            Err(_) => return TestResult::failed(),
        };

        let gitignore_patterns = &["/target/", "*.swp", "*.swo", "*~"];
        for pattern in gitignore_patterns {
            if !gitignore_content.contains(pattern) {
                eprintln!("Missing gitignore pattern: {}", pattern);
                return TestResult::failed();
            }
        }

        TestResult::passed()
    }
}

/// Utility function to validate project structure
pub fn validate_project_structure() -> Result<(), Vec<String>> {
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
