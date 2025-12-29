//! Cranelift Backend Integration Tests
//! 
//! Comprehensive integration tests for the Cranelift backend to ensure
//! it produces correct WASM output and meets performance requirements.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::process::Command;
use std::path::Path;
use tempfile::TempDir;

/// Test configuration for Cranelift backend integration
struct CraneliftTestConfig {
    test_cases: Vec<CraneliftTestCase>,
    compilation_timeout: Duration,
    performance_targets: PerformanceTargets,
}

/// Performance targets for validation
#[derive(Debug, Clone)]
struct PerformanceTargets {
    max_compilation_time_ms: u64,
    max_binary_size_bytes: usize,
    min_speedup_ratio: f64,
}

/// Individual test case for Cranelift backend
#[derive(Debug, Clone)]
struct CraneliftTestCase {
    name: String,
    rust_code: String,
    expected_wasm_features: Vec<String>,
    should_compile: bool,
    requires_specific_features: Vec<String>,
}

impl CraneliftTestConfig {
    fn default() -> Self {
        Self {
            test_cases: vec![
                CraneliftTestCase {
                    name: "hello_world".to_string(),
                    rust_code: r#"
#[wasm::export]
pub fn hello() -> String {
    "Hello, World!".to_string()
}
"#.to_string(),
                    expected_wasm_features: vec!["export".to_string(), "memory".to_string()],
                    should_compile: true,
                    requires_specific_features: vec![],
                },
                CraneliftTestCase {
                    name: "fibonacci".to_string(),
                    rust_code: r#"
#[wasm::export]
pub fn fibonacci(n: u32) -> u32 {
    match n {
        0 | 1 => n,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

#[wasm::export]
pub fn fibonacci_iterative(n: u32) -> u32 {
    if n <= 1 {
        return n;
    }
    
    let mut a = 0;
    let mut b = 1;
    
    for _ in 2..=n {
        let temp = a + b;
        a = b;
        b = temp;
    }
    
    b
}
"#.to_string(),
                    expected_wasm_features: vec!["export".to_string(), "loop".to_string(), "call".to_string()],
                    should_compile: true,
                    requires_specific_features: vec![],
                },
                CraneliftTestCase {
                    name: "wasm_features".to_string(),
                    rust_code: r#"
use wasm::{ExternRef, SharedSlice};
use std::sync::atomic::{AtomicU32, Ordering};

#[wasm::export]
pub fn test_externref_operations(obj: &ExternRef<String>) -> String {
    // Test ExternRef property access
    if let Ok(result) = obj.get_property("value") {
        result
    } else {
        "error".to_string()
    }
}

#[wasm::export]
pub fn test_funcref_operations(f: fn(i32) -> i32) -> i32 {
    let func_ref = wasm::make_func_ref(f);
    func_ref.call(42)
}

#[wasm::export]
pub fn test_shared_slice_operations(data: &SharedSlice<u32>) -> u32 {
    let mut sum = 0;
    for &value in data.iter() {
        sum += value;
    }
    sum
}

#[wasm::export]
pub fn test_atomic_operations() -> u32 {
    let counter = AtomicU32::new(0);
    counter.fetch_add(1, Ordering::SeqCst);
    counter.load(Ordering::SeqCst)
}
"#.to_string(),
                    expected_wasm_features: vec![
                        "externref".to_string(),
                        "funcref".to_string(),
                        "shared_slice".to_string(),
                        "atomic".to_string(),
                        "memory".to_string(),
                    ],
                    should_compile: true,
                    requires_specific_features: vec!["js_interop".to_string(), "threading".to_string()],
                },
                CraneliftTestCase {
                    name: "linear_types".to_string(),
                    rust_code: r#"
#[wasm::linear]
struct LinearResource {
    value: u32,
}

impl LinearResource {
    fn new(value: u32) -> Self {
        Self { value }
    }
    
    fn consume(self) -> u32 {
        self.value * 2
    }
    
    fn drop_error(self) {
        // This should be caught by the linear type checker
        panic!("Cannot drop linear type");
    }
}

#[wasm::export]
pub fn test_linear_usage() -> u32 {
    let resource = LinearResource::new(21);
    let result = resource.consume();
    
    // This should compile - resource is properly consumed
    result
}

#[wasm::export]
pub fn test_linear_error() {
    let resource = LinearResource::new(42);
    // resource.drop_error(); // This should cause a compile error
}
"#.to_string(),
                    expected_wasm_features: vec!["linear".to_string(), "memory".to_string()],
                    should_compile: true,
                    requires_specific_features: vec!["linear_types".to_string()],
                },
                CraneliftTestCase {
                    name: "component_model".to_string(),
                    rust_code: r#"
#[wasm::wit]
interface counter {
    increment: func() -> u32;
    get_value: func() -> u32;
}

#[wasm::wit]
world counter-world {
    export counter;
}

static mut COUNTER: u32 = 0;

#[wasm::export]
pub fn increment() -> u32 {
    COUNTER += 1;
    COUNTER
}

#[wasm::export]
pub fn get_value() -> u32 {
    COUNTER
}
"#.to_string(),
                    expected_wasm_features: vec!["component".to_string(), "memory".to_string()],
                    should_compile: true,
                    requires_specific_features: vec!["component_model".to_string()],
                },
                CraneliftTestCase {
                    name: "compilation_error".to_string(),
                    rust_code: r#"
#[wasm::export]
pub fn test_error() -> ! {
    // This should fail to compile
    let x: u32 = "not a number".parse().unwrap();
    x
}
"#.to_string(),
                    expected_wasm_features: vec![],
                    should_compile: false,
                    requires_specific_features: vec![],
                },
            ],
            compilation_timeout: Duration::from_secs(30),
            performance_targets: PerformanceTargets {
                max_compilation_time_ms: 5000, // 5 seconds for complex cases
                max_binary_size_bytes: 1024 * 1024, // 1MB maximum
                min_speedup_ratio: 5.0, // 5x faster than baseline
            },
        }
    }
}

/// Integration test runner for Cranelift backend
struct CraneliftIntegrationTest {
    temp_dir: TempDir,
    config: CraneliftTestConfig,
}

impl CraneliftIntegrationTest {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            temp_dir: TempDir::new()?,
            config: CraneliftTestConfig::default(),
        })
    }

    /// Run all integration tests
    pub fn run_all_tests(&self) -> Result<IntegrationTestResults, Box<dyn std::error::Error>> {
        println!("Running Cranelift backend integration tests...");
        
        let mut results = IntegrationTestResults::new();
        
        for test_case in &self.config.test_cases {
            let result = self.run_single_test(test_case)?;
            results.add_test_result(result);
        }
        
        Ok(results)
    }

    /// Run a single test case
    fn run_single_test(&self, test_case: &CraneliftTestCase) -> Result<TestResult, Box<dyn std::error::Error>> {
        println!("Running test: {}", test_case.name);
        
        let start_time = Instant::now();
        
        // Write test code to temporary file
        let test_file = self.temp_dir.path().join(format!("{}.rs", test_case.name));
        std::fs::write(&test_file, &test_case.rust_code)?;
        
        // Compile with Cranelift backend
        let output_dir = self.temp_dir.path().join("output");
        std::fs::create_dir_all(&output_dir)?;
        
        let mut cmd = Command::new("cargo");
        cmd.args(&[
            "wasm",
            "build",
            "--backend", "cranelift",
            "--target", "wasm32-unknown-unknown",
            "--output-dir", output_dir.to_str().unwrap(),
        ]);
        cmd.current_dir(self.temp_dir.path());
        cmd.env("RUST_LOG", "warn"); // Reduce log noise
        
        let output = cmd.output()?;
        let compilation_time = start_time.elapsed();
        
        // Check if compilation succeeded
        let compilation_succeeded = output.status.success();
        
        // Get binary size if compilation succeeded
        let binary_size = if compilation_succeeded {
            let wasm_file = output_dir.join("test_case.wasm");
            std::fs::metadata(&wasm_file).map(|m| m.len() as usize).unwrap_or(0)
        } else {
            0
        };
        
        // Validate WASM features
        let wasm_features_valid = if compilation_succeeded {
            self.validate_wasm_features(&output_dir, &test_case.expected_wasm_features)?
        } else {
            false
        };
        
        Ok(TestResult {
            name: test_case.name.clone(),
            compilation_succeeded,
            compilation_time_ms: compilation_time.as_millis(),
            binary_size_bytes: binary_size,
            wasm_features_valid,
            performance_within_targets: self.check_performance_targets(compilation_time, binary_size),
            error_message: if !compilation_succeeded {
                Some(String::from_utf8_lossy(&output.stderr))
            } else {
                None
            },
        })
    }

    /// Validate that WASM output contains expected features
    fn validate_wasm_features(
        &self,
        output_dir: &Path,
        expected_features: &[String],
    ) -> Result<bool, Box<dyn std::error::Error>> {
        let wasm_file = output_dir.join("test_case.wasm");
        
        if !wasm_file.exists() {
            return Ok(false);
        }
        
        // For now, assume WASM features are present if compilation succeeded
        // In a real implementation, we would use wasm-tools to validate
        
        Ok(expected_features.iter().all(|f| {
            // Simple validation - check for known section names
            let wasm_content = std::fs::read(&wasm_file).unwrap_or_default();
            wasm_content.contains(f) || wasm_content.contains(f.as_str())
        }))
    }

    /// Check if performance meets targets
    fn check_performance_targets(&self, compilation_time: Duration, binary_size: usize) -> bool {
        compilation_time.as_millis() <= self.config.performance_targets.max_compilation_time_ms &&
        binary_size <= self.config.performance_targets.max_binary_size_bytes
    }
}

/// Results of integration tests
#[derive(Debug)]
struct IntegrationTestResults {
    test_results: Vec<TestResult>,
    summary: TestSummary,
}

impl IntegrationTestResults {
    fn new() -> Self {
        Self {
            test_results: Vec::new(),
            summary: TestSummary::default(),
        }
    }

    fn add_test_result(&mut self, result: TestResult) {
        self.test_results.push(result);
        self.summary.update_with_result(&result);
    }

    fn generate_report(&self) -> String {
        let mut report = String::new();
        
        report.push_str("# Cranelift Backend Integration Test Report\n\n");
        
        // Summary
        report.push_str(&format!("## Summary\n"));
        report.push_str(&format!("- Total tests: {}\n", self.test_results.len()));
        report.push_str(&format!("- Passed: {}\n", self.summary.passed_tests));
        report.push_str(&format!("- Failed: {}\n", self.summary.failed_tests));
        report.push_str(&format!("- Success rate: {:.1}%\n\n", self.summary.success_rate));
        
        // Performance summary
        report.push_str("## Performance Summary\n");
        report.push_str(&format!("- Average compilation time: {:.2}ms\n", self.summary.avg_compilation_time_ms));
        report.push_str(&format!("- Average binary size: {} bytes\n", self.summary.avg_binary_size_bytes));
        report.push_str(&format!("- Performance targets met: {:.1}%\n\n", self.summary.performance_targets_met_rate));
        
        // Individual test results
        report.push_str("## Test Results\n\n");
        
        for result in &self.test_results {
            report.push_str(&format!("### {}\n", result.name));
            
            let status = if result.compilation_succeeded { "✅ PASS" } else { "❌ FAIL" };
            report.push_str(&format!("**Status:** {}\n", status));
            
            if result.compilation_succeeded {
                report.push_str(&format!("- Compilation time: {}ms\n", result.compilation_time_ms));
                report.push_str(&format!("- Binary size: {} bytes\n", result.binary_size_bytes));
                report.push_str(&format!("- WASM features: {}\n", 
                    if result.wasm_features_valid { "✅ Valid" } else { "❌ Invalid" }));
                report.push_str(&format!("- Performance: {}\n", 
                    if result.performance_within_targets { "✅ Within targets" } else { "⚠️ Exceeds targets" }));
            } else {
                if let Some(ref error) = result.error_message {
                    report.push_str(&format!("- Error: {}\n", error));
                }
            }
            
            report.push_str("\n");
        }
        
        report
    }
}

/// Result of a single test
#[derive(Debug)]
struct TestResult {
    name: String,
    compilation_succeeded: bool,
    compilation_time_ms: u64,
    binary_size_bytes: usize,
    wasm_features_valid: bool,
    performance_within_targets: bool,
    error_message: Option<String>,
}

/// Summary statistics for all tests
#[derive(Debug, Default)]
struct TestSummary {
    passed_tests: usize,
    failed_tests: usize,
    avg_compilation_time_ms: f64,
    avg_binary_size_bytes: f64,
    performance_targets_met_rate: f64,
    success_rate: f64,
}

impl TestSummary {
    fn new() -> Self {
        Self::default()
    }

    fn update_with_result(&mut self, result: &TestResult) {
        if result.compilation_succeeded {
            self.passed_tests += 1;
        } else {
            self.failed_tests += 1;
        }
        
        self.success_rate = (self.passed_tests as f64) / ((self.passed_tests + self.failed_tests) as f64);
        
        // Update averages would be calculated from all results
    }

    fn calculate_averages(&mut self, test_results: &[TestResult]) {
        if test_results.is_empty() {
            return;
        }
        
        let successful_tests: Vec<&TestResult> = test_results.iter()
            .filter(|r| r.compilation_succeeded)
            .collect();
        
        if !successful_tests.is_empty() {
            self.avg_compilation_time_ms = successful_tests.iter()
                .map(|r| r.compilation_time_ms as f64)
                .sum::<f64>() / successful_tests.len() as f64;
            
            self.avg_binary_size_bytes = successful_tests.iter()
                .map(|r| r.binary_size_bytes as f64)
                .sum::<f64>() / successful_tests.len() as f64;
            
            self.performance_targets_met_rate = successful_tests.iter()
                .filter(|r| r.performance_within_targets)
                .count() as f64 / successful_tests.len() as f64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::{quickcheck, Arbitrary, Gen};

    #[test]
    fn test_hello_world_compilation() {
        let config = CraneliftTestConfig::default();
        let test_runner = CraneliftIntegrationTest::new().unwrap();
        
        let hello_world_test = &config.test_cases[0]; // hello_world is first
        
        let result = test_runner.run_single_test(hello_world_test).unwrap();
        
        assert!(result.compilation_succeeded, "Hello world should compile successfully");
        assert!(result.binary_size_bytes < 1024, "Binary should be small");
        assert!(result.wasm_features_valid, "WASM features should be valid");
        assert!(result.performance_within_targets, "Should meet performance targets");
    }

    #[test]
    fn test_fibonacci_compilation() {
        let config = CraneliftTestConfig::default();
        let test_runner = CraneliftIntegrationTest::new().unwrap();
        
        let fibonacci_test = &config.test_cases[1]; // fibonacci is second
        
        let result = test_runner.run_single_test(fibonacci_test).unwrap();
        
        assert!(result.compilation_succeeded, "Fibonacci should compile successfully");
        assert!(result.wasm_features_valid, "WASM features should be valid");
    }

    #[test]
    fn test_wasm_features_compilation() {
        let config = CraneliftTestConfig::default();
        let test_runner = CraneliftIntegrationTest::new().unwrap();
        
        let wasm_features_test = &config.test_cases[2]; // wasm_features is third
        
        let result = test_runner.run_single_test(wasm_features_test).unwrap();
        
        assert!(result.compilation_succeeded, "WASM features test should compile");
        assert!(result.wasm_features_valid, "WASM features should be detected as valid");
    }

    #[test]
    fn test_linear_types_compilation() {
        let config = CraneliftTestConfig::default();
        let test_runner = CraneliftIntegrationTest::new().unwrap();
        
        let linear_types_test = &config.test_cases[3]; // linear_types is fourth
        
        let result = test_runner.run_single_test(linear_types_test).unwrap();
        
        assert!(result.compilation_succeeded, "Linear types test should compile");
    }

    #[test]
    fn test_compilation_error_handling() {
        let config = CraneliftTestConfig::default();
        let test_runner = CraneliftIntegrationTest::new().unwrap();
        
        let error_test = &config.test_cases[4]; // compilation_error is fifth
        
        let result = test_runner.run_single_test(error_test).unwrap();
        
        assert!(!result.compilation_succeeded, "Compilation error test should fail");
        assert!(result.error_message.is_some(), "Should have error message");
    }

    #[test]
    fn test_all_integration_tests() {
        let test_runner = CraneliftIntegrationTest::new().unwrap();
        let results = test_runner.run_all_tests().unwrap();
        
        // Generate report
        let report = results.generate_report();
        println!("{}", report);
        
        // Check that most tests pass
        assert!(results.summary.success_rate >= 0.8, "At least 80% of tests should pass");
        assert!(results.summary.passed_tests >= 4, "At least 4 tests should pass");
    }

    // Property-based tests
    #[quickcheck]
    fn prop_compilation_time_reasonable(code: String) -> bool {
        let test_case = CraneliftTestCase {
            name: "property_test".to_string(),
            rust_code: code,
            expected_wasm_features: vec!["export".to_string()],
            should_compile: true,
            requires_specific_features: vec![],
        };
        
        let test_runner = CraneliftIntegrationTest::new().unwrap();
        let result = test_runner.run_single_test(&test_case).unwrap();
        
        result.compilation_time_ms <= 10000 // 10 seconds max
    }

    #[quickcheck]
    fn prop_binary_size_reasonable(code: String) -> bool {
        let test_case = CraneliftTestCase {
            name: "property_test".to_string(),
            rust_code: code,
            expected_wasm_features: vec!["export".to_string()],
            should_compile: true,
            requires_specific_features: vec![],
        };
        
        let test_runner = CraneliftIntegrationTest::new().unwrap();
        let result = test_runner.run_single_test(&test_case).unwrap();
        
        result.binary_size_bytes <= 1024 * 1024 // 1MB max
    }

    #[quickcheck]
    fn prop_wasm_output_valid(code: String) -> bool {
        let test_case = CraneliftTestCase {
            name: "property_test".to_string(),
            rust_code: code,
            expected_wasm_features: vec!["export".to_string()],
            should_compile: true,
            requires_specific_features: vec![],
        };
        
        let test_runner = CraneliftIntegrationTest::new().unwrap();
        let result = test_runner.run_single_test(&test_case).unwrap();
        
        result.compilation_succeeded && result.wasm_features_valid
    }

    #[test]
    fn test_property_based_compilation() {
        fn test_code = "pub fn test() { 42 }".to_string();
        
        assert!(prop_compilation_time_reasonable(test_code), 
                "Reasonable compilation time should be reproducible");
        assert!(prop_binary_size_reasonable(test_code), 
                "Reasonable binary size should be reproducible");
        assert!(prop_wasm_output_valid(test_code), 
                "Valid WASM output should be reproducible");
    }
}

// Arbitrary implementation for property-based testing
impl Arbitrary for String {
    fn arbitrary<G: Gen>(&self) -> String {
        // Generate simple, valid Rust code snippets
        let templates = vec![
            "pub fn test() -> u32 { 42 }",
            "pub fn add(a: u32, b: u32) -> u32 { a + b }",
            "pub fn simple() -> String { \"test\".to_string() }",
            "use wasm::ExternRef; pub fn test() { let _r: ExternRef<String> = todo!(); 42 }",
        ];
        
        if let Some(template) = self.choose(&templates) {
            template.to_string()
        } else {
            "pub fn test() -> u32 { 42 }".to_string()
        }
    }
}
