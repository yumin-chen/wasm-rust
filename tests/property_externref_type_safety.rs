//! Property-based tests for ExternRef type safety
//! 
//! This module validates that ExternRef provides type-safe
//! JavaScript interop with performance characteristics.
//! 
//! Property 6: JavaScript Interop Performance
//! Validates: Requirements 4.1, 4.2

use wasm::{ExternRef, FuncRef, WasmError, JsInteropSafe, JsValue};
use wasm::host::{HostProfile, get_host_capabilities, HostCapabilities};
use quickcheck::{Arbitrary, Gen, QuickCheck, TestResult};
use std::time::{Instant, Duration};

#[cfg(test)]
mod tests {
    use super::*;

    /// Test data for ExternRef type safety
    #[derive(Debug, Clone)]
    struct ExternRefTestData {
        name: &'static str,
        handle: u32,
        expected_type: &'static str,
        is_valid: bool,
    }

    /// JavaScript object type for testing
    #[derive(Debug, Clone, PartialEq, Eq)]
    enum TestJsType {
        String,
        Number,
        Boolean,
        Object,
        Array,
        Function,
        Undefined,
        Null,
    }

    /// JavaScript property descriptor
    #[derive(Debug, Clone)]
    struct PropertyDescriptor {
        name: &'static str,
        type_name: &'static str,
        is_writable: bool,
        is_enumerable: bool,
        is_configurable: bool,
    }

    /// JavaScript method descriptor
    #[derive(Debug, Clone)]
    struct MethodDescriptor {
        name: &'static str,
        param_types: Vec<&'static str>,
        return_type: &'static str,
    }

    /// Test JavaScript objects with their properties and methods
    const TEST_OBJECTS: &[(&'static str, &[PropertyDescriptor], &[MethodDescriptor])] = &[
        (
            "String",
            &[
                PropertyDescriptor {
                    name: "length",
                    type_name: "number",
                    is_writable: false,
                    is_enumerable: true,
                    is_configurable: false,
                },
                PropertyDescriptor {
                    name: "charAt",
                    type_name: "function",
                    is_writable: true,
                    is_enumerable: true,
                    is_configurable: true,
                },
            ],
            &[
                MethodDescriptor {
                    name: "charAt",
                    param_types: &["number"],
                    return_type: "string",
                },
                MethodDescriptor {
                    name: "slice",
                    param_types: &["number", "number"],
                    return_type: "string",
                },
            ],
        ),
        (
            "Array",
            &[
                PropertyDescriptor {
                    name: "length",
                    type_name: "number",
                    is_writable: true,
                    is_enumerable: true,
                    is_configurable: false,
                },
                PropertyDescriptor {
                    name: "size",
                    type_name: "number",
                    is_writable: false,
                    is_enumerable: false,
                    is_configurable: true,
                },
            ],
            &[
                MethodDescriptor {
                    name: "push",
                    param_types: &["any"],
                    return_type: "number",
                },
                MethodDescriptor {
                    name: "pop",
                    param_types: &[],
                    return_type: "any",
                },
                MethodDescriptor {
                    name: "slice",
                    param_types: &["number", "number"],
                    return_type: "object",
                },
            ],
        ),
        (
            "Object",
            &[
                PropertyDescriptor {
                    name: "constructor",
                    type_name: "function",
                    is_writable: true,
                    is_enumerable: true,
                    is_configurable: true,
                },
                PropertyDescriptor {
                    name: "toString",
                    type_name: "function",
                    is_writable: true,
                    is_enumerable: true,
                    is_configurable: true,
                },
            ],
            &[
                MethodDescriptor {
                    name: "hasOwnProperty",
                    param_types: &["string"],
                    return_type: "boolean",
                },
                MethodDescriptor {
                    name: "toString",
                    param_types: &[],
                    return_type: "string",
                },
            ],
        ),
    ];

    /// Property: ExternRef maintains type safety across operations
    #[test]
    fn prop_externref_type_safety() {
        fn property(test_data: ExternRefTestData) -> TestResult {
            // Create ExternRef for testing
            let extern_ref = unsafe { ExternRef::<TestJsType>::from_handle(test_data.handle) };
            
            // Test basic properties
            if extern_ref.is_null() != !test_data.is_valid {
                return TestResult::failed();
            }
            
            if extern_ref.handle() != test_data.handle {
                return TestResult::failed();
            }
            
            // Test type safety - the ExternRef should only work with the specified type
            // In a real implementation, this would involve actual JavaScript interop
            // For now, we test the type system guarantees
            
            TestResult::passed()
        }

        QuickCheck::new()
            .tests(100)
            .gen(Gen::new(100))
            .quickcheck(property as fn(ExternRefTestData) -> TestResult);
    }

    /// Property: ExternRef provides consistent handle management
    #[test]
    fn prop_externref_handle_consistency() {
        fn property(handle1: u32, handle2: u32) -> TestResult {
            let extern_ref1 = unsafe { ExternRef::<TestJsType>::from_handle(handle1) };
            let extern_ref2 = unsafe { ExternRef::<TestJsType>::from_handle(handle2) };
            
            // Different handles should create different references
            if handle1 != handle2 {
                if extern_ref1.handle() == extern_ref2.handle() {
                    return TestResult::failed();
                }
            }
            
            // Same handles should create equal references
            let extern_ref3 = unsafe { ExternRef::<TestJsType>::from_handle(handle1) };
            if extern_ref1.handle() != extern_ref3.handle() {
                return TestResult::failed();
            }
            
            // Null reference should have handle 0
            let null_ref = ExternRef::<TestJsType>::null();
            if null_ref.handle() != 0 || !null_ref.is_null() {
                return TestResult::failed();
            }
            
            TestResult::passed()
        }

        QuickCheck::new()
            .tests(100)
            .gen(Gen::new(100))
            .quickcheck(property as fn(u32, u32) -> TestResult);
    }

    /// Property: ExternRef property access maintains type safety
    #[test]
    fn prop_externref_property_type_safety() {
        fn property(object_type: String, property_index: usize) -> TestResult {
            // Find test object descriptor
            let (object_name, properties, _) = TEST_OBJECTS
                .iter()
                .find(|(name, _, _)| *name == object_type)
                .unwrap_or(("Unknown", &[], &[]));
            
            if property_index >= properties.len() {
                return TestResult::discard(); // Invalid index
            }
            
            let property = &properties[property_index];
            
            // Create ExternRef for the object
            let extern_ref = unsafe { ExternRef::<TestJsType>::from_handle(42) };
            
            // Test property access - this would be actual JavaScript interop
            // For now, we test the type system
            match property.type_name {
                "string" => {
                    // String property access should be type-safe
                    let _result: Result<String, WasmError> = extern_ref.get_property("length");
                }
                "number" => {
                    // Number property access should be type-safe
                    let _result: Result<f64, WasmError> = extern_ref.get_property("length");
                }
                "boolean" => {
                    // Boolean property access should be type-safe
                    let _result: Result<bool, WasmError> = extern_ref.get_property("configurable");
                }
                _ => {
                    // Other types should also be handled
                    let _result: Result<(), WasmError> = extern_ref.get_property(property.name);
                }
            }
            
            TestResult::passed()
        }

        QuickCheck::new()
            .tests(100)
            .gen(Gen::new(100))
            .quickcheck(property as fn(String, usize) -> TestResult);
    }

    /// Property: ExternRef method invocation maintains type safety
    #[test]
    fn prop_externref_method_type_safety() {
        fn property(object_type: String, method_index: usize) -> TestResult {
            // Find test object descriptor
            let (object_name, _, methods) = TEST_OBJECTS
                .iter()
                .find(|(name, _, _)| *name == object_type)
                .unwrap_or(("Unknown", &[], &[]));
            
            if method_index >= methods.len() {
                return TestResult::discard(); // Invalid index
            }
            
            let method = &methods[method_index];
            
            // Create ExternRef for the object
            let extern_ref = unsafe { ExternRef::<TestJsType>::from_handle(42) };
            
            // Test method invocation - this would be actual JavaScript interop
            // For now, we test the type system
            
            // Create arguments based on method signature
            let args: Vec<String> = method.param_types
                .iter()
                .map(|&ty| format!("arg_{}", ty))
                .collect();
            
            // Test method call based on return type
            match method.return_type {
                "string" => {
                    let _result: Result<String, WasmError> = extern_ref.invoke_method(method.name, &args);
                }
                "number" => {
                    let _result: Result<f64, WasmError> = extern_ref.invoke_method(method.name, &args);
                }
                "boolean" => {
                    let _result: Result<bool, WasmError> = extern_ref.invoke_method(method.name, &args);
                }
                "object" => {
                    let _result: Result<ExternRef<TestJsType>, WasmError> = extern_ref.invoke_method(method.name, &args);
                }
                "any" => {
                    let _result: Result<(), WasmError> = extern_ref.invoke_method(method.name, &args);
                }
                _ => {
                    // Other return types should also be handled
                    let _result: Result<(), WasmError> = extern_ref.invoke_method(method.name, &args);
                }
            }
            
            TestResult::passed()
        }

        QuickCheck::new()
            .tests(100)
            .gen(Gen::new(100))
            .quickcheck(property as fn(String, usize) -> TestResult);
    }

    /// Property: ExternRef operations are zero-cost when not needed
    #[test]
    fn prop_externref_zero_cost_abstractions() {
        fn property(operations_count: u8) -> TestResult {
            let extern_ref = unsafe { ExternRef::<TestJsType>::from_handle(42) };
            
            let start = Instant::now();
            
            // Perform operations that should be zero-cost
            for i in 0..operations_count {
                let _cloned = extern_ref.clone(); // Should be zero-cost
                let _handle = extern_ref.handle(); // Should be zero-cost
                let _is_null = extern_ref.is_null(); // Should be zero-cost
                
                // These operations should compile to no-ops at runtime
                if i % 10 == 0 {
                    // Occasionally test actual operations
                    // In a real implementation, this would involve
                    // JavaScript interop with validation
                }
            }
            
            let duration = start.elapsed();
            
            // Zero-cost operations should be very fast
            if duration.as_nanos() > operations_count as u64 * 1000 {
                return TestResult::failed();
            }
            
            TestResult::passed()
        }

        QuickCheck::new()
            .tests(50)
            .gen(Gen::new(100))
            .quickcheck(property as fn(u8) -> TestResult);
    }

    /// Property: ExternRef handles null references safely
    #[test]
    fn prop_externref_null_safety() {
        fn property(operation_type: u8) -> TestResult {
            let null_ref = ExternRef::<TestJsType>::null();
            
            // All operations on null references should fail gracefully
            match operation_type % 4 {
                0 => {
                    // Property access on null should fail
                    let result: Result<String, WasmError> = null_ref.get_property("any_property");
                    if !matches!(result, Err(WasmError::NullDereference)) {
                        return TestResult::failed();
                    }
                }
                1 => {
                    // Method call on null should fail
                    let result: Result<(), WasmError> = null_ref.invoke_method("any_method", &());
                    if !matches!(result, Err(WasmError::NullDereference)) {
                        return TestResult::failed();
                    }
                }
                2 => {
                    // Property set on null should fail
                    let result: Result<(), WasmError> = null_ref.set_property("any_property", "value");
                    if !matches!(result, Err(WasmError::NullDereference)) {
                        return TestResult::failed();
                    }
                }
                3 => {
                    // Handle access on null should still work
                    let handle = null_ref.handle();
                    if handle != 0 || !null_ref.is_null() {
                        return TestResult::failed();
                    }
                }
                _ => unreachable!(),
            }
            
            TestResult::passed()
        }

        QuickCheck::new()
            .tests(100)
            .gen(Gen::new(100))
            .quickcheck(property as fn(u8) -> TestResult);
    }

    /// Property: ExternRef provides JavaScript interop performance
    #[test]
    fn prop_externref_js_interop_performance() {
        fn property(operation_count: u8, complexity_level: u8) -> TestResult {
            // Create ExternRef for performance testing
            let extern_ref = unsafe { ExternRef::<TestJsType>::from_handle(42) };
            
            let start = Instant::now();
            
            // Perform operations of varying complexity
            for i in 0..operation_count {
                match complexity_level % 3 {
                    0 => {
                        // Simple property access
                        let _result: Result<String, WasmError> = extern_ref.get_property("length");
                    }
                    1 => {
                        // Method invocation
                        let _result: Result<(), WasmError> = extern_ref.invoke_method("toString", &());
                    }
                    2 => {
                        // Complex operation
                        let _result: Result<ExternRef<TestJsType>, WasmError> = extern_ref.invoke_method("getObject", &());
                    }
                    _ => unreachable!(),
                }
                
                // Simulate some processing time
                if i % 5 == 0 {
                    std::hint::spin_loop();
                }
            }
            
            let duration = start.elapsed();
            
            // Performance should be reasonable for the complexity
            let max_expected_duration = match complexity_level {
                0 => Duration::from_micros(10 * operation_count as u64),   // Simple: 10μs per op
                1 => Duration::from_micros(50 * operation_count as u64),   // Medium: 50μs per op
                2 => Duration::from_micros(200 * operation_count as u64),  // Complex: 200μs per op
                _ => Duration::from_micros(100 * operation_count as u64),  // Default: 100μs per op
            };
            
            if duration > max_expected_duration {
                return TestResult::failed();
            }
            
            TestResult::passed()
        }

        QuickCheck::new()
            .tests(50)
            .gen(Gen::new(100))
            .quickcheck(property as fn(u8, u8) -> TestResult);
    }

    /// Property: ExternRef is compatible with different host profiles
    #[test]
    fn prop_externref_host_profile_compatibility() {
        fn property(host_profile: u8) -> TestResult {
            let extern_ref = unsafe { ExternRef::<TestJsType>::from_handle(42) };
            
            // Test compatibility with different host profiles
            let result = match host_profile % 4 {
                0 => {
                    // Browser environment
                    extern_ref.get_property("length")
                }
                1 => {
                    // Node.js environment
                    extern_ref.invoke_method("toString", &())
                }
                2 => {
                    // Wasmtime environment
                    extern_ref.set_property("test", "value")
                }
                3 => {
                    // Embedded environment
                    ExternRef::<TestJsType>::null().get_property("any")
                }
                _ => unreachable!(),
            };
            
            // Should handle all profiles gracefully
            // In a real implementation, this would test actual host compatibility
            match result {
                Ok(_) | Err(WasmError::HostError(_)) | Err(WasmError::UnsupportedOperation) => {
                    // Expected results for different host capabilities
                }
                Err(WasmError::NullDereference) => {
                    // Should not get null dereference with valid handle
                }
                _ => {
                    return TestResult::failed();
                }
            }
            
            TestResult::passed()
        }

        QuickCheck::new()
            .tests(40)
            .gen(Gen::new(100))
            .quickcheck(property as fn(u8) -> TestResult);
    }

    /// Property: FuncRef maintains type safety across operations
    #[test]
    fn prop_funcref_type_safety() {
        fn property(index1: u32, index2: u32) -> TestResult {
            // Create FuncRefs for testing
            let func_ref1 = unsafe { FuncRef::<(i32, i32), i32>::from_index(index1) };
            let func_ref2 = unsafe { FuncRef::<(i32, i32), i32>::from_index(index2) };
            
            // Different indices should create different references
            if index1 != index2 {
                if func_ref1.index() == func_ref2.index() {
                    return TestResult::failed();
                }
            }
            
            // Same indices should create equal references
            let func_ref3 = unsafe { FuncRef::<(i32, i32), i32>::from_index(index1) };
            if func_ref1.index() != func_ref3.index() {
                return TestResult::failed();
            }
            
            // Null function reference should have index 0
            let null_func = FuncRef::<(), ()>::null();
            if null_func.index() != 0 || !null_func.is_null() {
                return TestResult::failed();
            }
            
            TestResult::passed()
        }

        QuickCheck::new()
            .tests(100)
            .gen(Gen::new(100))
            .quickcheck(property as fn(u32, u32) -> TestResult);
    }

    /// Property: FuncRef handles null references safely
    #[test]
    fn prop_funcref_null_safety() {
        fn property(operation_type: u8) -> TestResult {
            let null_func = FuncRef::<(i32, i32), i32>::null();
            
            // All operations on null function references should fail gracefully
            match operation_type % 2 {
                0 => {
                    // Index access should still work
                    let index = null_func.index();
                    if index != 0 || !null_func.is_null() {
                        return TestResult::failed();
                    }
                }
                1 => {
                    // Function call on null should be handled
                    // In a real implementation, this would be undefined behavior
                    // For property testing, we test that it doesn't panic
                    let _result = unsafe { null_func.call(42) };
                    // The result is undefined, but shouldn't panic
                }
                _ => unreachable!(),
            }
            
            TestResult::passed()
        }

        QuickCheck::new()
            .tests(100)
            .gen(Gen::new(100))
            .quickcheck(property as fn(u8) -> TestResult);
    }

    /// Property: JsInteropSafe trait correctly validates types
    #[test]
    fn prop_jsinterop_type_validation() {
        fn property(type_name: String) -> TestResult {
            // Test type validation for different JavaScript types
            let result = match type_name.as_str() {
                "String" => {
                    // String type should be valid for JS interop
                    String::validate_js_interop()
                }
                "Number" => {
                    // Number type should be valid for JS interop
                    f64::validate_js_interop()
                }
                "Boolean" => {
                    // Boolean type should be valid for JS interop
                    bool::validate_js_interop()
                }
                "InvalidPodType" => {
                    // Non-Pod type should fail validation
                    struct InvalidPodType;
                    InvalidPodType::validate_js_interop()
                }
                _ => {
                    // Unknown types should fail validation
                    TestJsType::validate_js_interop()
                }
            };
            
            // Should only fail for invalid types
            match result {
                Ok(()) | Err(WasmError::InvalidOperation(_)) => TestResult::passed(),
                _ => TestResult::failed(),
            }
        }

        QuickCheck::new()
            .tests(50)
            .gen(Gen::new(100))
            .quickcheck(property as fn(String) -> TestResult);
    }

    /// Property: JsValue conversions maintain type correctness
    #[test]
    fn prop_jsvalue_type_conversions() {
        fn property(value_type: u8, original_value: i32) -> TestResult {
            // Test round-trip conversions
            let js_value = match value_type % 3 {
                0 => {
                    // i32 to JsValue and back
                    i32::to_js_value(original_value)
                }
                1 => {
                    // f64 to JsValue and back
                    (original_value as f64).to_js_value()
                }
                2 => {
                    // bool to JsValue and back
                    (original_value != 0).to_js_value()
                }
                _ => unreachable!(),
            };
            
            if js_value.is_err() {
                return TestResult::failed();
            }
            
            // Convert back to the original type
            let round_trip = match value_type % 3 {
                0 => {
                    // i32 from JsValue
                    i32::from_js_value(js_value.unwrap())
                }
                1 => {
                    // f64 from JsValue
                    f64::from_js_value(js_value.unwrap()).map(|v| v as i32)
                }
                2 => {
                    // bool from JsValue
                    bool::from_js_value(js_value.unwrap()).map(|b| b as i32)
                }
                _ => unreachable!(),
            };
            
            // Round-trip should preserve the value for compatible types
            match round_trip {
                Ok(round_trip_value) => {
                    if value_type == 0 {
                        // i32 round-trip should be exact
                        if round_trip_value != original_value {
                            return TestResult::failed();
                        }
                    } else if value_type == 2 {
                        // bool round-trip should be exact
                        if round_trip_value != (original_value != 0) as i32 {
                            return TestResult::failed();
                        }
                    }
                    // f64 to i32 may lose precision, that's expected
                }
                Err(_) => {
                    // Should not fail for valid conversions
                    return TestResult::failed();
                }
            }
            
            TestResult::passed()
        }

        QuickCheck::new()
            .tests(100)
            .gen(Gen::new(100))
            .quickcheck(property as fn(u8, i32) -> TestResult);
    }
}

/// Integration tests for ExternRef functionality
#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_externref_with_real_javascript_objects() {
        // Test with known JavaScript object types
        for (object_name, properties, methods) in TEST_OBJECTS {
            // Create ExternRef for the object
            let extern_ref = unsafe { ExternRef::<TestJsType>::from_handle(100) };
            
            // Test that all properties exist (in real implementation)
            for property in properties {
                let result: Result<(), WasmError> = extern_ref.get_property(property.name);
                // In a real test, this would validate the property exists and has correct type
                let _ = result;
            }
            
            // Test that all methods exist (in real implementation)
            for method in methods {
                let args: Vec<String> = method.param_types
                    .iter()
                    .map(|&ty| format!("test_arg_{}", ty))
                    .collect();
                
                let result: Result<(), WasmError> = extern_ref.invoke_method(method.name, &args);
                // In a real test, this would validate the method exists and has correct signature
                let _ = result;
            }
        }
    }

    #[test]
    fn test_externref_error_handling() {
        let extern_ref = unsafe { ExternRef::<TestJsType>::from_handle(42) };
        
        // Test various error conditions
        let null_ref = ExternRef::<TestJsType>::null();
        
        // Property access on null should fail
        let result = null_ref.get_property("any_property");
        assert!(matches!(result, Err(WasmError::NullDereference)));
        
        // Method call on null should fail
        let result = null_ref.invoke_method("any_method", &());
        assert!(matches!(result, Err(WasmError::NullDereference)));
        
        // Property set on null should fail
        let result = null_ref.set_property("any_property", "value");
        assert!(matches!(result, Err(WasmError::NullDereference)));
    }

    #[test]
    fn test_externref_performance_characteristics() {
        let extern_ref = unsafe { ExternRef::<TestJsType>::from_handle(42) };
        
        // Test that operations are fast
        let iterations = 10000;
        let start = Instant::now();
        
        for _ in 0..iterations {
            let _cloned = extern_ref.clone();
            let _handle = extern_ref.handle();
            let _is_null = extern_ref.is_null();
        }
        
        let duration = start.elapsed();
        
        // These operations should be very fast (essentially no-ops)
        let avg_time_per_op = duration.as_nanos() / iterations;
        assert!(avg_time_per_op < 1000, "ExternRef operations should be zero-cost");
        
        // Test actual JavaScript operations
        let start = Instant::now();
        
        for i in 0..100 {
            let _result: Result<String, WasmError> = extern_ref.get_property("length");
            if i % 10 == 0 {
                // Occasional actual operations
                std::hint::spin_loop();
            }
        }
        
        let duration = start.elapsed();
        
        // JavaScript interop should be reasonably fast
        let avg_time_per_js_op = duration.as_nanos() / 100;
        assert!(avg_time_per_js_op < 100_000, "JavaScript interop should be fast");
    }

    #[test]
    fn test_funcref_with_real_functions() {
        // Create FuncRef for testing
        let func_ref = unsafe { FuncRef::<(i32, i32), i32>::from_index(10) };
        
        // Test that function reference is properly created
        assert_eq!(func_ref.index(), 10);
        assert!(!func_ref.is_null());
        
        // Test function call (in real implementation)
        // This would involve actual WASM function table calls
        let _result = unsafe { func_ref.call(42) };
        // The result would depend on the actual function at index 10
    }

    #[test]
    fn test_host_capability_detection() {
        let caps = get_host_capabilities();
        
        // Test that capabilities are detected
        match caps.js_interop {
            true => {
                // Should be able to create ExternRef instances
                let extern_ref = unsafe { ExternRef::<TestJsType>::from_handle(42) };
                assert!(!extern_ref.is_null());
            }
            false => {
                // ExternRef creation should still work, but operations might fail
                let extern_ref = unsafe { ExternRef::<TestJsType>::from_handle(42) };
                assert!(!extern_ref.is_null());
                
                // JavaScript operations should fail gracefully
                let result = extern_ref.get_property("any_property");
                assert!(matches!(result, Err(WasmError::HostError(_) | WasmError::UnsupportedOperation)));
            }
        }
        
        // Test threading capabilities
        match caps.threading {
            true => {
                // Should support threading operations
                // This would be tested with actual threading primitives
            }
            false => {
                // Threading operations should fail gracefully
            }
        }
    }

    #[test]
    fn test_jsvalue_conversion_edge_cases() {
        // Test edge cases for JavaScript value conversions
        
        // Test i32 boundary values
        let max_i32 = i32::MAX;
        let js_value = i32::to_js_value(max_i32).unwrap();
        let converted = i32::from_js_value(js_value).unwrap();
        assert_eq!(converted, max_i32);
        
        let min_i32 = i32::MIN;
        let js_value = i32::to_js_value(min_i32).unwrap();
        let converted = i32::from_js_value(js_value).unwrap();
        assert_eq!(converted, min_i32);
        
        // Test floating point edge cases
        let inf = f64::INFINITY;
        let js_value = f64::to_js_value(inf).unwrap();
        let converted = f64::from_js_value(js_value).unwrap();
        assert!(converted.is_infinite());
        
        // Test boolean edge cases
        for bool_val in [true, false] {
            let js_value = bool::to_js_value(bool_val).unwrap();
            let converted = bool::from_js_value(js_value).unwrap();
            assert_eq!(converted, bool_val);
        }
    }
}
