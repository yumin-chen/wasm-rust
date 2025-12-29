//! Component model support for WasmRust
//! 
//! This module provides abstractions for the WebAssembly Component Model,
//! including type-safe interfaces, component instantiation, and
//! inter-component communication.

use crate::host::{get_host_capabilities, HostCapabilities};
use crate::wasmir::{WasmIR, Signature, Type, Instruction, Terminator};
use core::ptr::NonNull;
use core::marker::PhantomData;
use alloc::vec::Vec;

/// Component interface definition
/// 
/// `ComponentInterface` describes the public interface of a WebAssembly
/// component, including exported functions and their signatures.
pub struct ComponentInterface {
    /// Component name
    pub name: String,
    /// Exported functions
    pub exports: Vec<FunctionExport>,
    /// Imported functions
    pub imports: Vec<FunctionImport>,
    /// Component version
    pub version: String,
    /// Component metadata
    pub metadata: ComponentMetadata,
}

/// Function export from a component
#[derive(Debug, Clone)]
pub struct FunctionExport {
    /// Export name
    pub name: String,
    /// Function signature
    pub signature: Signature,
    /// Export attributes
    pub attributes: Vec<ExportAttribute>,
    /// Documentation
    pub docs: Option<String>,
}

/// Function import required by a component
#[derive(Debug, Clone)]
pub struct FunctionImport {
    /// Import name
    pub name: String,
    /// Function signature
    pub signature: Signature,
    /// Import attributes
    pub attributes: Vec<ImportAttribute>,
    /// Required interface
    pub interface: Option<String>,
}

/// Export attributes for component functions
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExportAttribute {
    /// Function is public (default)
    Public,
    /// Function is private to the component
    Private,
    /// Function is exported with specific name
    Named(String),
    /// Function has specific access level
    Access(AccessLevel),
}

/// Import attributes for component functions
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImportAttribute {
    /// Import is optional
    Optional,
    /// Import is required
    Required,
    /// Import has specific binding
    Binding(String),
    /// Import from specific interface
    Interface(String),
}

/// Access levels for component functions
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccessLevel {
    /// Public access (default)
    Public,
    /// Internal to component
    Internal,
    /// Private to module
    Private,
    /// Protected access
    Protected,
}

/// Component metadata
#[derive(Debug, Clone)]
pub struct ComponentMetadata {
    /// Component author
    pub author: Option<String>,
    /// Component description
    pub description: Option<String>,
    /// Component license
    pub license: Option<String>,
    /// Component keywords
    pub keywords: Vec<String>,
    /// Component repository
    pub repository: Option<String>,
    /// Component homepage
    pub homepage: Option<String>,
}

impl ComponentInterface {
    /// Creates a new component interface
    pub fn new(name: String) -> Self {
        Self {
            name,
            exports: Vec::new(),
            imports: Vec::new(),
            version: "1.0.0".to_string(),
            metadata: ComponentMetadata {
                author: None,
                description: None,
                license: None,
                keywords: Vec::new(),
                repository: None,
                homepage: None,
            },
        }
    }

    /// Adds a function export
    pub fn add_export(&mut self, export: FunctionExport) {
        self.exports.push(export);
    }

    /// Adds a function import
    pub fn add_import(&mut self, import: FunctionImport) {
        self.imports.push(import);
    }

    /// Gets an export by name
    pub fn get_export(&self, name: &str) -> Option<&FunctionExport> {
        self.exports.iter().find(|e| e.name == name)
    }

    /// Gets an import by name
    pub fn get_import(&self, name: &str) -> Option<&FunctionImport> {
        self.imports.iter().find(|i| i.name == name)
    }

    /// Validates the component interface
    pub fn validate(&self) -> Result<(), ComponentError> {
        // Check for duplicate export names
        let mut export_names = std::collections::HashSet::new();
        for export in &self.exports {
            if export_names.contains(&export.name) {
                return Err(ComponentError::DuplicateExport(export.name.clone()));
            }
            export_names.insert(export.name.clone());
        }

        // Check for duplicate import names
        let mut import_names = std::collections::HashSet::new();
        for import in &self.imports {
            if import_names.contains(&import.name) {
                return Err(ComponentError::DuplicateImport(import.name.clone()));
            }
            import_names.insert(import.name.clone());
        }

        // Validate export signatures
        for export in &self.exports {
            self.validate_signature(&export.signature)?;
        }

        // Validate import signatures
        for import in &self.imports {
            self.validate_signature(&import.signature)?;
        }

        Ok(())
    }

    /// Validates a function signature
    fn validate_signature(&self, signature: &Signature) -> Result<(), ComponentError> {
        // Validate parameter types
        for param_type in &signature.params {
            self.validate_type(param_type)?;
        }

        // Validate return type
        if let Some(return_type) = &signature.returns {
            self.validate_type(return_type)?;
        }

        Ok(())
    }

    /// Validates a type
    fn validate_type(&self, type_ref: &Type) -> Result<(), ComponentError> {
        match type_ref {
            Type::I32 | Type::I64 | Type::F32 | Type::F64 | Type::Void => Ok(()),
            Type::Ref(ty) => {
                // Check reference type name
                if ty.is_empty() {
                    return Err(ComponentError::InvalidType("Empty reference type".to_string()));
                }
                Ok(())
            }
            Type::Array { element_type, .. } => {
                self.validate_type(element_type)
            }
            Type::Struct { fields } => {
                for field_type in fields {
                    self.validate_type(field_type)?;
                }
                Ok(())
            }
            Type::Pointer(target_type) => {
                self.validate_type(target_type)
            }
            Type::Linear { inner_type } => {
                self.validate_type(inner_type)
            }
            Type::Capability { inner_type, .. } => {
                self.validate_type(inner_type)
            }
        }
    }
}

/// Component instance
/// 
/// `ComponentInstance` represents a running instance of a WebAssembly component
/// with its imported and exported functions bound.
pub struct ComponentInstance {
    /// Component interface
    pub interface: ComponentInterface,
    /// Component handle
    handle: NonNull<()>,
    /// Instance state
    state: ComponentState,
    /// Export table
    export_table: Vec<FunctionExport>,
    /// Import table
    import_table: Vec<FunctionImport>,
}

/// Component state
#[derive(Debug, Clone)]
pub enum ComponentState {
    /// Component is initialized but not started
    Initialized,
    /// Component is running
    Running,
    /// Component is paused
    Paused,
    /// Component is stopped
    Stopped,
    /// Component has error
    Error(String),
}

impl ComponentInstance {
    /// Creates a new component instance
    pub fn new(interface: ComponentInterface) -> Result<Self, ComponentError> {
        // Validate interface first
        interface.validate()?;

        // Check component model support
        let caps = get_host_capabilities();
        if !caps.component_model {
            return Err(ComponentError::ComponentModelNotSupported);
        }

        // Create component instance
        let handle = Self::create_component_instance(&interface)?;
        let mut instance = Self {
            interface,
            handle: NonNull::dangling(),
            state: ComponentState::Initialized,
            export_table: Vec::new(),
            import_table: Vec::new(),
        };

        // Set up export table
        instance.setup_export_table()?;

        // Set up import table
        instance.setup_import_table()?;

        Ok(instance)
    }

    /// Starts the component instance
    pub fn start(&mut self) -> Result<(), ComponentError> {
        match self.state {
            ComponentState::Initialized | ComponentState::Stopped => {
                // Start the component
                self.start_component_internal()?;
                self.state = ComponentState::Running;
                Ok(())
            }
            ComponentState::Running => {
                Err(ComponentError::AlreadyRunning)
            }
            ComponentState::Paused => {
                // Resume from paused
                self.resume_component_internal()?;
                self.state = ComponentState::Running;
                Ok(())
            }
            ComponentState::Error(_) => {
                Err(ComponentError::InstanceError)
            }
        }
    }

    /// Stops the component instance
    pub fn stop(&mut self) -> Result<(), ComponentError> {
        match self.state {
            ComponentState::Running | ComponentState::Paused => {
                // Stop the component
                self.stop_component_internal()?;
                self.state = ComponentState::Stopped;
                Ok(())
            }
            ComponentState::Stopped => {
                Err(ComponentError::AlreadyStopped)
            }
            ComponentState::Initialized => {
                Err(ComponentError::NotStarted)
            }
            ComponentState::Error(_) => {
                Err(ComponentError::InstanceError)
            }
        }
    }

    /// Pauses the component instance
    pub fn pause(&mut self) -> Result<(), ComponentError> {
        match self.state {
            ComponentState::Running => {
                // Pause the component
                self.pause_component_internal()?;
                self.state = ComponentState::Paused;
                Ok(())
            }
            ComponentState::Paused => {
                Err(ComponentError::AlreadyPaused)
            }
            _ => {
                Err(ComponentError::CannotPause)
            }
        }
    }

    /// Gets the current state
    pub fn state(&self) -> &ComponentState {
        &self.state
    }

    /// Calls an exported function
    pub fn call_export<Args, Ret>(
        &self,
        name: &str,
        args: Args,
    ) -> Result<Ret, ComponentError>
    where
        Args: 'static,
        Ret: 'static,
    {
        // Find the export
        let export = self.get_export_by_name(name)
            .ok_or_else(|| ComponentError::ExportNotFound(name.to_string()))?;

        // Call the function
        self.call_function_internal(&export.signature, args)
    }

    /// Gets an export by name
    fn get_export_by_name(&self, name: &str) -> Option<&FunctionExport> {
        self.export_table.iter().find(|e| e.name == name)
    }

    /// Sets up the export table
    fn setup_export_table(&mut self) -> Result<(), ComponentError> {
        self.export_table = self.interface.exports.clone();
        Ok(())
    }

    /// Sets up the import table
    fn setup_import_table(&mut self) -> Result<(), ComponentError> {
        self.import_table = self.interface.imports.clone();
        Ok(())
    }

    /// Creates the component instance
    fn create_component_instance(interface: &ComponentInterface) -> Result<NonNull<()>, ComponentError> {
        // In a real implementation, this would:
        // 1. Load the component module
        // 2. Validate imports
        // 3. Allocate instance memory
        // 4. Initialize the component
        // For now, return a dummy handle
        Ok(NonNull::dangling())
    }

    /// Starts the component internally
    fn start_component_internal(&self) -> Result<(), ComponentError> {
        // In a real implementation, this would:
        // 1. Initialize the component runtime
        // 2. Start any background tasks
        // 3. Set up event handlers
        Ok(())
    }

    /// Stops the component internally
    fn stop_component_internal(&self) -> Result<(), ComponentError> {
        // In a real implementation, this would:
        // 1. Stop background tasks
        // 2. Clean up event handlers
        // 3. Release resources
        Ok(())
    }

    /// Pauses the component internally
    fn pause_component_internal(&self) -> Result<(), ComponentError> {
        // In a real implementation, this would:
        // 1. Pause background tasks
        // 2. Save current state
        Ok(())
    }

    /// Resumes the component internally
    fn resume_component_internal(&self) -> Result<(), ComponentError> {
        // In a real implementation, this would:
        // 1. Resume background tasks
        // 2. Restore saved state
        Ok(())
    }

    /// Calls a function internally
    fn call_function_internal<Args, Ret>(
        &self,
        _signature: &Signature,
        args: Args,
    ) -> Result<Ret, ComponentError> {
        // In a real implementation, this would:
        // 1. Validate argument types
        // 2. Convert arguments to WASM format
        // 3. Call the function through WASM interface
        // 4. Convert result back to Rust type
        // For now, this is a placeholder
        Err(ComponentError::NotImplemented)
    }
}

impl Drop for ComponentInstance {
    fn drop(&mut self) {
        // Clean up component resources
        if let ComponentState::Running = self.state {
            let _ = self.stop();
        }
    }
}

/// Component-related errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComponentError {
    /// Component model not supported
    ComponentModelNotSupported,
    /// Duplicate export name
    DuplicateExport(String),
    /// Duplicate import name
    DuplicateImport(String),
    /// Invalid type
    InvalidType(String),
    /// Export not found
    ExportNotFound(String),
    /// Import not found
    ImportNotFound(String),
    /// Component already running
    AlreadyRunning,
    /// Component already stopped
    AlreadyStopped,
    /// Component already paused
    AlreadyPaused,
    /// Component cannot be paused
    CannotPause,
    /// Component not started
    NotStarted,
    /// Component instance error
    InstanceError,
    /// Feature not implemented
    NotImplemented,
    /// Validation failed
    ValidationFailed(String),
}

impl core::fmt::Display for ComponentError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ComponentError::ComponentModelNotSupported => {
                write!(f, "Component model not supported")
            }
            ComponentError::DuplicateExport(name) => {
                write!(f, "Duplicate export: {}", name)
            }
            ComponentError::DuplicateImport(name) => {
                write!(f, "Duplicate import: {}", name)
            }
            ComponentError::InvalidType(ty) => {
                write!(f, "Invalid type: {}", ty)
            }
            ComponentError::ExportNotFound(name) => {
                write!(f, "Export not found: {}", name)
            }
            ComponentError::ImportNotFound(name) => {
                write!(f, "Import not found: {}", name)
            }
            ComponentError::AlreadyRunning => {
                write!(f, "Component is already running")
            }
            ComponentError::AlreadyStopped => {
                write!(f, "Component is already stopped")
            }
            ComponentError::AlreadyPaused => {
                write!(f, "Component is already paused")
            }
            ComponentError::CannotPause => {
                write!(f, "Component cannot be paused")
            }
            ComponentError::NotStarted => {
                write!(f, "Component has not been started")
            }
            ComponentError::InstanceError => {
                write!(f, "Component instance error")
            }
            ComponentError::NotImplemented => {
                write!(f, "Feature not implemented")
            }
            ComponentError::ValidationFailed(msg) => {
                write!(f, "Validation failed: {}", msg)
            }
        }
    }
}

/// Component factory for creating instances
pub struct ComponentFactory {
    /// Component registry
    registry: Vec<ComponentInterface>,
    /// Default capabilities
    default_capabilities: HostCapabilities,
}

impl ComponentFactory {
    /// Creates a new component factory
    pub fn new() -> Self {
        Self {
            registry: Vec::new(),
            default_capabilities: get_host_capabilities(),
        }
    }

    /// Registers a component interface
    pub fn register_component(&mut self, interface: ComponentInterface) -> Result<(), ComponentError> {
        // Validate the interface
        interface.validate()?;

        // Check for duplicates
        if self.registry.iter().any(|c| c.name == interface.name) {
            return Err(ComponentError::DuplicateExport(interface.name));
        }

        self.registry.push(interface);
        Ok(())
    }

    /// Creates a component instance
    pub fn create_instance(
        &self,
        name: &str,
    ) -> Result<ComponentInstance, ComponentError> {
        // Find the component interface
        let interface = self.registry
            .iter()
            .find(|c| c.name == name)
            .ok_or_else(|| ComponentError::ExportNotFound(name.to_string()))?;

        // Create an instance
        ComponentInstance::new(interface.clone())
    }

    /// Gets all registered components
    pub fn get_components(&self) -> &[ComponentInterface] {
        &self.registry
    }

    /// Gets a component by name
    pub fn get_component(&self, name: &str) -> Option<&ComponentInterface> {
        self.registry.iter().find(|c| c.name == name)
    }
}

impl Default for ComponentFactory {
    fn default() -> Self {
        Self::new()
    }
}

/// Initialize component support
pub fn initialize_component_support() -> Result<(), ComponentError> {
    // Check component model support
    let caps = get_host_capabilities();
    
    if !caps.component_model {
        return Err(ComponentError::ComponentModelNotSupported);
    }

    // Initialize component registry
    // In a real implementation, this would set up:
    // 1. Component loading infrastructure
    // 2. Interface validation
    // 3. Type checking
    // 4. Runtime initialization
    
    Ok(())
}

/// Gets component capabilities
pub fn get_component_capabilities() -> ComponentCapabilities {
    let caps = get_host_capabilities();
    
    ComponentCapabilities {
        supported: caps.component_model,
        max_instances: if caps.component_model { 1024 } else { 0 },
        supports_imports: caps.component_model,
        supports_exports: caps.component_model,
        supports_interface_validation: caps.component_model,
        supports_dynamic_loading: caps.component_model,
        supports_static_linking: true, // Always supported
    }
}

/// Component capabilities information
#[derive(Debug, Clone)]
pub struct ComponentCapabilities {
    pub supported: bool,
    pub max_instances: usize,
    pub supports_imports: bool,
    pub supports_exports: bool,
    pub supports_interface_validation: bool,
    pub supports_dynamic_loading: bool,
    pub supports_static_linking: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_component_interface_creation() {
        let interface = ComponentInterface::new("test_component".to_string());
        assert_eq!(interface.name, "test_component");
        assert_eq!(interface.version, "1.0.0");
        assert!(interface.exports.is_empty());
        assert!(interface.imports.is_empty());
    }

    #[test]
    fn test_function_export() {
        let export = FunctionExport {
            name: "test_function".to_string(),
            signature: Signature {
                params: vec![Type::I32],
                returns: Some(Type::I32),
            },
            attributes: vec![ExportAttribute::Public],
            docs: Some("Test function".to_string()),
        };
        
        assert_eq!(export.name, "test_function");
        assert_eq!(export.signature.params.len(), 1);
        assert_eq!(export.signature.returns, Some(Type::I32));
    }

    #[test]
    fn test_function_import() {
        let import = FunctionImport {
            name: "imported_function".to_string(),
            signature: Signature {
                params: vec![Type::I64],
                returns: Some(Type::F64),
            },
            attributes: vec![ImportAttribute::Required],
            interface: Some("test_interface".to_string()),
        };
        
        assert_eq!(import.name, "imported_function");
        assert_eq!(import.signature.params.len(), 1);
        assert_eq!(import.signature.returns, Some(Type::F64));
        assert_eq!(import.interface, Some("test_interface".to_string()));
    }

    #[test]
    fn test_component_interface_validation() {
        let mut interface = ComponentInterface::new("test".to_string());
        
        // Add valid export
        let export = FunctionExport {
            name: "valid_function".to_string(),
            signature: Signature {
                params: vec![Type::I32],
                returns: Some(Type::I32),
            },
            attributes: vec![ExportAttribute::Public],
            docs: None,
        };
        interface.add_export(export);
        
        // Validation should succeed
        assert!(interface.validate().is_ok());
        
        // Add duplicate export
        let duplicate_export = FunctionExport {
            name: "valid_function".to_string(),
            signature: Signature {
                params: vec![],
                returns: None,
            },
            attributes: vec![ExportAttribute::Public],
            docs: None,
        };
        interface.add_export(duplicate_export);
        
        // Validation should fail
        assert!(interface.validate().is_err());
    }

    #[test]
    fn test_component_instance() {
        let interface = ComponentInterface::new("test".to_string());
        let instance = ComponentInstance::new(interface);
        
        // Should succeed if component model is supported
        // (In real tests, we would mock host capabilities)
        match instance {
            Ok(_) => {
                // Instance created successfully
            }
            Err(ComponentError::ComponentModelNotSupported) => {
                // Expected if component model not supported
            }
            Err(_) => {
                panic!("Unexpected error creating component instance");
            }
        }
    }

    #[test]
    fn test_component_lifecycle() {
        let interface = ComponentInterface::new("test".to_string());
        
        if let Ok(mut instance) = ComponentInstance::new(interface) {
            // Test lifecycle state changes
            match instance.state() {
                ComponentState::Initialized => {
                    // Correct initial state
                }
                _ => panic!("Component should be in Initialized state"),
            }
            
            // Start the component
            let start_result = instance.start();
            match start_result {
                Ok(()) => {
                    match instance.state() {
                        ComponentState::Running => {
                            // Correct state after start
                        }
                        _ => panic!("Component should be in Running state"),
                    }
                }
                Err(_) => {
                    // Start failed, state should remain Initialized
                    match instance.state() {
                        ComponentState::Initialized | ComponentState::Error(_) => {
                            // Expected
                        }
                        _ => panic!("Unexpected state after failed start"),
                    }
                }
            }
            
            // Test pause functionality
            if let ComponentState::Running = instance.state() {
                let pause_result = instance.pause();
                match pause_result {
                    Ok(()) => {
                        match instance.state() {
                            ComponentState::Paused => {
                                // Correct state after pause
                            }
                            _ => panic!("Component should be in Paused state"),
                        }
                    }
                    Err(_) => {
                        // Pause failed, state should remain Running
                        match instance.state() {
                            ComponentState::Running => {
                                // Expected
                            }
                            _ => panic!("Unexpected state after failed pause"),
                        }
                    }
                }
            }
            
            // Test stop functionality
            if instance.state() != ComponentState::Stopped {
                let stop_result = instance.stop();
                match stop_result {
                    Ok(()) => {
                        match instance.state() {
                            ComponentState::Stopped => {
                                // Correct state after stop
                            }
                            _ => panic!("Component should be in Stopped state"),
                        }
                    }
                    Err(_) => {
                        // Stop failed, state should remain as is
                        // (except for error states)
                    }
                }
            }
        }
    }

    #[test]
    fn test_component_factory() {
        let mut factory = ComponentFactory::new();
        
        // Register a component
        let interface = ComponentInterface::new("factory_test".to_string());
        assert!(factory.register_component(interface.clone()).is_ok());
        
        // Create an instance
        let instance = factory.create_instance("factory_test");
        assert!(instance.is_ok() || 
               matches!(instance, Err(ComponentError::ComponentModelNotSupported)));
        
        // Get component by name
        let retrieved = factory.get_component("factory_test");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "factory_test");
    }

    #[test]
    fn test_component_capabilities() {
        let caps = get_component_capabilities();
        
        // Should return valid capabilities
        let _ = caps.supported;
        let _ = caps.max_instances;
        let _ = caps.supports_imports;
        let _ = caps.supports_exports;
        let _ = caps.supports_interface_validation;
        let _ = caps.supports_dynamic_loading;
        let _ = caps.supports_static_linking;
    }

    #[test]
    fn test_component_error_display() {
        let error = ComponentError::DuplicateExport("test".to_string());
        let display = format!("{}", error);
        assert!(display.contains("Duplicate export"));
        assert!(display.contains("test"));
        
        let error = ComponentError::ComponentModelNotSupported;
        let display = format!("{}", error);
        assert!(display.contains("Component model not supported"));
        
        let error = ComponentError::ValidationFailed("test error".to_string());
        let display = format!("{}", error);
        assert!(display.contains("Validation failed"));
        assert!(display.contains("test error"));
    }
}
