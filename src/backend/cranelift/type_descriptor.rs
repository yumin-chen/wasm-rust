//! WasmTypeDescriptor Implementation
//! 
//! This module implements the type descriptor system for thin monomorphization.
//! Type descriptors provide runtime type information including size, alignment,
//! and function pointers for type-specific operations like drop, clone, and copy.

use crate::wasmir::{Type, WasmIR, Instruction, Operand};
use rustc_middle::ty::{TyS, TyKind};
use rustc_target::spec::Target;
use std::collections::HashMap;

/// WasmTypeDescriptor - Runtime type information for thin monomorphization
/// 
/// This structure contains all type-specific information needed by thinned
/// functions to operate on opaque pointers safely and efficiently.
#[derive(Debug, Clone, PartialEq)]
pub struct WasmTypeDescriptor {
    /// Unique identifier for this type
    pub type_id: u32,
    /// Size of the type in bytes
    pub size: u32,
    /// Alignment requirement in bytes
    pub align: u32,
    /// Function pointer to drop glue (destructor)
    pub drop_glue: Option<u32>, // Function table index
    /// Function pointer to clone function (if Copy is not available)
    pub clone_fn: Option<u32>, // Function table index
    /// Whether the type is Copy (can be bitwise copied)
    pub is_copy: bool,
    /// Whether the type is Pod (plain old data)
    pub is_pod: bool,
    /// Name of the type for debugging
    pub name: String,
}

impl WasmTypeDescriptor {
    /// Creates a new type descriptor
    pub fn new(
        type_id: u32,
        size: u32,
        align: u32,
        name: String,
    ) -> Self {
        Self {
            type_id,
            size,
            align,
            drop_glue: None,
            clone_fn: None,
            is_copy: false,
            is_pod: false,
            name,
        }
    }

    /// Sets the drop glue function
    pub fn with_drop_glue(mut self, drop_glue: u32) -> Self {
        self.drop_glue = Some(drop_glue);
        self
    }

    /// Sets the clone function
    pub fn with_clone_fn(mut self, clone_fn: u32) -> Self {
        self.clone_fn = Some(clone_fn);
        self
    }

    /// Marks the type as Copy
    pub fn with_copy(mut self, is_copy: bool) -> Self {
        self.is_copy = is_copy;
        self
    }

    /// Marks the type as Pod
    pub fn with_pod(mut self, is_pod: bool) -> Self {
        self.is_pod = is_pod;
        self
    }

    /// Calculates the descriptor layout size in WASM memory
    pub fn layout_size(&self) -> u32 {
        // Layout: [type_id(4), size(4), align(4), drop_glue(4), clone_fn(4), flags(4)]
        24
    }

    /// Gets the flags byte encoding
    pub fn flags(&self) -> u32 {
        let mut flags = 0u32;
        if self.is_copy {
            flags |= 0x01;
        }
        if self.is_pod {
            flags |= 0x02;
        }
        if self.drop_glue.is_some() {
            flags |= 0x04;
        }
        if self.clone_fn.is_some() {
            flags |= 0x08;
        }
        flags
    }

    /// Generates WASM data section for this descriptor
    pub fn generate_wasm_data(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(self.layout_size() as usize);
        
        // type_id
        data.extend_from_slice(&self.type_id.to_le_bytes());
        // size
        data.extend_from_slice(&self.size.to_le_bytes());
        // align
        data.extend_from_slice(&self.align.to_le_bytes());
        // drop_glue
        let drop_glue = self.drop_glue.unwrap_or(0);
        data.extend_from_slice(&drop_glue.to_le_bytes());
        // clone_fn
        let clone_fn = self.clone_fn.unwrap_or(0);
        data.extend_from_slice(&clone_fn.to_le_bytes());
        // flags
        data.extend_from_slice(&self.flags().to_le_bytes());
        
        data
    }
}

/// Type descriptor registry for managing all type descriptors
pub struct TypeDescriptorRegistry {
    /// Map from Rust type to descriptor
    descriptors: HashMap<String, WasmTypeDescriptor>,
    /// Next available type ID
    next_type_id: u32,
    /// Function table for type-specific functions
    function_table: Vec<String>,
}

impl TypeDescriptorRegistry {
    /// Creates a new type descriptor registry
    pub fn new() -> Self {
        Self {
            descriptors: HashMap::new(),
            next_type_id: 1,
            function_table: Vec::new(),
        }
    }

    /// Registers a new type descriptor
    pub fn register_descriptor(&mut self, type_name: &str, descriptor: WasmTypeDescriptor) -> u32 {
        let type_id = descriptor.type_id;
        self.descriptors.insert(type_name.to_string(), descriptor);
        
        // Update next type ID if needed
        if type_id >= self.next_type_id {
            self.next_type_id = type_id + 1;
        }
        
        type_id
    }

    /// Gets a type descriptor by name
    pub fn get_descriptor(&self, type_name: &str) -> Option<&WasmTypeDescriptor> {
        self.descriptors.get(type_name)
    }

    /// Gets a type descriptor by ID
    pub fn get_descriptor_by_id(&self, type_id: u32) -> Option<&WasmTypeDescriptor> {
        self.descriptors.values().find(|desc| desc.type_id == type_id)
    }

    /// Creates or gets a descriptor for a Rust type
    pub fn create_descriptor_for_type(
        &mut self,
        rust_type: &TyS,
        target: &Target,
    ) -> Result<u32, DescriptorError> {
        let type_name = self.type_to_string(rust_type);
        
        // Check if already registered
        if let Some(desc) = self.get_descriptor(&type_name) {
            return Ok(desc.type_id);
        }
        
        // Create new descriptor
        let type_id = self.next_type_id;
        self.next_type_id += 1;
        
        let (size, align) = self.calculate_type_size_align(rust_type, target)?;
        let is_copy = self.is_type_copy(rust_type);
        let is_pod = self.is_type_pod(rust_type);
        
        let mut descriptor = WasmTypeDescriptor::new(type_id, size, align, type_name.clone())
            .with_copy(is_copy)
            .with_pod(is_pod);
        
        // Generate drop glue if needed
        if !is_pod && self.type_needs_drop(rust_type) {
            let drop_fn_id = self.generate_drop_glue(&type_name, rust_type)?;
            descriptor = descriptor.with_drop_glue(drop_fn_id);
        }
        
        // Generate clone function if needed
        if !is_copy && self.type_needs_clone(rust_type) {
            let clone_fn_id = self.generate_clone_fn(&type_name, rust_type)?;
            descriptor = descriptor.with_clone_fn(clone_fn_id);
        }
        
        self.register_descriptor(&type_name, descriptor);
        Ok(type_id)
    }

    /// Converts a Rust type to string representation
    fn type_to_string(&self, rust_type: &TyS) -> String {
        match rust_type.kind() {
            TyKind::Bool => "bool".to_string(),
            TyKind::Int(int_ty) => format!("i{}", int_ty.bit_width().unwrap_or(32)),
            TyKind::Uint(uint_ty) => format!("u{}", uint_ty.bit_width().unwrap_or(32)),
            TyKind::Float(float_ty) => match float_ty.bit_width() {
                Some(32) => "f32".to_string(),
                Some(64) => "f64".to_string(),
                _ => "funknown".to_string(),
            },
            TyKind::Adt(adt_def, substs) => {
                let name = adt_def.def_id().to_string();
                if substs.is_empty() {
                    name
                } else {
                    format!("{}<{}>", name, self.substs_to_string(substs))
                }
            },
            TyKind::Slice(ty) => format!("[{}]", self.type_to_string(ty)),
            TyKind::Array(ty, len) => format!("[{}; {}]", self.type_to_string(ty), len),
            TyKind::Ref(_, ty, _) => format!("&{}", self.type_to_string(ty)),
            TyKind::Tuple(tys) => {
                if tys.is_empty() {
                    "()".to_string()
                } else {
                    format!("({})", tys.iter().map(|ty| self.type_to_string(ty)).collect::<Vec<_>>().join(", "))
                }
            },
            _ => format!("unknown_type_{:?}", rust_type.kind()),
        }
    }

    /// Converts substs to string
    fn substs_to_string(&self, substs: &[rustc_middle::ty::GenericArg<'_>]) -> String {
        substs.iter()
            .map(|arg| match arg.unpack() {
                rustc_middle::ty::GenericArgKind::Type(ty) => self.type_to_string(ty),
                _ => "?".to_string(),
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Calculates size and alignment for a type
    fn calculate_type_size_align(
        &self,
        rust_type: &TyS,
        target: &Target,
    ) -> Result<(u32, u32), DescriptorError> {
        // Simplified size/alignment calculation
        // In practice, this would use rustc's layout computation
        match rust_type.kind() {
            TyKind::Bool => Ok((1, 1)),
            TyKind::Int(int_ty) => {
                let bits = int_ty.bit_width().unwrap_or(target.pointer_width());
                Ok((bits / 8, bits / 8))
            },
            TyKind::Uint(uint_ty) => {
                let bits = uint_ty.bit_width().unwrap_or(target.pointer_width());
                Ok((bits / 8, bits / 8))
            },
            TyKind::Float(float_ty) => {
                let bits = float_ty.bit_width().unwrap_or(32);
                Ok((bits / 8, bits / 8))
            },
            TyKind::Ref(_, inner, _) => {
                let (size, align) = self.calculate_type_size_align(inner, target)?;
                Ok((target.pointer_width() / 8, target.pointer_width() / 8))
            },
            TyKind::Tuple(tys) => {
                let mut total_size = 0;
                let mut max_align = 1;
                
                for ty in tys {
                    let (size, align) = self.calculate_type_size_align(ty, target)?;
                    // Align to field's alignment
                    total_size = (total_size + align - 1) & !(align - 1);
                    total_size += size;
                    max_align = max_align.max(align);
                }
                
                // Align struct to its maximum alignment
                total_size = (total_size + max_align - 1) & !(max_align - 1);
                Ok((total_size, max_align))
            },
            _ => Err(DescriptorError::UnsupportedType(format!("Cannot calculate size for type: {:?}", rust_type.kind()))),
        }
    }

    /// Checks if a type is Copy
    fn is_type_copy(&self, rust_type: &TyS) -> bool {
        // Simplified Copy check
        match rust_type.kind() {
            TyKind::Bool | TyKind::Int(_) | TyKind::Uint(_) | TyKind::Float(_) => true,
            TyKind::Ref(_, _, _) => true, // References are always copy
            TyKind::Tuple(tys) => tys.iter().all(|ty| self.is_type_copy(ty)),
            _ => false,
        }
    }

    /// Checks if a type is Pod (plain old data)
    fn is_type_pod(&self, rust_type: &TyS) -> bool {
        // Pod types are Copy and don't need special handling
        self.is_type_copy(rust_type) && !self.type_needs_drop(rust_type)
    }

    /// Checks if a type needs drop glue
    fn type_needs_drop(&self, rust_type: &TyS) -> bool {
        // Simplified drop check
        match rust_type.kind() {
            TyKind::Bool | TyKind::Int(_) | TyKind::Uint(_) | TyKind::Float(_) => false,
            TyKind::Ref(_, _, _) => false, // References don't need drop
            TyKind::Tuple(tys) => tys.iter().any(|ty| self.type_needs_drop(ty)),
            TyKind::Adt(adt_def, _) => {
                // Check if the ADT has a custom Drop implementation
                // This is simplified - in practice, would check ADT's destructor
                adt_def.has_dtor()
            },
            _ => true, // Conservative: assume needs drop
        }
    }

    /// Checks if a type needs clone function
    fn type_needs_clone(&self, rust_type: &TyS) -> bool {
        // Clone is needed for types that are not Copy but can be cloned
        !self.is_type_copy(rust_type) && self.type_can_be_cloned(rust_type)
    }

    /// Checks if a type can be cloned
    fn type_can_be_cloned(&self, rust_type: &TyS) -> bool {
        // Simplified clone check
        match rust_type.kind() {
            TyKind::Bool | TyKind::Int(_) | TyKind::Uint(_) | TyKind::Float(_) => true,
            TyKind::Tuple(tys) => tys.iter().all(|ty| self.type_can_be_cloned(ty)),
            TyKind::Ref(_, inner, _) => self.type_can_be_cloned(inner),
            _ => false, // Conservative: assume cannot be cloned
        }
    }

    /// Generates drop glue function for a type
    fn generate_drop_glue(&mut self, type_name: &str, rust_type: &TyS) -> Result<u32, DescriptorError> {
        let function_name = format!("drop_{}", type_name);
        let function_index = self.function_table.len() as u32;
        
        self.function_table.push(function_name.clone());
        
        // In practice, this would generate actual WASM code for the drop function
        // For now, we just register the function name
        Ok(function_index)
    }

    /// Generates clone function for a type
    fn generate_clone_fn(&mut self, type_name: &str, rust_type: &TyS) -> Result<u32, DescriptorError> {
        let function_name = format!("clone_{}", type_name);
        let function_index = self.function_table.len() as u32;
        
        self.function_table.push(function_name.clone());
        
        // In practice, this would generate actual WASM code for the clone function
        Ok(function_index)
    }

    /// Gets all registered descriptors
    pub fn get_all_descriptors(&self) -> impl Iterator<Item = &WasmTypeDescriptor> {
        self.descriptors.values()
    }

    /// Gets the function table entries
    pub fn get_function_table(&self) -> &[String] {
        &self.function_table
    }

    /// Generates WASM data section for all descriptors
    pub fn generate_descriptor_data_section(&self) -> Vec<u8> {
        let mut data = Vec::new();
        
        for descriptor in self.get_all_descriptors() {
            data.extend_from_slice(&descriptor.generate_wasm_data());
        }
        
        data
    }

    /// Generates the function table indices for descriptors
    pub fn generate_descriptor_table(&self) -> Vec<u32> {
        self.get_all_descriptors()
            .map(|desc| desc.type_id)
            .collect()
    }
}

/// Errors that can occur during descriptor creation
#[derive(Debug, Clone)]
pub enum DescriptorError {
    /// Unsupported type for descriptor creation
    UnsupportedType(String),
    /// Layout computation failed
    LayoutError(String),
    /// Function generation failed
    FunctionGenerationError(String),
}

impl std::fmt::Display for DescriptorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DescriptorError::UnsupportedType(msg) => write!(f, "Unsupported type: {}", msg),
            DescriptorError::LayoutError(msg) => write!(f, "Layout error: {}", msg),
            DescriptorError::FunctionGenerationError(msg) => write!(f, "Function generation error: {}", msg),
        }
    }
}

impl std::error::Error for DescriptorError {}

#[cfg(test)]
mod tests {
    use super::*;
    use rustc_target::spec::Target;

    #[test]
    fn test_type_descriptor_creation() {
        let descriptor = WasmTypeDescriptor::new(1, 8, 8, "TestType".to_string())
            .with_copy(true)
            .with_pod(true)
            .with_drop_glue(42);

        assert_eq!(descriptor.type_id, 1);
        assert_eq!(descriptor.size, 8);
        assert_eq!(descriptor.align, 8);
        assert_eq!(descriptor.name, "TestType");
        assert!(descriptor.is_copy);
        assert!(descriptor.is_pod);
        assert_eq!(descriptor.drop_glue, Some(42));
    }

    #[test]
    fn test_descriptor_layout_size() {
        let descriptor = WasmTypeDescriptor::new(1, 8, 8, "Test".to_string());
        assert_eq!(descriptor.layout_size(), 24); // 6 fields * 4 bytes each
    }

    #[test]
    fn test_descriptor_flags() {
        let descriptor = WasmTypeDescriptor::new(1, 8, 8, "Test".to_string())
            .with_copy(true)
            .with_pod(false)
            .with_drop_glue(42);

        let flags = descriptor.flags();
        assert!(flags & 0x01 != 0); // Copy flag
        assert!(flags & 0x02 == 0); // Pod flag
        assert!(flags & 0x04 != 0); // Drop flag
        assert!(flags & 0x08 == 0); // Clone flag
    }

    #[test]
    fn test_descriptor_wasm_data_generation() {
        let descriptor = WasmTypeDescriptor::new(1, 8, 4, "Test".to_string())
            .with_copy(true)
            .with_pod(true);

        let data = descriptor.generate_wasm_data();
        assert_eq!(data.len(), 24); // Layout size
        
        // Check type_id (little endian)
        assert_eq!(data[0], 1);
        assert_eq!(data[1], 0);
        assert_eq!(data[2], 0);
        assert_eq!(data[3], 0);
        
        // Check size (little endian)
        assert_eq!(data[4], 8);
        assert_eq!(data[5], 0);
        assert_eq!(data[6], 0);
        assert_eq!(data[7], 0);
        
        // Check align (little endian)
        assert_eq!(data[8], 4);
        assert_eq!(data[9], 0);
        assert_eq!(data[10], 0);
        assert_eq!(data[11], 0);
    }

    #[test]
    fn test_type_descriptor_registry() {
        let mut registry = TypeDescriptorRegistry::new();
        
        let descriptor = WasmTypeDescriptor::new(1, 4, 4, "i32".to_string())
            .with_copy(true)
            .with_pod(true);
        
        let type_id = registry.register_descriptor("i32", descriptor);
        assert_eq!(type_id, 1);
        
        let retrieved = registry.get_descriptor("i32").unwrap();
        assert_eq!(retrieved.type_id, 1);
        assert_eq!(retrieved.name, "i32");
        assert!(retrieved.is_copy);
        assert!(retrieved.is_pod);
    }

    #[test]
    fn test_simple_type_classification() {
        let registry = TypeDescriptorRegistry::new();
        
        // These would normally come from rustc type system
        // For testing, we'll simulate the classification logic
        
        // Primitive types should be Copy and Pod
        assert!(registry.is_type_copy(&/* simulate bool type */));
        assert!(registry.is_type_pod(&/* simulate bool type */));
        
        // References should be Copy
        assert!(registry.is_type_copy(&/* simulate &i32 type */));
        
        // Complex types might not be Copy
        assert!(!registry.is_type_copy(&/* simulate String type */));
    }

    #[test]
    fn test_descriptor_data_section() {
        let mut registry = TypeDescriptorRegistry::new();
        
        let desc1 = WasmTypeDescriptor::new(1, 4, 4, "i32".to_string())
            .with_copy(true).with_pod(true);
        let desc2 = WasmTypeDescriptor::new(2, 1, 1, "bool".to_string())
            .with_copy(true).with_pod(true);
        
        registry.register_descriptor("i32", desc1);
        registry.register_descriptor("bool", desc2);
        
        let data_section = registry.generate_descriptor_data_section();
        assert_eq!(data_section.len(), 48); // 2 descriptors * 24 bytes each
    }
}