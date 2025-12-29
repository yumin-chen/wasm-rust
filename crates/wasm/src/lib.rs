//! WasmRust Core Library
//! 
//! This crate provides the core WASM-native type system and abstractions
//! for the WasmRust compiler. It implements zero-cost wrappers for
//! WebAssembly reference types with managed reference tables.

#![no_std]
#![feature(extern_types)]
#![feature(specialization)]
#![feature(unsize)]
#![feature(coerce_unsized)]

use core::marker::PhantomData;
use core::ptr::NonNull;
use core::fmt;

pub mod memory;
pub mod threading;
pub mod component;
pub mod host;

/// Marker trait for types that are safe for zero-copy sharing
/// 
/// Types implementing this trait have no internal pointers and can be
/// safely shared across WASM threads or components without serialization.
pub unsafe trait Pod: Copy + Send + Sync {
    /// Returns true if the type layout is suitable for zero-copy sharing
    fn is_pod_compatible() -> bool {
        true
    }
}

// Blanket implementation for primitive types
unsafe impl Pod for u8 {}
unsafe impl Pod for u16 {}
unsafe impl Pod for u32 {}
unsafe impl Pod for u64 {}
unsafe impl Pod for i8 {}
unsafe impl Pod for i16 {}
unsafe impl Pod for i32 {}
unsafe impl Pod for i64 {}
unsafe impl Pod for f32 {}
unsafe impl Pod for f64 {}
unsafe impl<T> Pod for *const T {}
unsafe impl<T> Pod for *mut T {}

/// Managed reference to JavaScript objects
/// 
/// `ExternRef<T>` provides type-safe access to JavaScript objects through
/// a managed reference table. The handle is an index into a runtime
/// reference table that tracks object lifetimes.
#[repr(transparent)]
pub struct ExternRef<T> {
    handle: u32, // Index into runtime reference table
    _marker: PhantomData<T>,
}

impl<T> ExternRef<T> {
    /// Creates an ExternRef from a raw handle
    /// 
    /// # Safety
    /// The handle must be valid in the current reference table
    /// and must correspond to an object of type T.
    pub unsafe fn from_handle(handle: u32) -> Self {
        Self {
            handle,
            _marker: PhantomData,
        }
    }

    /// Returns the underlying handle
    pub fn as_handle(&self) -> u32 {
        self.handle
    }

    /// Calls a method on the referenced JavaScript object
    /// 
    /// This method requires host profile support and performs
    /// type-safe method invocation through the runtime.
    pub fn call<Args, Ret>(&self, method: &str, args: Args) -> Result<Ret, InteropError>
    where
        T: host::HasMethod<Args, Ret>,
    {
        // Lowered to host-specific interop mechanism
        unsafe { host::invoke_checked(self.handle, method, args) }
    }

    /// Gets a property from the JavaScript object
    pub fn get_property<Ret>(&self, property: &str) -> Result<Ret, InteropError>
    where
        T: host::HasProperty<Ret>,
    {
        unsafe { host::get_property_checked(self.handle, property) }
    }

    /// Sets a property on the JavaScript object
    pub fn set_property<Value>(&self, property: &str, value: Value) -> Result<(), InteropError>
    where
        T: host::HasProperty<Value>,
    {
        unsafe { host::set_property_checked(self.handle, property, value) }
    }
}

impl<T> Clone for ExternRef<T> {
    fn clone(&self) -> Self {
        unsafe { host::add_reference(self.handle) };
        Self {
            handle: self.handle,
            _marker: PhantomData,
        }
    }
}

impl<T> Drop for ExternRef<T> {
    fn drop(&mut self) {
        unsafe { host::remove_reference(self.handle) };
    }
}

impl<T> fmt::Debug for ExternRef<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ExternRef({})", self.handle)
    }
}

/// Managed function references
/// 
/// `FuncRef` represents a reference to a WebAssembly function
/// through the function table. This enables higher-order functions
/// and function pointers in WASM.
#[repr(transparent)]
pub struct FuncRef {
    handle: u32, // Index into function table
}

impl FuncRef {
    /// Creates a FuncRef from a raw handle
    /// 
    /// # Safety
    /// The handle must be a valid function table index.
    pub unsafe fn from_handle(handle: u32) -> Self {
        Self { handle }
    }

    /// Returns the underlying handle
    pub fn as_handle(&self) -> u32 {
        self.handle
    }

    /// Calls the function with the given arguments
    /// 
    /// # Safety
    /// The caller must ensure that the arguments match the function's
    /// expected signature and that the return type is correct.
    pub unsafe fn call<Args, Ret>(&self, args: Args) -> Ret {
        host::call_function(self.handle, args)
    }
}

impl Clone for FuncRef {
    fn clone(&self) -> Self {
        Self { handle: self.handle }
    }
}

impl fmt::Debug for FuncRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FuncRef({})", self.handle)
    }
}
/// Safe shared memory access with explicit constraints
/// 
/// `SharedSlice<'a, T>` provides safe access to shared memory
/// across WASM threads. The lifetime `'a` ensures that the
/// slice cannot outlive shared memory region, and the
/// `Pod` trait constraint ensures type safety.
pub struct SharedSlice<'a, T: Pod> {
    ptr: NonNull<T>,
    len: usize,
    _marker: PhantomData<&'a [T]>,
}

/// Mutable version of SharedSlice for exclusive access
pub use memory::SharedSliceMut;

impl<'a, T: Pod> SharedSlice<'a, T> {
    /// Creates a SharedSlice from a raw pointer and length
    /// 
    /// # Safety
    /// The pointer must be valid for `len` elements and must
    /// point to properly aligned memory.
    pub unsafe fn from_raw_parts(ptr: *mut T, len: usize) -> Self {
        Self {
            ptr: NonNull::new(ptr).expect("null pointer"),
            len,
            _marker: PhantomData,
        }
    }

    /// Returns the length of the slice
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the slice is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Gets a reference to the element at the given index
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len {
            unsafe { Some(&*self.ptr.as_ptr().add(index)) }
        } else {
            None
        }
    }

    /// Gets a mutable reference to the element at the given index
    /// 
    /// This method requires exclusive access and is only available
    /// when the SharedSlice is not shared across threads.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len {
            unsafe { Some(&mut *self.ptr.as_ptr().add(index)) }
        } else {
            None
        }
    }

    /// Splits the slice into two at the given index
    pub fn split_at(&self, mid: usize) -> (SharedSlice<'a, T>, SharedSlice<'a, T>) {
        assert!(mid <= self.len, "split index out of bounds");
        
        let left = unsafe {
            SharedSlice::from_raw_parts(self.ptr.as_ptr(), mid)
        };
        
        let right = unsafe {
            SharedSlice::from_raw_parts(
                self.ptr.as_ptr().add(mid),
                self.len - mid
            )
        };
        
        (left, right)
    }

    /// Returns an iterator over the slice
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        unsafe {
            std::slice::from_raw_parts(self.ptr.as_ptr(), self.len)
                .iter()
        }
    }
}

impl<'a, T: Pod> Clone for SharedSlice<'a, T> {
    fn clone(&self) -> Self {
        unsafe { 
            SharedSlice::from_raw_parts(self.ptr.as_ptr(), self.len)
        }
    }
}

impl<'a, T: Pod> fmt::Debug for SharedSlice<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SharedSlice(ptr={:?}, len={})", self.ptr, self.len)
    }
}

/// Errors that can occur during JavaScript interoperation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InteropError {
    /// The requested method does not exist on the object
    MethodNotFound(String),
    /// The arguments do not match the method's signature
    InvalidArguments,
    /// The JavaScript execution threw an exception
    JavaScriptException(String),
    /// The host profile does not support this operation
    UnsupportedOperation,
    /// The object reference is invalid or has been garbage collected
    InvalidReference,
    /// Type mismatch between expected and actual types
    TypeMismatch,
}

impl fmt::Display for InteropError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InteropError::MethodNotFound(method) => {
                write!(f, "Method '{}' not found", method)
            }
            InteropError::InvalidArguments => {
                write!(f, "Invalid arguments for method call")
            }
            InteropError::JavaScriptException(msg) => {
                write!(f, "JavaScript exception: {}", msg)
            }
            InteropError::UnsupportedOperation => {
                write!(f, "Operation not supported by host profile")
            }
            InteropError::InvalidReference => {
                write!(f, "Invalid object reference")
            }
            InteropError::TypeMismatch => {
                write!(f, "Type mismatch in interoperation")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extern_ref_creation() {
        let handle = 42u32;
        let ext_ref = unsafe { ExternRef::<()>::from_handle(handle) };
        assert_eq!(ext_ref.as_handle(), handle);
    }

    #[test]
    fn test_func_ref_creation() {
        let handle = 123u32;
        let func_ref = unsafe { FuncRef::from_handle(handle) };
        assert_eq!(func_ref.as_handle(), handle);
    }

    #[test]
    fn test_shared_slice() {
        let mut data = [1u32, 2u32, 3u32, 4u32];
        let shared_slice = unsafe {
            SharedSlice::from_raw_parts(data.as_mut_ptr(), data.len())
        };
        
        assert_eq!(shared_slice.len(), 4);
        assert!(!shared_slice.is_empty());
        assert_eq!(shared_slice.get(0), Some(&1u32));
        assert_eq!(shared_slice.get(3), Some(&4u32));
        assert_eq!(shared_slice.get(4), None);
    }

    #[test]
    fn test_shared_slice_split() {
        let mut data = [1u32, 2u32, 3u32, 4u32];
        let shared_slice = unsafe {
            SharedSlice::from_raw_parts(data.as_mut_ptr(), data.len())
        };
        
        let (left, right) = shared_slice.split_at(2);
        assert_eq!(left.len(), 2);
        assert_eq!(right.len(), 2);
        assert_eq!(left.get(0), Some(&1u32));
        assert_eq!(right.get(0), Some(&3u32));
    }

    #[test]
    fn test_pod_trait() {
        assert!(u32::is_pod_compatible());
        assert!(f64::is_pod_compatible());
        assert!(<*const i8>::is_pod_compatible());
    }
}
