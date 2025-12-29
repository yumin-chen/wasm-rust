//! WasmRust Core Library
//! 
//! This crate provides the core abstractions for WasmRust-optimized
//! Rust-to-WebAssembly compilation, including WASM-native types,
//! zero-cost wrappers, and safe memory abstractions.

#![no_std]
#![feature(extern_types)]
#![feature(unsize)]
#![feature(coerce_unsized)]
#![feature(generic_associated_types)]
#![feature(min_specialization)]
#![feature(const_fn)]
#![feature(const_mut_refs)]

extern crate alloc;
use alloc::vec::Vec;
use alloc::string::ToString;
use core::marker::PhantomData;
use core::ptr::{self, NonNull};
use core::slice;
use core::mem;
use core::ops::{Deref, DerefMut, Index, IndexMut};

pub mod host;
pub mod memory;
pub mod threading;
pub mod component;

use host::{HostProfile, HostCapabilities, get_host_capabilities};

/// Error types for WASM operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WasmError {
    /// Type mismatch error
    TypeMismatch,
    /// Null dereference error
    NullDereference,
    /// Out of bounds access
    OutOfBounds,
    /// Invalid operation
    InvalidOperation(String),
    /// Host error
    HostError(host::InteropError),
    /// Memory error
    MemoryError(memory::MemoryError),
    /// Threading error
    ThreadingError(threading::ThreadError),
    /// Component error
    ComponentError(component::ComponentError),
}

// Implement error conversions for seamless error handling
impl From<memory::MemoryError> for WasmError {
    fn from(err: memory::MemoryError) -> Self {
        WasmError::MemoryError(err)
    }
}

impl From<threading::ThreadError> for WasmError {
    fn from(err: threading::ThreadError) -> Self {
        WasmError::ThreadingError(err)
    }
}

impl From<component::ComponentError> for WasmError {
    fn from(err: component::ComponentError) -> Self {
        WasmError::ComponentError(err)
    }
}

impl From<host::InteropError> for WasmError {
    fn from(err: host::InteropError) -> Self {
        WasmError::HostError(err)
    }
}

impl core::fmt::Display for WasmError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            WasmError::TypeMismatch => write!(f, "Type mismatch"),
            WasmError::NullDereference => write!(f, "Null dereference"),
            WasmError::OutOfBounds => write!(f, "Out of bounds access"),
            WasmError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
            WasmError::HostError(err) => write!(f, "Host error: {}", err),
            WasmError::MemoryError(err) => write!(f, "Memory error: {}", err),
            WasmError::ThreadingError(err) => write!(f, "Threading error: {}", err),
            WasmError::ComponentError(err) => write!(f, "Component error: {}", err),
        }
    }
}

/// Trait for types that can be safely shared without copying
/// 
/// This trait ensures that types have no interior mutability
/// and can be safely shared across thread boundaries when wrapped
/// in appropriate synchronization primitives.
///
/// # Safety
///
/// Implementing `Pod` for a type asserts that **all** of the following hold:
///
/// - The type has no padding bytes with semantic meaning
/// - The type contains no pointers or references
/// - The type has no interior mutability
/// - The type has no drop glue
/// - All bit patterns are valid
///
/// ## Undefined Behavior
///
/// Implementing `Pod` for a type that violates any of the above invariants
/// results in undefined behavior when used by this crate.
///
/// ## Compiler Assumptions
///
/// The compiler may assume:
/// - `T: Pod` can be freely copied
/// - Shared access is race-free
/// - Aggressive memory optimizations are valid
///
/// The compiler must not:
/// - Assume zeroed memory is valid unless proven
/// - Insert hidden validation or runtime checks
pub unsafe trait Pod: 'static {
    /// Returns true if the type is valid for zero-copy sharing
    fn is_valid_for_sharing() -> bool {
        true
    }
}

// Implement Pod for primitive types
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
unsafe impl Pod for bool {}
unsafe impl Pod for () {}

/// Implement Pod for arrays of Pod types
unsafe impl<T: Pod> Pod for [T] {}
unsafe impl<T: Pod, const N: usize> Pod for [T; N] {}
/// ExternRef<T> - Type-safe JavaScript object reference
/// 
/// This type provides zero-cost wrapper for JavaScript objects with
/// compile-time type checking and automatic reference management.
///
/// # Safety
///
/// `ExternRef<T>` is safe to use **only if all of the following invariants hold**:
///
/// - The underlying handle refers to a valid host-managed object
/// - The host guarantees stable identity for lifetime of handle
/// - The handle is not reused for a different object
/// - The logical type `T` matches the host object's expected type
///
/// ## Undefined Behavior
///
/// The following actions result in undefined behavior:
///
/// - Constructing an `ExternRef<T>` from an arbitrary integer
/// - Transmuting between `ExternRef<U>` and `ExternRef<V>`
/// - Using an `ExternRef` after the host invalidates the handle
/// - Passing an `ExternRef<T>` to a host expecting a different logical type
///
/// ## Compiler Assumptions
///
/// The compiler may assume:
/// - `ExternRef<T>` is an opaque, non-aliasing handle
/// - Cloning and copying are constant-time
/// - No linear memory is accessed through this type
///
/// The compiler must not assume:
/// - Any relationship to linear memory
/// - Dereferenceable data
#[repr(transparent)]
pub struct ExternRef<T: ?Sized> {
    /// Internal handle to the JavaScript object
    handle: u32,
    /// Phantom data for type safety
    _phantom: PhantomData<T>,
}

impl<T> ExternRef<T> {
    /// Creates a new ExternRef from a handle
    /// 
    /// # Safety
    /// 
    /// Caller must ensure that:
    /// - The handle refers to a valid host-managed object
    /// - The object's logical type matches `T`
    /// - The handle is not a fabricated value
    pub unsafe fn from_handle(handle: u32) -> Self {
        Self {
            handle,
            _phantom: PhantomData,
        }
    }

    /// Gets the internal handle
    pub fn handle(&self) -> u32 {
        self.handle
    }

    /// Checks if the reference is null
    pub fn is_null(&self) -> bool {
        self.handle == 0
    }

    /// Creates a null reference
    pub fn null() -> Self {
        Self {
            handle: 0,
            _phantom: PhantomData,
        }
    }

    /// Safely accesses a property on the JavaScript object
    /// 
    /// # Safety
    /// 
    /// This operation is safe when:
    /// - `self.handle` refers to a valid JavaScript object
    /// - The property name is a valid UTF-8 string
    /// - The object has the specified property with type `R`
    /// - No other thread is concurrently modifying the object
    /// 
    /// Undefined behavior occurs if:
    /// - The handle is invalid or has been garbage collected
    /// - The property does not exist on the object
    /// - The property type does not match `R`
    /// - The host environment does not support property access
    pub fn get_property<R>(&self, property: &str) -> Result<R, WasmError>
    where
        T: host::HasProperty<R>,
    {
        if self.is_null() {
            return Err(WasmError::NullDereference);
        }

        // Validate property exists and has correct type
        T::validate_property(property)
            .map_err(|e| WasmError::HostError(e))?;

        // SAFETY: Property access is guarded by null check and type validation
        // The host guarantees that handle refers to a valid object of type T
        unsafe {
            host::get_property_checked::<V>(self.handle, property)
                .map_err(|e| WasmError::HostError(e))
        }
    }

    /// Safely sets a property on the JavaScript object
    /// 
    /// # Safety
    /// 
    /// This operation is safe when:
    /// - `self.handle` refers to a valid JavaScript object
    /// - The property name is a valid UTF-8 string
    /// - The object has the specified property accepting type `V`
    /// - No other thread is concurrently modifying the same property
    /// 
    /// Undefined behavior occurs if:
    /// - The handle is invalid or has been garbage collected
    /// - The property does not exist or is read-only
    /// - The property type does not match `V`
    /// - The host environment does not support property mutation
    pub fn set_property<V>(&self, property: &str, value: V) -> Result<(), WasmError>
    where
        T: host::HasProperty<V>,
    {
        if self.is_null() {
            return Err(WasmError::NullDereference);
        }

        // Validate property exists and has correct type
        T::validate_property(property)
            .map_err(|e| WasmError::HostError(e))?;

        // SAFETY: Property mutation is guarded by null check and type validation
        // The host guarantees atomic property mutation for the object type
        unsafe {
            host::set_property_checked::<T>(self.handle, property, value)
                .map_err(|e| WasmError::HostError(e))
        }
    }

    /// Safely invokes a method on the JavaScript object
    /// 
    /// # Safety
    /// 
    /// This operation is safe when:
    /// - `self.handle` refers to a valid JavaScript object
    /// - The method name is a valid UTF-8 string
    /// - The object has a method with matching signature `(Args) -> Ret`
    /// - The arguments are of types compatible with the method
    /// - No other thread is concurrently mutating the object
    /// 
    /// Undefined behavior occurs if:
    /// - The handle is invalid or has been garbage collected
    /// - The method does not exist or has incompatible signature
    /// - The arguments are of incorrect types
    /// - The host environment does not support method invocation
    pub fn invoke_method<Args, Ret>(&self, method: &str, args: Args) -> Result<Ret, WasmError>
    where
        T: host::HasMethod<Args, Ret>,
    {
        if self.is_null() {
            return Err(WasmError::NullDereference);
        }

        // Validate method exists and has correct signature
        T::validate_method(method)
            .map_err(|e| WasmError::HostError(e))?;

        // SAFETY: Method invocation is guarded by null check and signature validation
        // The host guarantees that handle refers to a valid object with method T
        unsafe {
            host::invoke_checked::<T, Args, Ret>(self.handle, method, args)
                .map_err(|e| WasmError::HostError(e))
        }
    }
}

impl<T> Clone for ExternRef<T> {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle,
            _phantom: PhantomData,
        }
    }
}

impl<T> Copy for ExternRef<T> {}

impl<T> PartialEq for ExternRef<T> {
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle
    }
}

impl<T> Eq for ExternRef<T> {}

impl<T> core::hash::Hash for ExternRef<T> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
    }
}

/// FuncRef - Type-safe function reference
/// 
/// This type provides zero-cost wrapper for WASM function references
/// with compile-time type checking.
#[repr(transparent)]
pub struct FuncRef<Args, Ret> {
    /// Function table index
    index: u32,
    /// Phantom data for type safety
    _phantom: PhantomData<(Args, Ret)>,
}

impl<Args, Ret> FuncRef<Args, Ret> {
    /// Creates a new FuncRef from a table index
    /// 
    /// # Safety
    /// Caller must ensure that the index is valid and the function
    /// at that index has the correct signature.
    pub unsafe fn from_index(index: u32) -> Self {
        Self {
            index,
            _phantom: PhantomData,
        }
    }

    /// Gets the function table index
    pub fn index(&self) -> u32 {
        self.index
    }

    /// Creates a null function reference
    pub fn null() -> Self {
        Self {
            index: 0,
            _phantom: PhantomData,
        }
    }

    /// Checks if the function reference is null
    pub fn is_null(&self) -> bool {
        self.index == 0
    }

    /// Calls the function with the given arguments
    /// 
    /// # Safety
    /// Caller must ensure that the function signature matches
    /// the actual function at the table index.
    pub unsafe fn call(&self, args: Args) -> Ret {
        // This would be implemented with WASM function table call
        // For now, this is a placeholder
        panic!("Function calling not implemented")
    }
}

impl<Args, Ret> Clone for FuncRef<Args, Ret> {
    fn clone(&self) -> Self {
        Self {
            index: self.index,
            _phantom: PhantomData,
        }
    }
}

impl<Args, Ret> Copy for FuncRef<Args, Ret> {}

impl<Args, Ret> PartialEq for FuncRef<Args, Ret> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl<Args, Ret> Eq for FuncRef<Args, Ret> {}

impl<Args, Ret> core::hash::Hash for FuncRef<Args, Ret> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.index.hash(state);
    }
}

/// SharedSlice<'a, T> - Safe concurrent memory access
/// 
/// This type provides thread-safe shared memory access for Pod types
/// with compile-time data race prevention.
///
/// # Safety
///
/// `SharedSlice<'a, T>` is sound **only if**:
///
/// - `ptr` points to valid linear memory for `len` elements
/// - The backing memory is immutable for lifetime `'a`
/// - `T: Pod` guarantees race-free shared access
/// - No aliasing mutable access exists during `'a`
///
/// ## Undefined Behavior
///
/// - Mutating memory while a `SharedSlice` exists
/// - Constructing from invalid or dangling pointers
/// - Using `SharedSlice` in unsupported threading environments
/// - Creating with non-`Pod` types
///
/// ## Compiler Assumptions
///
/// The compiler may assume:
/// - `SharedSlice<'a, T>` behaves like `&'a [T]`
/// - No writes occur through any alias
/// - Reads are race-free
///
/// This enables:
/// - Vectorization
/// - Load hoisting
/// - Bounds-check elimination
/// - Cross-thread read reordering
///
/// The compiler must not:
/// - Assume mutable access is safe
/// - Skip bounds checking based on trust
/// - Assume memory remains valid beyond `'a`
pub struct SharedSlice<'a, T: Pod> {
    /// Pointer to the shared memory region
    ptr: NonNull<T>,
    /// Length of the slice
    len: usize,
    /// Phantom lifetime marker
    _phantom: PhantomData<&'a [T]>,
}

impl<'a, T: Pod> SharedSlice<'a, T> {
    /// Creates a new SharedSlice from a raw pointer and length
    /// 
    /// # Safety
    /// 
    /// This operation is safe when:
    /// - `ptr` is valid and properly aligned for type `T`
    /// - The memory region contains exactly `len` valid `T` values
    /// - The memory region remains valid and immutable for lifetime `'a`
    /// - `T: Pod` guarantees that shared reads are race-free
    /// 
    /// Undefined behavior occurs if:
    /// - `ptr` is null, dangling, or misaligned
    /// - The memory region contains uninitialized or invalid `T` values
    /// - The memory is mutated during `'a`
    /// - `len` exceeds the actual allocated memory size
    /// - `T` is not actually `Pod` (violates shared access guarantees)
    pub unsafe fn from_raw_parts(ptr: *const T, len: usize) -> Self {
        Self {
            ptr: NonNull::new_unchecked(ptr as *mut T),
            len,
            _phantom: PhantomData,
        }
    }

    /// Creates a SharedSlice from a slice
    /// 
    /// Returns error if the type is not Pod
    pub fn from_slice(slice: &'a [T]) -> Result<Self, WasmError> {
        if !T::is_valid_for_sharing() {
            return Err(WasmError::InvalidOperation(
                "Type is not valid for sharing".to_string()
            ));
        }

        unsafe {
            Ok(Self::from_raw_parts(slice.as_ptr(), slice.len()))
        }
    }

    /// Creates a SharedSlice from a mutable slice
    /// 
    /// Returns error if the type is not Pod or threading is not available
    pub fn from_slice_mut(slice: &'a mut [T]) -> Result<Self, WasmError> {
        let caps = get_host_capabilities();
        
        // Check if threading is available for mutable access
        if !caps.threading {
            return Err(WasmError::ThreadingError(
                threading::ThreadError::ThreadingNotSupported
            ));
        }

        if !T::is_valid_for_sharing() {
            return Err(WasmError::InvalidOperation(
                "Type is not valid for sharing".to_string()
            ));
        }

        unsafe {
            Ok(Self::from_raw_parts(slice.as_ptr(), slice.len()))
        }
    }

    /// Gets the length of the slice
    pub fn len(&self) -> usize {
        self.len
    }

    /// Checks if the slice is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Gets a pointer to the start of the slice
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Gets a mutable pointer to the start of the slice
    /// 
    /// Returns error if threading is not available
    pub fn as_mut_ptr(&self) -> Result<*mut T, WasmError> {
        let caps = get_host_capabilities();
        
        if !caps.threading {
            return Err(WasmError::ThreadingError(
                threading::ThreadError::ThreadingNotSupported
            ));
        }

        Ok(self.ptr.as_ptr())
    }

    /// Gets the slice as a regular slice (read-only access)
    pub fn as_slice(&self) -> &'a [T] {
        unsafe {
            slice::from_raw_parts(self.ptr.as_ptr(), self.len)
        }
    }

    /// Gets the slice as a mutable slice
    /// 
    /// Returns error if threading is not available
    pub fn as_slice_mut(&self) -> Result<&'a mut [T], WasmError> {
        let caps = get_host_capabilities();
        
        if !caps.threading {
            return Err(WasmError::ThreadingError(
                threading::ThreadError::ThreadingNotSupported
            ));
        }

        unsafe {
            Ok(slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len))
        }
    }

    /// Gets the value at the given index
    /// 
    /// Performs bounds checking
    pub fn get(&self, index: usize) -> Result<&'a T, WasmError> {
        if index >= self.len {
            return Err(WasmError::OutOfBounds);
        }

        unsafe {
            Ok(&*self.ptr.as_ptr().add(index))
        }
    }

    /// Gets a mutable reference to the value at the given index
    /// 
    /// Returns error if threading is not available or out of bounds
    pub fn get_mut(&self, index: usize) -> Result<&'a mut T, WasmError> {
        let caps = get_host_capabilities();
        
        if !caps.threading {
            return Err(WasmError::ThreadingError(
                threading::ThreadError::ThreadingNotSupported
            ));
        }

        if index >= self.len {
            return Err(WasmError::OutOfBounds);
        }

        unsafe {
            Ok(&mut *self.ptr.as_ptr().add(index))
        }
    }

    /// Splits the slice at the given index
    pub fn split_at(&self, mid: usize) -> (Self, Self) {
        assert!(mid <= self.len, "split_at out of bounds");

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

    /// Gets a subslice of the shared slice
    pub fn get_slice(&self, range: core::ops::Range<usize>) -> Result<Self, WasmError> {
        if range.end > self.len {
            return Err(WasmError::OutOfBounds);
        }

        unsafe {
            Ok(SharedSlice::from_raw_parts(
                self.ptr.as_ptr().add(range.start),
                range.end - range.start
            ))
        }
    }
}

impl<'a, T: Pod> Clone for SharedSlice<'a, T> {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            len: self.len,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T: Pod> Copy for SharedSlice<'a, T> {}

impl<'a, T: Pod> Deref for SharedSlice<'a, T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        unsafe {
            slice::from_raw_parts(self.ptr.as_ptr(), self.len)
        }
    }
}

impl<'a, T: Pod> DerefMut for SharedSlice<'a, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        let caps = get_host_capabilities();
        
        if !caps.threading {
            // This would panic in debug builds, but we return empty slice in release
            panic!("Cannot get mutable reference in non-threading environment");
        }

        unsafe {
            slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len)
        }
    }
}

impl<'a, T: Pod> Index<usize> for SharedSlice<'a, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("Index out of bounds")
    }
}

impl<'a, T: Pod> IndexMut<usize> for SharedSlice<'a, T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index).expect("Index out of bounds")
    }
}

/// Iterator for SharedSlice
pub struct SharedSliceIter<'a, T: Pod> {
    slice: SharedSlice<'a, T>,
    index: usize,
}

impl<'a, T: Pod> Iterator for SharedSliceIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.slice.len {
            None
        } else {
            let item = unsafe {
                &*self.slice.ptr.as_ptr().add(self.index)
            };
            self.index += 1;
            Some(item)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.slice.len - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a, T: Pod> IntoIterator for SharedSlice<'a, T> {
    type Item = &'a T;
    type IntoIter = SharedSliceIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        SharedSliceIter {
            slice: self,
            index: 0,
        }
    }
}

/// Shared memory region with type-safe access
pub struct SharedMemory<T: Pod> {
    /// Base pointer to the memory region
    base: NonNull<T>,
    /// Size of the memory region in bytes
    size: usize,
    /// Whether this memory allows mutable access
    mutable: bool,
}

impl<T: Pod> SharedMemory<T> {
    /// Creates a new shared memory region
    /// 
    /// Returns error if memory allocation fails or type is not Pod
    pub fn new(size: usize, mutable: bool) -> Result<Self, WasmError> {
        if !T::is_valid_for_sharing() {
            return Err(WasmError::InvalidOperation(
                "Type is not valid for sharing".to_string()
            ));
        }

        // Allocate shared memory
        let base = memory::allocate_shared(size * mem::size_of::<T>())
            .map_err(|e| WasmError::MemoryError(e))?;

        Ok(Self {
            base: NonNull::new(base as *mut T)
                .expect("Shared memory allocation returned null"),
            size,
            mutable,
        })
    }

    /// Gets the size of the memory region
    pub fn size(&self) -> usize {
        self.size
    }

    /// Gets whether the memory is mutable
    pub fn is_mutable(&self) -> bool {
        self.mutable
    }

    /// Gets a pointer to the memory region
    pub fn as_ptr(&self) -> *const T {
        self.base.as_ptr()
    }

    /// Gets a mutable pointer to the memory region
    /// 
    /// Returns error if memory is not mutable
    pub fn as_mut_ptr(&self) -> Result<*mut T, WasmError> {
        if !self.mutable {
            return Err(WasmError::InvalidOperation(
                "Memory is not mutable".to_string()
            ));
        }

        Ok(self.base.as_ptr())
    }

    /// Creates a SharedSlice covering the entire memory region
    pub fn as_shared_slice(&self) -> SharedSlice<T> {
        unsafe {
            SharedSlice::from_raw_parts(self.base.as_ptr(), self.size / mem::size_of::<T>())
        }
    }

    /// Creates a SharedSlice covering part of the memory region
    pub fn as_shared_slice_range(&self, start: usize, len: usize) -> Result<SharedSlice<T>, WasmError> {
        let total_elements = self.size / mem::size_of::<T>();
        
        if start >= total_elements || start + len > total_elements {
            return Err(WasmError::OutOfBounds);
        }

        unsafe {
            Ok(SharedSlice::from_raw_parts(
                self.base.as_ptr().add(start),
                len
            ))
        }
    }
}

impl<T: Pod> Drop for SharedMemory<T> {
    fn drop(&mut self) {
        // Deallocate shared memory
        unsafe {
            memory::deallocate_shared(self.base.as_ptr() as *mut u8, self.size);
        }
    }
}

/// Marker trait for types that can be used in JavaScript interop
pub trait JsInteropSafe {
    /// Validates that the type can be safely used in JavaScript interop
    fn validate_js_interop() -> Result<(), WasmError>;
}

// Implement JsInteropSafe for Pod types
impl<T: Pod> JsInteropSafe for T {
    fn validate_js_interop() -> Result<(), WasmError> {
        if !T::is_valid_for_sharing() {
            return Err(WasmError::InvalidOperation(
                "Type is not valid for JavaScript interop".to_string()
            ));
        }
        Ok(())
    }
}

// Implement JsInteropSafe for ExternRef
impl<T> JsInteropSafe for ExternRef<T> {
    fn validate_js_interop() -> Result<(), WasmError> {
        // ExternRef is always safe for JS interop by design
        Ok(())
    }
}

/// Trait for converting types to/from JavaScript values
pub trait JsValue {
    /// Convert from JavaScript value
    fn from_js_value(value: host::JsValue) -> Result<Self, WasmError>
    where
        Self: Sized;
    
    /// Convert to JavaScript value
    fn to_js_value(&self) -> Result<host::JsValue, WasmError>;
}

// Implement JsValue for primitive types
impl JsValue for i32 {
    fn from_js_value(value: host::JsValue) -> Result<Self, WasmError> {
        host::convert_js_to_i32(value)
            .map_err(|e| WasmError::HostError(e))
    }

    fn to_js_value(&self) -> Result<host::JsValue, WasmError> {
        host::convert_i32_to_js(*self)
            .map_err(|e| WasmError::HostError(e))
    }
}

impl JsValue for f64 {
    fn from_js_value(value: host::JsValue) -> Result<Self, WasmError> {
        host::convert_js_to_f64(value)
            .map_err(|e| WasmError::HostError(e))
    }

    fn to_js_value(&self) -> Result<host::JsValue, WasmError> {
        host::convert_f64_to_js(*self)
            .map_err(|e| WasmError::HostError(e))
    }
}

impl JsValue for bool {
    fn from_js_value(value: host::JsValue) -> Result<Self, WasmError> {
        host::convert_js_to_bool(value)
            .map_err(|e| WasmError::HostError(e))
    }

    fn to_js_value(&self) -> Result<host::JsValue, WasmError> {
        host::convert_bool_to_js(*self)
            .map_err(|e| WasmError::HostError(e))
    }
}

/// WASM runtime initialization
pub fn initialize() -> Result<(), WasmError> {
    // Initialize host profile detection
    let _profile = host::detect_host_profile();
    
    // Initialize memory management
    memory::initialize_memory_management()?;
    
    // Initialize threading support
    threading::initialize_threading_support()?;
    
    // Initialize component model support
    component::initialize_component_support()?;
    
    Ok(())
}

/// Gets the current WASM runtime version
pub fn runtime_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Checks if a capability is available in the current host
pub fn has_capability(cap: &str) -> bool {
    let caps = get_host_capabilities();
    
    match cap {
        "threading" => caps.threading,
        "component_model" => caps.component_model,
        "memory_regions" => caps.memory_regions,
        "js_interop" => caps.js_interop,
        "external_functions" => caps.external_functions,
        "file_system" => caps.file_system,
        "network" => caps.network,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_externref_creation() {
        let extern_ref = ExternRef::<i32>::from_handle(42);
        assert_eq!(extern_ref.handle(), 42);
        assert!(!extern_ref.is_null());

        let null_ref = ExternRef::<i32>::null();
        assert_eq!(null_ref.handle(), 0);
        assert!(null_ref.is_null());
    }

    #[test]
    fn test_externref_equality() {
        let ref1 = ExternRef::<i32>::from_handle(42);
        let ref2 = ExternRef::<i32>::from_handle(42);
        let ref3 = ExternRef::<i32>::from_handle(43);

        assert_eq!(ref1, ref2);
        assert_ne!(ref1, ref3);
    }

    #[test]
    fn test_funcref_creation() {
        let func_ref = unsafe { FuncRef::<(i32, i32), i32>::from_index(10) };
        assert_eq!(func_ref.index(), 10);
        assert!(!func_ref.is_null());

        let null_func = FuncRef::<(), ()>::null();
        assert_eq!(null_func.index(), 0);
        assert!(null_func.is_null());
    }

    #[test]
    fn test_sharedslice_creation() {
        let data = [1, 2, 3, 4, 5];
        let shared_slice = SharedSlice::from_slice(&data).unwrap();
        
        assert_eq!(shared_slice.len(), 5);
        assert!(!shared_slice.is_empty());
        assert_eq!(shared_slice.as_slice(), &data);
    }

    #[test]
    fn test_sharedslice_access() {
        let data = [10, 20, 30];
        let shared_slice = SharedSlice::from_slice(&data).unwrap();
        
        assert_eq!(*shared_slice.get(0).unwrap(), 10);
        assert_eq!(*shared_slice.get(1).unwrap(), 20);
        assert_eq!(*shared_slice.get(2).unwrap(), 30);
        
        assert!(shared_slice.get(3).is_err());
    }

    #[test]
    fn test_sharedslice_iteration() {
        let data = [1, 2, 3];
        let shared_slice = SharedSlice::from_slice(&data).unwrap();
        
        let collected: Vec<_> = shared_slice.into_iter().collect();
        assert_eq!(collected, vec![&1, &2, &3]);
    }

    #[test]
    fn test_sharedmemory_creation() {
        let shared_mem = SharedMemory::<i32>::new(100, true).unwrap();
        assert_eq!(shared_mem.size(), 100 * mem::size_of::<i32>());
        assert!(shared_mem.is_mutable());
        
        let shared_slice = shared_mem.as_shared_slice();
        assert_eq!(shared_slice.len(), 100);
    }

    #[test]
    fn test_pod_trait() {
        // All primitive types should implement Pod
        assert!(u32::is_valid_for_sharing());
        assert!(i64::is_valid_for_sharing());
        assert!(f32::is_valid_for_sharing());
        assert!(bool::is_valid_for_sharing());
    }

    #[test]
    fn test_jsinterop_trait() {
        // Pod types should be JsInteropSafe
        assert!(i32::validate_js_interop().is_ok());
        assert!(f64::validate_js_interop().is_ok());
        
        // ExternRef should be JsInteropSafe
        assert!(ExternRef::<i32>::validate_js_interop().is_ok());
    }

    #[test]
    fn test_jsvalue_trait() {
        let value = 42i32;
        let js_value = value.to_js_value().unwrap();
        let converted_back = i32::from_js_value(js_value).unwrap();
        
        assert_eq!(value, converted_back);
    }

    #[test]
    fn test_runtime_initialization() {
        // Should initialize without errors
        assert!(initialize().is_ok());
        
        // Version should be available
        assert!(!runtime_version().is_empty());
    }

    #[test]
    fn test_capability_checking() {
        // Should check known capabilities
        let result = has_capability("threading");
        // Result depends on host environment, but should not panic
        let _ = result;
        
        // Unknown capability should return false
        assert!(!has_capability("unknown_capability"));
    }
}
