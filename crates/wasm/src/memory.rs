//! Memory management abstractions for WasmRust
//! 
//! This module provides safe memory management primitives including
//! SharedSlice for concurrent access, memory regions with intent
//! validation, and scoped arenas for temporary allocations.

use crate::Pod;
use crate::host::{get_host_capabilities, HostCapabilities};
use core::ptr::NonNull;
use core::sync::atomic::{AtomicUsize, AtomicPtr, Ordering};
use core::marker::PhantomData;
use core::ops::{Deref, DerefMut, Index, IndexMut};
use core::slice;

/// Global memory allocation tracking
static ALLOCATED_MEMORY: AtomicUsize = AtomicUsize::new(0);

/// Shared memory region with concurrent access protection
/// 
/// `SharedMemory<T>` provides a thread-safe wrapper around
/// memory regions that can be safely accessed by multiple threads
/// when the host supports threading.
pub struct SharedMemory<T: Pod> {
    ptr: NonNull<T>,
    len: usize,
    capacity: usize,
    ref_count: AtomicUsize,
    _marker: PhantomData<T>,
}

impl<T: Pod> SharedMemory<T> {
    /// Creates a new shared memory region with given capacity
    pub fn new(capacity: usize) -> Result<Self, MemoryError> {
        if capacity == 0 {
            return Err(MemoryError::InvalidSize);
        }

        let layout = unsafe { core::alloc::Layout::array::<T>(capacity) }
            .map_err(|_| MemoryError::InvalidSize)?;
        
        let ptr = unsafe { core::alloc::alloc(layout) as *mut T };
        if ptr.is_null() {
            return Err(MemoryError::OutOfMemory);
        }

        ALLOCATED_MEMORY.fetch_add(layout.size(), Ordering::Relaxed);

        Ok(Self {
            ptr: NonNull::new(ptr).expect("null pointer after allocation"),
            len: 0,
            capacity,
            ref_count: AtomicUsize::new(1),
            _marker: PhantomData,
        })
    }

    /// Creates shared memory from existing slice
    pub fn from_slice(slice: &[T]) -> Result<Self, MemoryError> {
        let mut shared = Self::new(slice.len())?;
        unsafe {
            core::ptr::copy_nonoverlapping(
                slice.as_ptr(),
                shared.ptr.as_ptr(),
                slice.len()
            );
        }
        shared.len = slice.len();
        Ok(shared)
    }

    /// Returns the current length of the shared memory
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the capacity of the shared memory
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns true if the shared memory is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Adds a reference to the shared memory
    pub fn add_ref(&self) {
        self.ref_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Removes a reference and potentially deallocates memory
    pub fn remove_ref(&self) {
        if self.ref_count.fetch_sub(1, Ordering::AcqRel) == 1 {
            unsafe { self.deallocate() };
        }
    }

    /// Gets the current reference count
    pub fn ref_count(&self) -> usize {
        self.ref_count.load(Ordering::Acquire)
    }

    /// Gets a shared slice view of the memory
    pub fn as_shared_slice(&self) -> crate::SharedSlice<'_, T> {
        unsafe {
            crate::SharedSlice::from_raw_parts(self.ptr.as_ptr(), self.len)
        }
    }

    /// Gets a mutable shared slice view (requires exclusive access)
    pub fn as_mut_shared_slice(&mut self) -> crate::SharedSliceMut<'_, T> {
        unsafe {
            crate::SharedSliceMut::from_raw_parts(self.ptr.as_ptr(), self.len)
        }
    }

    /// Tries to reserve additional capacity
    pub fn reserve(&mut self, additional: usize) -> Result<(), MemoryError> {
        let new_capacity = self.len.checked_add(additional)
            .ok_or(MemoryError::InvalidSize)?;

        if new_capacity <= self.capacity {
            return Ok(());
        }

        self.reallocate(new_capacity)
    }

    /// Shrinks the capacity to match the current length
    pub fn shrink_to_fit(&mut self) -> Result<(), MemoryError> {
        if self.capacity == self.len {
            return Ok(());
        }

        self.reallocate(self.len)
    }

    unsafe fn reallocate(&mut self, new_capacity: usize) -> Result<(), MemoryError> {
        let new_layout = core::alloc::Layout::array::<T>(new_capacity)
            .map_err(|_| MemoryError::InvalidSize)?;
        
        let new_ptr = core::alloc::realloc(
            self.ptr.as_ptr() as *mut u8,
            core::alloc::Layout::array::<T>(self.capacity)
                .map_err(|_| MemoryError::InvalidSize)?,
            new_layout.size(),
        ) as *mut T;

        if new_ptr.is_null() {
            return Err(MemoryError::OutOfMemory);
        }

        // Update allocation tracking
        let old_size = core::alloc::Layout::array::<T>(self.capacity)
            .map_err(|_| MemoryError::InvalidSize)?.size();
        let new_size = new_layout.size();
        ALLOCATED_MEMORY.fetch_sub(old_size, Ordering::Relaxed);
        ALLOCATED_MEMORY.fetch_add(new_size, Ordering::Relaxed);

        self.ptr = NonNull::new(new_ptr).expect("null pointer after reallocation");
        self.capacity = new_capacity;

        // Ensure length doesn't exceed new capacity
        if self.len > new_capacity {
            self.len = new_capacity;
        }

        Ok(())
    }

    unsafe fn deallocate(&self) {
        if !self.ptr.as_ptr().is_null() {
            let layout = core::alloc::Layout::array::<T>(self.capacity)
                .unwrap_or_else(|_| core::alloc::Layout::new::<T>());
            
            core::alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout);
            ALLOCATED_MEMORY.fetch_sub(layout.size(), Ordering::Relaxed);
        }
    }
}

impl<T: Pod> Drop for SharedMemory<T> {
    fn drop(&mut self) {
        unsafe { self.deallocate() };
    }
}

impl<T: Pod> Clone for SharedMemory<T> {
    fn clone(&self) -> Self {
        self.add_ref();
        Self {
            ptr: self.ptr,
            len: self.len,
            capacity: self.capacity,
            ref_count: AtomicUsize::new(self.ref_count()),
            _marker: PhantomData,
        }
    }
}

/// Mutable version of SharedSlice for exclusive access
pub struct SharedSliceMut<'a, T: Pod> {
    ptr: NonNull<T>,
    len: usize,
    _marker: PhantomData<&'a mut [T]>,
}

impl<'a, T: Pod> SharedSliceMut<'a, T> {
    /// Creates a mutable shared slice from raw parts
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
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len {
            unsafe { Some(&mut *self.ptr.as_ptr().add(index)) }
        } else {
            None
        }
    }

    /// Splits the slice into two at the given index
    pub fn split_at(self, mid: usize) -> (Self, Self) {
        assert!(mid <= self.len, "split index out of bounds");
        
        let left = unsafe {
            SharedSliceMut::from_raw_parts(self.ptr.as_ptr(), mid)
        };
        
        let right = unsafe {
            SharedSliceMut::from_raw_parts(
                self.ptr.as_ptr().add(mid),
                self.len - mid
            )
        };
        
        (left, right)
    }

    /// Converts to immutable shared slice
    pub fn into_shared_slice(self) -> crate::SharedSlice<'a, T> {
        unsafe {
            crate::SharedSlice::from_raw_parts(self.ptr.as_ptr(), self.len)
        }
    }
}

impl<'a, T: Pod> Deref for SharedSliceMut<'a, T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        unsafe {
            slice::from_raw_parts(self.ptr.as_ptr(), self.len)
        }
    }
}

impl<'a, T: Pod> DerefMut for SharedSliceMut<'a, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe {
            slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len)
        }
    }
}

/// Memory region with intent validation
/// 
/// Memory regions allow specifying requirements like data residency
/// and encryption that are validated by the host at load time.
#[repr(transparent)]
pub struct MemoryRegion<T: Pod> {
    data: T,
}

impl<T: Pod> MemoryRegion<T> {
    /// Creates a new memory region with intent
    pub fn new(data: T, intent: MemoryIntent) -> Self {
        // In a real implementation, this would register the intent
        // with the host for validation
        Self { data }
    }

    /// Gets a reference to the underlying data
    pub fn get(&self) -> &T {
        &self.data
    }

    /// Gets a mutable reference to the underlying data
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.data
    }

    /// Consumes the memory region and returns the data
    pub fn into_inner(self) -> T {
        self.data
    }
}

/// Intent specification for memory regions
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryIntent {
    pub region: Option<String>,
    pub encryption: Option<EncryptionType>,
    pub durability: DurabilityLevel,
    pub access_pattern: AccessPattern,
}

impl MemoryIntent {
    /// Creates a new memory intent
    pub fn new() -> Self {
        Self {
            region: None,
            encryption: None,
            durability: DurabilityLevel::Volatile,
            access_pattern: AccessPattern::Sequential,
        }
    }

    /// Sets the geographic region requirement
    pub fn region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Sets the encryption requirement
    pub fn encryption(mut self, encryption: EncryptionType) -> Self {
        self.encryption = Some(encryption);
        self
    }

    /// Sets the durability level
    pub fn durability(mut self, durability: DurabilityLevel) -> Self {
        self.durability = durability;
        self
    }

    /// Sets the access pattern hint
    pub fn access_pattern(mut self, pattern: AccessPattern) -> Self {
        self.access_pattern = pattern;
        self
    }

    /// Validates the intent against host capabilities
    pub fn validate(&self) -> Result<(), MemoryError> {
        let caps = get_host_capabilities();

        // Check region support
        if self.region.is_some() && !caps.memory_regions {
            return Err(MemoryError::UnsupportedIntent("memory regions".to_string()));
        }

        // Check encryption support
        if self.encryption.is_some() && !caps.memory_regions {
            return Err(MemoryError::UnsupportedIntent("encryption".to_string()));
        }

        Ok(())
    }
}

/// Encryption types for memory regions
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncryptionType {
    None,
    AES128GCM,
    AES256GCM,
    ChaCha20Poly1305,
}

/// Durability levels for memory regions
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DurabilityLevel {
    Volatile,
    Persistent,
    Durable,
}

/// Access pattern hints for optimization
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccessPattern {
    Sequential,
    Random,
    Streaming,
    RandomAccess,
}

/// Scoped arena for temporary allocations
/// 
/// ScopedArena provides fast allocation for temporary data
/// that is automatically cleaned up when the scope ends.
pub struct ScopedArena<'a> {
    buffer: Vec<u8>,
    _marker: PhantomData<&'a mut ()>,
}

impl<'a> ScopedArena<'a> {
    /// Creates a new scoped arena
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// Allocates memory in the arena
    pub fn alloc<T: Pod>(&mut self, value: T) -> &'a mut T {
        let ptr = self.buffer.as_mut_ptr() as *mut T;
        let size = core::mem::size_of::<T>();
        
        // Ensure alignment
        let align = core::mem::align_of::<T>();
        let offset = (self.buffer.len() + align - 1) & !(align - 1);
        
        self.buffer.resize(offset + size, 0);
        let ptr = unsafe { self.buffer.as_mut_ptr().add(offset) as *mut T };
        
        unsafe {
            ptr.write(value);
            &mut *ptr
        }
    }

    /// Allocates a slice in the arena
    pub fn alloc_slice<T: Pod>(&mut self, slice: &[T]) -> &'a [T] {
        let ptr = self.buffer.as_mut_ptr() as *mut T;
        let size = slice.len() * core::mem::size_of::<T>();
        let align = core::mem::align_of::<T>();
        let offset = (self.buffer.len() + align - 1) & !(align - 1);
        
        self.buffer.resize(offset + size, 0);
        let ptr = unsafe { self.buffer.as_mut_ptr().add(offset) as *mut T };
        
        unsafe {
            core::ptr::copy_nonoverlapping(slice.as_ptr(), ptr, slice.len());
            slice::from_raw_parts(ptr, slice.len())
        }
    }

    /// Clears the arena (releases all allocations)
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Returns the total allocated bytes
    pub fn allocated_bytes(&self) -> usize {
        self.buffer.len()
    }
}

impl<'a> Default for ScopedArena<'a> {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory-related errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryError {
    /// Invalid memory size requested
    InvalidSize,
    /// Out of memory
    OutOfMemory,
    /// Memory intent not supported by host
    UnsupportedIntent(String),
    /// Memory allocation failed
    AllocationFailed,
    /// Memory region validation failed
    ValidationFailed(String),
}

impl core::fmt::Display for MemoryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MemoryError::InvalidSize => write!(f, "Invalid memory size"),
            MemoryError::OutOfMemory => write!(f, "Out of memory"),
            MemoryError::UnsupportedIntent(intent) => {
                write!(f, "Unsupported memory intent: {}", intent)
            }
            MemoryError::AllocationFailed => write!(f, "Memory allocation failed"),
            MemoryError::ValidationFailed(msg) => {
                write!(f, "Memory validation failed: {}", msg)
            }
        }
    }
}

/// Global memory statistics
pub fn get_memory_stats() -> MemoryStats {
    MemoryStats {
        allocated_bytes: ALLOCATED_MEMORY.load(Ordering::Acquire),
        peak_allocated_bytes: 0, // Would need additional tracking
    }
}

/// Initialize memory management system
pub fn initialize_memory_management() -> Result<(), MemoryError> {
    // Initialize global memory tracking
    ALLOCATED_MEMORY.store(0, Ordering::Relaxed);
    
    // Initialize memory region tracking
    // In a real implementation, this would set up
    // memory region validation and capability checking
    Ok(())
}

/// Allocate shared memory for SharedMemory
pub fn allocate_shared(size: usize) -> Result<*mut u8, MemoryError> {
    let layout = unsafe { 
        core::alloc::Layout::from_size_align(size, 8) 
            .map_err(|_| MemoryError::InvalidSize)? 
    };
    
    let ptr = unsafe { core::alloc::alloc(layout) };
    
    if ptr.is_null() {
        return Err(MemoryError::OutOfMemory);
    }
    
    ALLOCATED_MEMORY.fetch_add(size, Ordering::Relaxed);
    Ok(ptr)
}

/// Deallocate shared memory
pub fn deallocate_shared(ptr: *mut u8, size: usize) {
    if !ptr.is_null() && size > 0 {
        let layout = unsafe {
            core::alloc::Layout::from_size_align(size, 8)
                .unwrap_or_else(|_| core::alloc::Layout::new::<u8>())
        };
        
        unsafe {
            core::alloc::dealloc(ptr, layout);
        }
        
        ALLOCATED_MEMORY.fetch_sub(size, Ordering::Relaxed);
    }
}

/// Validate memory intent against host capabilities
pub fn validate_memory_intent(intent: &MemoryIntent) -> Result<(), MemoryError> {
    intent.validate()
}

/// Memory usage statistics
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryStats {
    pub allocated_bytes: usize,
    pub peak_allocated_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_memory_creation() {
        let shared = SharedMemory::<u32>::new(10).unwrap();
        assert_eq!(shared.len(), 0);
        assert_eq!(shared.capacity(), 10);
        assert_eq!(shared.ref_count(), 1);
    }

    #[test]
    fn test_shared_memory_from_slice() {
        let data = [1u32, 2u32, 3u32, 4u32];
        let shared = SharedMemory::from_slice(&data).unwrap();
        assert_eq!(shared.len(), 4);
        assert_eq!(shared.capacity(), 4);
        
        let slice = shared.as_shared_slice();
        assert_eq!(slice.len(), 4);
        assert_eq!(slice.get(0), Some(&1u32));
    }

    #[test]
    fn test_shared_memory_reference_counting() {
        let shared = SharedMemory::<u32>::new(10).unwrap();
        assert_eq!(shared.ref_count(), 1);
        
        shared.add_ref();
        assert_eq!(shared.ref_count(), 2);
        
        shared.remove_ref();
        assert_eq!(shared.ref_count(), 1);
    }

    #[test]
    fn test_shared_slice_mut() {
        let mut data = [1u32, 2u32, 3u32];
        let mut slice = unsafe {
            SharedSliceMut::from_raw_parts(data.as_mut_ptr(), data.len())
        };
        
        assert_eq!(slice.len(), 3);
        assert_eq!(slice.get(0), Some(&1u32));
        assert_eq!(slice.get_mut(0), Some(&mut 1u32));
        
        slice[0] = 42;
        assert_eq!(data[0], 42);
    }

    #[test]
    fn test_memory_intent() {
        let intent = MemoryIntent::new()
            .region("eu-west-1")
            .encryption(EncryptionType::AES256GCM)
            .durability(DurabilityLevel::Persistent)
            .access_pattern(AccessPattern::Random);
        
        assert_eq!(intent.region, Some("eu-west-1".to_string()));
        assert_eq!(intent.encryption, Some(EncryptionType::AES256GCM));
        assert_eq!(intent.durability, DurabilityLevel::Persistent);
        assert_eq!(intent.access_pattern, AccessPattern::Random);
    }

    #[test]
    fn test_scoped_arena() {
        let mut arena = ScopedArena::new();
        
        let value = arena.alloc(42u32);
        assert_eq!(*value, 42);
        
        let slice = arena.alloc_slice(&[1u8, 2u8, 3u8]);
        assert_eq!(slice, &[1u8, 2u8, 3u8]);
        
        assert!(arena.allocated_bytes() > 0);
        
        arena.clear();
        assert_eq!(arena.allocated_bytes(), 0);
    }

    #[test]
    fn test_memory_stats() {
        let stats_before = get_memory_stats();
        
        let _shared = SharedMemory::<u32>::new(100).unwrap();
        
        let stats_after = get_memory_stats();
        assert!(stats_after.allocated_bytes > stats_before.allocated_bytes);
    }
}
