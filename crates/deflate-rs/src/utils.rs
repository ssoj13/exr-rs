//! Port of libdeflate `lib/utils.c` — allocator indirection.
//! <https://github.com/ebiggers/libdeflate/blob/v1.26/lib/utils.c>
//!
//! `utils.c` exists to let C callers swap libdeflate's `malloc`/`free` (via
//! `libdeflate_set_memory_allocator`) and to provide over-aligned allocations
//! (`libdeflate_aligned_malloc`/`_free`) for the compressor/decompressor structs and
//! SIMD scratch buffers.
//!
//! **None of it is needed in the Rust port — documented here, not ported:**
//! - The allocator-override API (`libdeflate_set_memory_allocator`,
//!   `libdeflate_default_malloc_func`/`_free_func`, the `malloc_func`/`free_func`
//!   typedefs) is a C mechanism with no portable, safe Rust equivalent (it requires raw
//!   function pointers / `unsafe` FFI). deflate-rs allocates through ordinary `Vec`/`Box`,
//!   i.e. the Rust global allocator. If a custom allocator is ever required, it is the
//!   crate user's `#[global_allocator]`, not a deflate-rs API.
//! - `libdeflate_aligned_malloc`/`_free` provide over-aligned heap blocks for the
//!   `libdeflate_compressor`/`libdeflate_decompressor` objects. In Rust those objects are
//!   plain structs owned by `Box`/the caller; alignment is the type's own, so no aligned
//!   allocator is needed. (SIMD scratch buffers, when those tiers land, will use
//!   `Vec`/`[T; N]` whose alignment the compiler guarantees.)
//! - `FREESTANDING` `memset`/`memcpy`/`memmove`/`memcmp` shims — provided by `core` in Rust.
//! - `LIBDEFLATE_ENABLE_ASSERTIONS` / `libdeflate_assertion_failed` — `debug_assert!` in Rust.
//!
//! This module is intentionally code-free; it records the porting decision so the mapping
//! from `utils.c` is explicit (skip-nothing accounting), not silently dropped.
