//! Integration tests for the process-wide thread count and the single
//! `threading::global_pool` it backs.
//!
//! These mutate process-wide state (the atomic count AND the shared pool),
//! so they live in their own test BINARY (`cargo test` gives each
//! `tests/*.rs` file a fresh process) rather than in `src/threading.rs`'s
//! unit tests, which run in the shared lib test binary alongside every other
//! unit test and must never touch either.
//!
//! `cargo test` runs the tests within *this* binary in parallel by default
//! too, so each test here still restores the global count (and therefore the
//! pool) to its original value before returning, serialized against every
//! other test in this binary via a single mutex — they all share the one
//! process-wide pool, so interleaved mutation would make results depend on
//! run order.

use std::sync::Mutex;
use std::time::Duration;

use exr_core::error::ExrError;
use exr_core::threading::{
    global_pool, global_thread_count, run_on_global_pool, set_global_thread_count, ThreadCount,
};

// Serializes the tests in this binary against each other, since they all
// share the one process-wide `GLOBAL_THREAD_COUNT` atomic and the one
// process-wide pool it builds.
static LOCK: Mutex<()> = Mutex::new(());

#[test]
fn default_is_zero_serial_no_pool() {
    let _guard = LOCK.lock().unwrap();
    assert_eq!(global_thread_count(), 0);
    assert!(global_pool().unwrap().is_none());
}

#[test]
fn set_and_read_round_trips() {
    let _guard = LOCK.lock().unwrap();
    let before = global_thread_count();
    set_global_thread_count(4).unwrap();
    assert_eq!(global_thread_count(), 4);
    set_global_thread_count(0).unwrap();
    assert_eq!(global_thread_count(), 0);
    set_global_thread_count(before).unwrap();
}

#[test]
fn negative_count_is_rejected_and_leaves_global_unchanged() {
    let _guard = LOCK.lock().unwrap();
    let before = global_thread_count();
    match set_global_thread_count(-1) {
        Err(ExrError::Arg(_)) => {}
        other => panic!("expected ExrError::Arg, got {other:?}"),
    }
    assert_eq!(
        global_thread_count(),
        before,
        "a rejected set must not mutate the global count"
    );
}

#[test]
fn zero_and_one_build_no_pool() {
    let _guard = LOCK.lock().unwrap();
    let before = global_thread_count();
    for n in [0, 1] {
        set_global_thread_count(n).unwrap();
        assert!(global_pool().unwrap().is_none(), "n={n} should not build a pool");
    }
    set_global_thread_count(before).unwrap();
}

#[test]
fn pool_sizing_matches_requested_count() {
    let _guard = LOCK.lock().unwrap();
    let before = global_thread_count();
    for n in [2, 4, 8] {
        set_global_thread_count(n).unwrap();
        let pool = global_pool().unwrap().expect("n>1 must build a pool");
        assert_eq!(pool.current_num_threads(), n as usize);
    }
    set_global_thread_count(before).unwrap();
}

#[test]
fn resizing_between_reads_is_observed_immediately() {
    let _guard = LOCK.lock().unwrap();
    let before = global_thread_count();
    set_global_thread_count(2).unwrap();
    assert_eq!(
        global_pool().unwrap().unwrap().current_num_threads(),
        2
    );
    set_global_thread_count(8).unwrap();
    assert_eq!(
        global_pool().unwrap().unwrap().current_num_threads(),
        8
    );
    set_global_thread_count(before).unwrap();
}

#[test]
fn run_on_global_pool_serial_runs_inline() {
    let _guard = LOCK.lock().unwrap();
    let before = global_thread_count();
    set_global_thread_count(0).unwrap();
    let count = ThreadCount::capture(None).unwrap();
    assert!(!count.is_parallel());
    let result = run_on_global_pool(count, || 21 * 2).unwrap();
    assert_eq!(result, 42);
    set_global_thread_count(before).unwrap();
}

#[test]
fn run_on_global_pool_parallel_uses_the_shared_pool() {
    let _guard = LOCK.lock().unwrap();
    let before = global_thread_count();
    set_global_thread_count(4).unwrap();
    let count = ThreadCount::capture(None).unwrap();
    assert!(count.is_parallel());
    let sum: i32 = run_on_global_pool(count, || (1..=100).sum()).unwrap();
    assert_eq!(sum, 5050);
    set_global_thread_count(before).unwrap();
}

#[test]
fn resize_during_in_flight_work_does_not_break_it() {
    // The safety property under test: a caller that already holds the pool
    // it started `run_on_global_pool` on must be unaffected by a concurrent
    // `set_global_thread_count` replacing the global pool underneath it
    // (`global_pool` clones an `Arc<rayon::ThreadPool>` out from under its
    // lock, so the in-flight caller's clone keeps the OLD pool alive and
    // running until it finishes, standard `Arc` semantics — see
    // `threading::global_pool`'s rustdoc).
    let _guard = LOCK.lock().unwrap();
    let before = global_thread_count();
    set_global_thread_count(2).unwrap();
    let count = ThreadCount::capture(None).unwrap();

    let handle = std::thread::spawn(move || {
        run_on_global_pool(count, || {
            // Long enough that the main thread's resize below reliably
            // lands WHILE this task is still running on the old pool.
            std::thread::sleep(Duration::from_millis(200));
            42
        })
    });

    // Give the spawned task time to call `global_pool()` and clone its Arc
    // to the size-2 pool before we resize out from under it.
    std::thread::sleep(Duration::from_millis(50));
    set_global_thread_count(6).unwrap();

    let result = handle.join().expect("in-flight task must not panic");
    assert_eq!(
        result.unwrap(),
        42,
        "in-flight work must complete correctly even though the pool was resized under it"
    );

    // The NEXT caller must see the new size — the resize was not lost.
    assert_eq!(
        global_pool().unwrap().unwrap().current_num_threads(),
        6,
        "a later caller must observe the post-resize pool"
    );

    set_global_thread_count(before).unwrap();
}

#[test]
fn a_files_captured_count_is_independent_of_a_later_global_change() {
    let _guard = LOCK.lock().unwrap();
    let before = global_thread_count();
    // A file opened while the global is 4 captures 4 for its own lifetime...
    set_global_thread_count(4).unwrap();
    let captured = ThreadCount::capture(None).unwrap();
    assert_eq!(captured.get(), 4);

    // ...even after some other, later-opened file raises the global.
    set_global_thread_count(8).unwrap();
    assert_eq!(captured.get(), 4, "an already-captured ThreadCount must not track later changes");

    set_global_thread_count(before).unwrap();
}
