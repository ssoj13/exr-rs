//! Texture allocations retained until their last submitted GPU work completes.
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

struct Entry {
    _texture: wgpu::Texture,
    bytes: u64,
    complete: Arc<AtomicBool>,
}
#[derive(Default)]
pub struct RetiredTextures {
    entries: Vec<Entry>,
}
impl RetiredTextures {
    pub fn bytes(&self) -> u64 {
        self.entries.iter().map(|e| e.bytes).sum()
    }
    pub fn retire(&mut self, queue: &wgpu::Queue, texture: wgpu::Texture, bytes: u64) {
        let complete = Arc::new(AtomicBool::new(false));
        let signal = complete.clone();
        queue.on_submitted_work_done(move || signal.store(true, Ordering::Release));
        self.entries.push(Entry {
            _texture: texture,
            bytes,
            complete,
        });
    }
    pub fn reap(&mut self) {
        self.entries.retain(|e| !e.complete.load(Ordering::Acquire));
    }
}
