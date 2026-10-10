//! GPU tile atlas:`Rgba16Float` pages of
//! `TILE x TILE` slots, allocated lazily up to a VRAM budget, recycled
//! least-recently-used.
//!
//! [`SlotMap`] is the bookkeeping (pure, unit-tested); [`TileAtlas`] adds the
//! page textures and the uploads.

use std::hash::Hash;

use super::cache::Lru;
use super::grid::TILE;
use super::reduce::TileData;

/// Bytes per atlas texel (`Rgba16Float`).
pub const TEXEL_BYTES: u64 = 8;

/// Where a resident tile lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot {
    /// Page index.
    pub page: u32,
    /// Left texel of the slot in its page.
    pub x: u32,
    /// Top texel of the slot in its page.
    pub y: u32,
}

/// The outcome of [`SlotMap::alloc`].
#[derive(Debug, PartialEq, Eq)]
pub struct Alloc<K> {
    /// Slot index to fill.
    pub slot: u32,
    /// A page the caller must create first (the pool grew).
    pub new_page: Option<u32>,
    /// The key whose slot was recycled.
    pub evicted: Option<K>,
}

/// Slot bookkeeping: key -> slot index, LRU recycling, page growth up to a cap.
pub struct SlotMap<K> {
    lru: Lru<K, u32>,
    free: Vec<u32>,
    allocated: u32,
    per_page: u32,
    max_pages: u32,
}

impl<K: Clone + Eq + Hash> SlotMap<K> {
    /// `per_page` slots per page, at most `max_pages` pages (>= 1).
    pub fn new(per_page: u32, max_pages: u32) -> Self {
        Self {
            // The pool itself is the capacity limit; the LRU only orders.
            lru: Lru::new(usize::MAX),
            free: Vec::new(),
            allocated: 0,
            per_page: per_page.max(1),
            max_pages: max_pages.max(1),
        }
    }

    /// The slot of a resident `key`, marking it recently used.
    pub fn get(&mut self, key: &K) -> Option<u32> {
        self.lru.get(key).copied()
    }

    /// A slot for a new `key`: a free one, else a new page while under the cap,
    /// else the least recently used unpinned tile's. `None` when every slot is
    /// pinned (the caller draws a coarser tile instead).
    pub fn alloc(&mut self, key: K) -> Option<Alloc<K>> {
        let mut new_page = None;
        if self.free.is_empty() && self.allocated < self.per_page * self.max_pages {
            let page = self.allocated / self.per_page;
            self.free.extend((self.allocated..self.allocated + self.per_page).rev());
            self.allocated += self.per_page;
            new_page = Some(page);
        }
        let (slot, evicted) = match self.free.pop() {
            Some(s) => (s, None),
            None => {
                let (old, s) = self.lru.pop_lru()?;
                (s, Some(old))
            }
        };
        self.lru.insert(key, slot, 1);
        Some(Alloc { slot, new_page, evicted })
    }

    /// Pin `key` for this frame (a drawn tile must not be recycled by a later
    /// upload in the same frame).
    pub fn pin(&mut self, key: &K) {
        self.lru.set_pinned(key, true);
    }

    /// End of frame: nothing is pinned any more.
    pub fn unpin_all(&mut self) {
        self.lru.unpin_all();
    }

    /// Drop every key failing `keep`, returning their slots to the free list.
    pub fn retain(&mut self, keep: impl FnMut(&K) -> bool) {
        for (_, slot) in self.lru.retain(keep) {
            self.free.push(slot);
        }
    }

    /// Resident tiles.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.lru.len()
    }

    /// Pages allocated so far.
    #[cfg(test)]
    pub fn pages(&self) -> u32 {
        self.allocated / self.per_page
    }
}

/// Page geometry: `side x side` slots of `TILE` texels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageLayout {
    /// Page edge in texels (a multiple of `TILE`).
    pub size: u32,
    /// Slots per page edge.
    pub side: u32,
    /// Pages allowed by the budget.
    pub max_pages: u32,
}

impl PageLayout {
    /// Pages of at most 8192 texels (and at most the device limit) filling
    /// `budget` bytes; at least one page.
    pub fn new(max_texture_dim: u32, budget: u64) -> Self {
        let mut size = (max_texture_dim.min(8192) / TILE).max(1) * TILE;
        // A tiny image/budget must not allocate an 8192-square (512 MiB) page.
        // One 512-square tile (2 MiB) is the explicit minimum working set.
        while size > TILE && u64::from(size) * u64::from(size) * TEXEL_BYTES > budget {
            size -= TILE;
        }
        let side = size / TILE;
        let page_bytes = u64::from(size) * u64::from(size) * TEXEL_BYTES;
        let max_pages = (budget / page_bytes).clamp(1, u64::from(u32::MAX)) as u32;
        Self { size, side, max_pages }
    }

    /// Bytes of one page.
    pub fn page_bytes(&self) -> u64 {
        u64::from(self.size) * u64::from(self.size) * TEXEL_BYTES
    }

    /// Slots per page.
    pub fn per_page(&self) -> u32 {
        self.side * self.side
    }

    /// Where slot `s` lives.
    pub fn slot(&self, s: u32) -> Slot {
        let i = s % self.per_page();
        Slot { page: s / self.per_page(), x: (i % self.side) * TILE, y: (i / self.side) * TILE }
    }
}

/// One atlas page.
pub struct Page {
    /// The page texture (kept alive; writes go to it).
    pub tex: wgpu::Texture,
    /// Bind group of the page texture (group 1 of the tile pipeline).
    pub bind: wgpu::BindGroup,
}

/// The GPU atlas: slot bookkeeping plus page textures.
pub struct TileAtlas<K> {
    /// Page geometry.
    pub layout: PageLayout,
    slots: SlotMap<K>,
    pages: Vec<Page>,
    retired: super::retire::RetiredTextures,
}

impl<K: Clone + Eq + Hash> TileAtlas<K> {
    /// An empty atlas; pages are created on first use.
    pub fn new(layout: PageLayout) -> Self {
        Self {
            slots: SlotMap::new(layout.per_page(), layout.max_pages),
            layout,
            pages: Vec::new(),
            retired: Default::default(),
        }
    }

    /// Release page textures as well as slots after another presentation is ready.
    pub fn clear(&mut self, queue: &wgpu::Queue) {
        self.slots = SlotMap::new(self.layout.per_page(), self.layout.max_pages);
        let bytes = self.layout.page_bytes();
        for page in self.pages.drain(..) {
            self.retired.retire(queue, page.tex, bytes);
        }
    }

    pub fn reap(&mut self) {
        self.retired.reap();
    }

    /// Existing pages are retained for continuous presentation; future page
    /// growth is capped by the remainder after other live GPU allocations.
    pub fn limit_growth(&mut self, allowance: u64) {
        let pages = allowance.saturating_sub(self.retired.bytes()) / self.layout.page_bytes();
        self.slots.max_pages = pages.min(u64::from(self.layout.max_pages)) as u32;
    }

    /// The pages, for binding at draw time.
    pub fn pages(&self) -> &[Page] {
        &self.pages
    }

    /// VRAM held by the pages created so far, and the budget (the status bar).
    pub fn bytes(&self) -> (u64, u64) {
        let page = self.layout.page_bytes();
        (
            self.pages.len() as u64 * page + self.retired.bytes(),
            u64::from(self.layout.max_pages) * page,
        )
    }

    /// The slot of a resident `key` (marked recently used and pinned for the frame).
    pub fn resident(&mut self, key: &K) -> Option<Slot> {
        let s = self.slots.get(key)?;
        self.slots.pin(key);
        Some(self.layout.slot(s))
    }

    /// Upload `tile` for `key` and pin it; `None` when every slot is pinned.
    pub fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        page_bgl: &wgpu::BindGroupLayout,
        key: K,
        tile: &TileData,
    ) -> Option<Slot> {
        let a = self.slots.alloc(key.clone())?;
        if let Some(page) = a.new_page {
            debug_assert_eq!(page as usize, self.pages.len());
            self.pages.push(new_page(device, page_bgl, self.layout.size));
        }
        let slot = self.layout.slot(a.slot);
        let page = self.pages.get(slot.page as usize)?;
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &page.tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x: slot.x, y: slot.y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice::<u16, u8>(&tile.rgba),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(tile.w * TEXEL_BYTES as u32),
                rows_per_image: Some(tile.h),
            },
            wgpu::Extent3d { width: tile.w, height: tile.h, depth_or_array_layers: 1 },
        );
        self.slots.pin(&key);
        Some(slot)
    }

    /// End of frame.
    pub fn end_frame(&mut self) {
        self.slots.unpin_all();
    }

    /// Forget every tile whose key fails `keep` (a new frame or layer).
    pub fn retain(&mut self, keep: impl FnMut(&K) -> bool) {
        self.slots.retain(keep);
    }
}

fn new_page(device: &wgpu::Device, bgl: &wgpu::BindGroupLayout, size: u32) -> Page {
    let usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
    #[cfg(test)]
    let usage = usage | wgpu::TextureUsages::COPY_SRC;
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("exr_view.tile_page"),
        size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        // Always filterable (unlike Rgba32Float), half the VRAM; tiles are display data.
        format: wgpu::TextureFormat::Rgba16Float,
        usage,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("exr_view.tile_page.bg"),
        layout: bgl,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) }],
    });
    Page { tex, bind }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_fill_pages_then_grow_then_recycle() {
        let mut m = SlotMap::new(4, 2);
        let mut pages = Vec::new();
        for k in 0..8u32 {
            let a = m.alloc(k).unwrap();
            assert_eq!(a.evicted, None);
            pages.extend(a.new_page);
        }
        assert_eq!(pages, [0, 1], "a page per four slots, up to the cap");
        assert_eq!(m.pages(), 2);
        m.get(&0); // 0 becomes the newest; 1 is now the oldest
        let a = m.alloc(100).unwrap();
        assert_eq!((a.evicted, a.new_page), (Some(1), None));
        assert_eq!(m.len(), 8);
        assert!(m.get(&1).is_none());
    }

    #[test]
    fn pinned_tiles_are_not_recycled_and_full_pins_refuse() {
        let mut m = SlotMap::new(2, 1);
        m.alloc('a').unwrap();
        m.alloc('b').unwrap();
        m.pin(&'a');
        assert_eq!(m.alloc('c').unwrap().evicted, Some('b'));
        m.pin(&'c');
        assert!(m.alloc('d').is_none(), "every slot is on screen");
        m.unpin_all();
        assert_eq!(m.alloc('d').unwrap().evicted, Some('a'));
    }

    #[test]
    fn page_capacity_obeys_budget_except_explicit_minimum() {
        for limit in [512, 1024, 4096, 8192, 16384] {
            for budget in [0, 1, (2 << 20) - 1, 2 << 20, 3 << 20, 64 << 20, 500 << 20] {
                let layout = PageLayout::new(limit, budget);
                assert!(layout.size <= limit && layout.size <= 8192);
                let capacity = layout.page_bytes() * u64::from(layout.max_pages);
                if budget < 2 << 20 {
                    assert_eq!(capacity, 2 << 20);
                } else {
                    assert!(capacity <= budget, "capacity{capacity} budget{budget}");
                }
            }
        }
    }
    #[test]
    fn a_reduced_allowance_prevents_new_pages() {
        let mut atlas = TileAtlas::<u32>::new(PageLayout {
            size: 512,
            side: 1,
            max_pages: 4,
        });
        atlas.limit_growth(2 << 20);
        assert_eq!(atlas.slots.alloc(1).unwrap().new_page, Some(0));
        assert_eq!(
            atlas.slots.alloc(2).unwrap().new_page,
            None,
            "existing slot recycled"
        );
        atlas.limit_growth(0);
        assert_eq!(
            atlas.slots.alloc(3).unwrap().new_page,
            None,
            "existing allocation retained without growth"
        );
    }
    #[test]
    fn retain_frees_slots_for_reuse() {
        let mut m = SlotMap::new(2, 1);
        let a = m.alloc(1).unwrap().slot;
        m.alloc(2).unwrap();
        m.retain(|k| *k != 1);
        let b = m.alloc(3).unwrap();
        assert_eq!((b.slot, b.evicted), (a, None), "the freed slot, nothing evicted");
    }

    #[test]
    fn page_layout_follows_device_limit_and_budget() {
        let l = PageLayout::new(16384, 2 << 30);
        // An 8192^2 Rgba16Float page is 512 MiB, so 2 GiB of budget is 4 pages.
        assert_eq!((l.page_bytes(), l.max_pages), (512 << 20, 4));
        assert_eq!((l.size, l.side), (8192, 16));
        assert_eq!(l.max_pages, 4, "2 GiB / 512 MiB pages");
        let l = PageLayout::new(4096, 0);
        assert_eq!(
            (l.size, l.side, l.max_pages),
            (512, 1, 1),
            "explicit minimum page even below its 2 MiB allocation"
        );
        let l = PageLayout::new(1000, 1 << 30);
        assert_eq!(l.size, 512, "rounded down to whole tiles");
        let l = PageLayout::new(8192, u64::MAX);
        assert_eq!(l.slot(0), Slot { page: 0, x: 0, y: 0 });
        assert_eq!(l.slot(17), Slot { page: 0, x: TILE, y: TILE });
        assert_eq!(l.slot(256), Slot { page: 1, x: 0, y: 0 });
    }
}
