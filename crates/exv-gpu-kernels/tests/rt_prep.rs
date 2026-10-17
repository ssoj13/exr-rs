//! `rt_prep` (EXV-RT descriptor expansion on the GPU) against a host reference, including FORGED tile records: widths
//! outside 1..=4, a block count past the tile's width span and a row range past the output arrays must give inert
//! rows / dropped rows, deterministically, never an out-of-bounds access. Needs a real adapter:
//! `cargo test -p exv-gpu-kernels --test rt_prep -- --include-ignored`.

use exv_gpu_kernels::{DESC_WORDS, GpuContext, RT_DESC_WORDS, TILE_DESC_WORDS, map_read_blocking, prep_rows, storage_u32};
use gpu_info::wgpu;

/// Copy `buf` to a MAP_READ staging buffer and return its words.
fn words(ctx: &GpuContext, buf: &wgpu::Buffer) -> Vec<u32> {
    let st = exv_gpu_kernels::staging(&ctx.device, "rd", buf.size());
    let mut e = ctx.device.create_command_encoder(&Default::default());
    e.copy_buffer_to_buffer(buf, 0, &st, 0, buf.size());
    ctx.queue.submit([e.finish()]);
    let b = map_read_blocking(&ctx.device, &st).expect("map");
    b.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect()
}

/// Field order == WGSL `TileDesc`.
fn rec(unit0: u32, nblk: u32, units: u32, w_base: u32, w_len: u32, sym_base: u32, packed: u32) -> [u32; TILE_DESC_WORDS] {
    // lanes, table_log, freq, cum, slot, lane_base | block, nbx, x0, y0, full_w, x1, y1, q, out_base, packed
    [unit0, nblk, units, 2, 12, 100, 200, 300, sym_base, 7, w_base, w_len, 8, 2, 16, 24, 64, 40, 50, 3, 9, packed]
}

#[test]
#[ignore = "needs a GPU"]
fn prep_matches_host_and_survives_forged_records() {
    let Ok(ctx) = GpuContext::new() else { return };
    // widths: tile A = [1,2,3,4]; tile B = [2,9,0] (forged: 9 and 0 are illegal).
    let widths: Vec<u8> = vec![1, 2, 3, 4, 2, 9, 0, 0];
    let packed_w: Vec<u32> = widths.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let b_w = storage_u32(&ctx.device, &ctx.queue, "w", &packed_w, wgpu::BufferUsages::empty());
    // Both symbol layouts: one word per symbol, and byte-packed (4 per word: a block takes `ceil(n / 4)` words).
    for packed in [0u32, 1] {
        let words_of = |n: u32| if packed == 1 { n.div_ceil(4) } else { n };
        // A: 4 blocks + 2 padding rows (rows 0..6). B: claims 1000 blocks over a 3-byte span (rows 6..14, 8 rows).
        // C: row range starts past the arrays (rows 14..): must be dropped, not written out of bounds.
        let recs = [rec(0, 4, 6, 0, 4, 0, packed), rec(6, 1000, 8, 4, 3, 1000, packed), rec(500, 4, 4, 0, 4, 0, packed)];
        let units = 14usize;
        let (rd, rt) = prep_rows(&ctx, &recs, &b_w, units).expect("prep");
        let (rd, rt) = (words(&ctx, &rd), words(&ctx, &rt));
        assert_eq!(rd.len(), units * DESC_WORDS);
        assert_eq!(rt.len(), units * RT_DESC_WORDS);
        let (ncoeff, mut sym) = (64u32, 0u32);
        for i in 0..6usize {
            let r = &rd[i * DESC_WORDS..(i + 1) * DESC_WORDS];
            let (wb, real) = (widths.get(i).copied().unwrap_or(0) as u32, i < 4);
            let n = if real { ncoeff * wb } else { 0 };
            assert_eq!(&r[..8], &[2, 12, n, 7 + if real { i as u32 * 2 } else { 0 }, 100, 200, 300, sym], "A rans row {i} packed {packed}");
            let t = &rt[i * RT_DESC_WORDS..(i + 1) * RT_DESC_WORDS];
            if real {
                assert_eq!(t, &[wb, sym, 8, 16 + (i as u32 % 2) * 8, 24 + (i as u32 / 2) * 8, 64, 40, 50, 3, 9, packed], "A inv row {i} packed {packed}");
            } else {
                assert!(t.iter().all(|&x| x == 0), "A padding inv row {i}");
            }
            sym += words_of(n);
        }
        // B: only block 0 (width 2) is live; the forged 9 / 0 / past-the-span blocks are inert and keep `sym`.
        let sym_b = 1000;
        assert_eq!(rd[6 * DESC_WORDS + 2], ncoeff * 2);
        assert_eq!(rd[6 * DESC_WORDS + 7], sym_b);
        for i in 7..14usize {
            assert_eq!(rd[i * DESC_WORDS + 2], 0, "B row {i} must be inert");
            assert_eq!(rd[i * DESC_WORDS + 7], sym_b + words_of(ncoeff * 2));
            assert!(rt[i * RT_DESC_WORDS..(i + 1) * RT_DESC_WORDS].iter().all(|&x| x == 0), "B inv row {i}");
        }
    }
}
