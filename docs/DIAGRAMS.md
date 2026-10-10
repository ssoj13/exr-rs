# exr-rs diagrams

Mermaid diagrams of the current architecture. Every node that names a module, type, or
function exists in the code.

## Crate dependency graph

Solid boxes are workspace crates; rounded boxes are git dependencies (own repos).

```mermaid
flowchart TD
    CLI[exr-rs CLI] --> core[exr-core]
    CLI --> crypto[cryptomatte]
    CLI --> imath[imath-rs]
    CLI --> util[exr-util]
    CLI -->|feature viewer| view[exr-view]
    util --> core
    py[openexr-py<br/>maturin only] --> core
    py --> imath
    py --> crypto
    crypto --> core
    crypto --> imath
    crypto --> murmur3([murmur3])
    view --> core
    view --> imath
    view --> crypto
    view --> widgets([egui-widgets-rs])
    view --> scanseq([scanseq])
    view --> scancache([scancache])
    view --> winfo([wgpu-info / gpu-info-rs])
    view --> ocio([vfx-ocio / oiio-rs])
    view -->|feature ofx| ofx([ofx-rs])
    view -->|UploadLayout sizing only| gpu
    gpu[exr-gpu] --> core
    core --> imath
    core --> deflate[deflate-rs]
    core --> jph([jph-rs])
    core --> simd([codec-simd])
    core --> murmur3
    xtask[xtask] --> vcv([vcv-rs])
```

`exr-gpu` is a library for downstream consumers; the viewer uses only its checked
`UploadLayout` (texture sizing) and keeps its own upload path. `jph-rs` ports OpenJPH 0.31.0
(the version the C++ oracle links); `deflate-rs` ports libdeflate 1.26.

## Read path (facade)

```mermaid
flowchart TD
    R["exr_core::read / Image::read_from<br/>(Image::read_all keeps levels + sampling)"] --> K["peek_kind<br/>magic + version flags"]
    K -->|multipart / deep| U["Err(ExrError::Unsupported(kind))"]
    K --> H["InputFile / TiledInputFile::new<br/>header read + Header::sanity_check_at_open"]
    H --> P["ReadPlan::new<br/>sample_dims (checked);<br/>subsampled -> Unsupported for read"]
    P --> FB["ReadPlan::frame_buffer<br/>one Slice per channel (native-endian)"]
    FB --> S{"scanline or tiled?"}
    S -->|scanline| SI["scanline::InputFile::read_pixels"]
    S -->|tiled| TI["tiled::TiledInputFile::read_tiles<br/>(level 0, or every level)"]
    SI --> D["Compressor::uncompress<br/>(raw when dataSize == unpacked)"]
    TI --> DT["Compressor::uncompress_tile(range)"]
    D --> C["misc::copy_into_frame_buffer<br/>XDR -> native"]
    DT --> C
    C --> I["ReadPlan::into_planes -> Image"]
```

## Write path (facade)

```mermaid
flowchart TD
    W["Image::write / write_with_attrs / write_rgba_file"] --> BH["build_header"]
    W --> BB["Planes::write_buf<br/>native-endian planes + FrameBuffer"]
    BH --> O["scanline::OutputFile::new<br/>magic + version + header + zeroed offset table"]
    BH -.->|tiling set| TO["TiledOutputFile: every level<br/>(same seam, write_tiles)"]
    BB --> SF["OutputFile::set_frame_buffer(&fb)<br/>stores the LAYOUT only"]
    O --> SF
    SF --> WP["OutputFile::write_pixels(pixels, n)<br/>bytes passed per call, bounds-checked"]
    BB -->|pixel bytes| WP
    WP --> G["misc::copy_from_frame_buffer<br/>native -> XDR"]
    G --> K["Compressor::compress"]
    K --> SC["compressor::stored_chunk<br/>raw unless strictly smaller"]
    SC --> CH["write chunk"]
    CH --> BP["back-patch offset table entry"]
```

## Codec dispatch

```mermaid
flowchart LR
    C["Compression<br/>(Compression::ALL)"] --> F["compressor::new_compressor<br/>compressor::new_tile_compressor"]
    F --> N["NONE: None"]
    F --> RLE["rle::RleCompressor"]
    F --> ZIP["zip::ZipCompressor<br/>ZIPS 1 line / ZIP 16 lines"]
    F --> PIZ["piz::PizCompressor"]
    F --> PXR["pxr24::Pxr24Compressor"]
    F --> B44["b44::B44Compressor<br/>B44 / B44A"]
    F --> DWA["dwa::DwaCompressor<br/>DWAA / DWAB"]
    F --> HT["htj2k::Htj2kCompressor<br/>HTJ2K256 / HTJ2K32"]
    RLE --> PR["predictor.rs"]
    ZIP --> PR
    ZIP --> DF["deflate-rs<br/>level = Header::zip_level (default 4)"]
    PXR --> DF
    HT --> JPH["jph-rs codestream<br/>(OpenJPH 0.31.0, Result API)"]
    DWA --> DF
```

## Multipart part caching

```mermaid
flowchart TD
    MO["MultiPartOutputFile<br/>owns stream + per-part Option&lt;*State&gt;"] --> OP["output_part(n) / tiled_output_part(n) /<br/>deep_*_output_part(n)"]
    OP -->|first access| NS["*OutputFileState::from_part<br/>part_number = (parts &gt; 1).then_some(n)"]
    OP -->|later access| RS["reuse persisted state<br/>(tile_map, current line, offsets)"]
    NS --> VW["view { &mut stream, &mut state }<br/>*OutputPartWriter"]
    RS --> VW
    MI["MultiPartInputFile<br/>owns stream + per-part reader state"] --> IP["input_part(n) / tiled_input_part(n) /<br/>deep_*_input_part(n)"]
    IP --> RV["*InputPartReader view<br/>(frame buffer persists across calls)"]
```

## Deep sample addressing

```mermaid
flowchart TD
    CS["sample-count Slice<br/>(base_offset, x_stride, y_stride)"] --> CBO["sample_table::count_byte_offset<br/>base_offset + y*yStride + x*xStride"]
    CBO --> SCA["sample_table::sample_count_at<br/>Result&lt;u32&gt; (out of buffer = hard error)"]
    SCA --> SBP["sample_table::samples_before_pixel<br/>(whole level-0 window raster)"]
    SBP --> ADDR["heap offset = DeepSlice base_offset<br/>+ (samples_before + i) * sample_stride"]
    subgraph Chunk codec
        ENC["encode_cumulative_row<br/>checked u64, &lt;= i32::MAX"]
        DEC["decode_cumulative_row<br/>non-negative + monotonic"]
        CHK["check_sample_data_sizes<br/>before any allocation"]
    end
    DEC --> SCA
    ENC --> CBO
    CHK --> ADDR
```

## Viewer / cache pipeline

```mermaid
flowchart TD
    OP["open_path / sequence detection"] --> SC["scancache: decode, playhead-ranked prefetch"]
    SC --> IB["ImageBackend / current source"]
    IB --> EL{"resident eligibility"}
    EL -->|eligible ≤8192, within budgets| RS["reused RGBA32Float source"]
    EL -->|fallback| TS["512-pixel pyramid / file bands"]
    RS --> SH["shared GPU display / OCIO transform"]
    TS --> AT["f16 tile atlas + draw plan"]
    AT --> SH
    SH --> CA["extended-sRGB f16 canvas + egui"]
    CA --> PP["present: SDR / PQ / HLG / scRGB"]
    TS --> RET["keep the current GPU image until its replacement can draw"]
    RS --> RET
    IB --> SP["ensure_linear / GPU scopes"]
```

The displayed result (with OpenFX effects applied) is one immutable snapshot that the
scopes, viewport readout, Metadata, Pixel Info and Save view all read.

## Parity harness flow

```mermaid
flowchart TD
    X["cargo run -p xtask -- parity-exr / parity / parity-deflate"] --> V["vcv-rs: MSVC env"]
    X --> VP["vcpkg::Vcpkg::resolve(probe header)<br/>VCPKG_INSTALLED or first x64-windows* triplet"]
    VP --> L["static or DLL: CRT /MT or /MD,<br/>lib names, run_path"]
    V --> CL["cl oracle.cpp / oracle.c"]
    L --> CL
    CL --> OR["oracle.exe"]
    OR --> GD["regenerate goldens<br/>golden*.exr / golden.txt"]
    OR --> T["cargo test (oracle found at tests/parity/oracle.exe<br/>or EXR_ORACLE_EXE; PATH = run_path)"]
    GD --> A["(a) C++ goldens -> our readers"]
    T --> A
    T --> B["(b) our writers -> oracle read"]
```


## deflate-rs (libdeflate v1.26 port): compress / decompress

```mermaid
flowchart LR
    subgraph exrcore["exr-core codecs"]
        ZIP["zip.rs"]:::codec
        PXR["pxr24.rs"]:::codec
        DWA["dwa/mod.rs"]:::codec
    end
    ZIP & PXR & DWA --> API["lib.rs facade<br/>Compressor / Decompressor"]
    API --> ZC["zlib_compress / zlib_decompress_ex"]
    ZC --> DC["deflate_compress"]
    ZC --> DD["deflate_decompress_ex"]
    ZC --> AD["adler32<br/>scalar | sse2 | avx2"]
    DC --> MF{"level"}
    MF -->|1| HT["ht_matchfinder"]
    MF -->|2-9| HC["hc_matchfinder"]
    MF -->|10-12| BT["bt_matchfinder + near-optimal"]
    DC --> HUF["Huffman build"]
    DC --> BS["bitstream flush_bits!"]
    DD --> TB["decode tables"]
    DD --> FST["fastloop"]
    FST --> GL["generic loop"]
    CPU["cpu.rs runtime dispatch"] -.-> AD
    CPU -.->|"BMI2 instantiation"| FST
    classDef codec fill:#eef
```
