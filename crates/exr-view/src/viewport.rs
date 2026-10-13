//! Flat 2-D image viewport: pan / zoom / fit / home + pixel readout, drawing
//! the GPU display texture from [`crate::image::ImageBackend`].
//!
//! The viewer owns its wgpu device/queue/renderer directly; controls are plain
//! `exposure`/`gamma`/`ChannelDisplay` on [`crate::app::ViewControls`] (no OCIO).
//! Drawing: `present()` → `painter.image(tex_id, …)`.
//!
//! Navigation:
//!   MMB drag       — pan
//!   RMB drag        — zoom about the viewport centre (drag up = in, down = out)
//!   Scroll          — zoom toward cursor
//!   LMB drag        — scrub the sequence (on-screen animation scrubber, playa-style);
//!                     on a single image LMB click pins the pixel readout instead
//!
//! Keys (F, H/0, Ctrl +/-, R/G/B/A, ...) are `commands::COMMANDS`; the navigation they
//! ask for arrives here as `ViewControls::nav`.

use crate::app::ViewControls;
use crate::commands::NavCmd;
use crate::theme;
use crate::image::{DisplayTarget, ImageBackend, ImageStage, Windows};
use crate::tiles::render::Look;
use crate::tiles::view::Placement;

const ZOOM_SPEED: f32 = 0.002;
/// Per-pixel zoom rate for right-button drag (vertical).
const ZOOM_DRAG_SPEED: f32 = 0.005;
/// Multiplicative zoom step for one Ctrl + `+`/`-` key press.
const ZOOM_KEY_STEP: f32 = 1.25;
const ZOOM_MIN: f32 = 0.01;
const ZOOM_MAX: f32 = 64.0;

/// The viewport's own background grey, also painted behind the toolbar strip's band.
pub const VIEW_BG: egui::Color32 = egui::Color32::from_gray(28);

/// The toolbar strip as the viewport sees it THIS frame (a per-frame value, never stored: a
/// frame that draws no strip passes [`Strip::NONE`]).
#[derive(Clone, Copy, Debug)]
pub struct Strip {
    /// The strip's actual (animated) rect; `Rect::NOTHING` when none is drawn.
    pub rect: egui::Rect,
    /// The pointer is over it: the wheel and the raw right-drag are not routed by hit-testing,
    /// so the viewport skips them then.
    pub hot: bool,
}

impl Strip {
    /// No strip this frame.
    pub const NONE: Self = Self { rect: egui::Rect::NOTHING, hot: false };
    /// How far from the viewport's top (bottom) edge the top-left (bottom-right) HUD block can reach.
    const HUD_ZONE: f32 = 80.0;
    /// How far from the right edge the bottom-right HUD block (the zoom) reaches.
    const HUD_WIDTH: f32 = 90.0;

    /// The height the viewport leaves free ABOVE itself for a strip docked at the top edge of
    /// `view` (its top within half a point of `view.top() + margin`: the widget's own anchor),
    /// in step with the roll: full when rolled out, none for the bare toggle. A strip floating
    /// elsewhere, or none, takes nothing: it covers the image and the HUD steps aside instead.
    pub fn band(self, view: egui::Rect, margin: f32, button_width: f32) -> f32 {
        if !self.rect.is_positive() || self.rect.top() > view.top() + margin + 0.5 {
            return 0.0;
        }
        let rolled_out = view.width() - 2.0 * margin;
        let rolled = if rolled_out > button_width {
            ((self.rect.width() - button_width) / (rolled_out - button_width)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        (self.rect.bottom() - view.top() + 2.0) * rolled
    }

    /// The rect the HUD blocks are anchored in: the viewport's, pulled away from the strip on
    /// whichever edge it sits. Below the strip when it is in the top-left block's zone
    /// (collapsed toggle, a strip dragged near the top); above it when it reaches the
    /// bottom-right block (a strip dragged to the bottom, rolled out to the right).
    fn hud_rect(self, rect: egui::Rect) -> egui::Rect {
        if !self.rect.is_positive() {
            return rect;
        }
        let mut hud = rect;
        if self.rect.top() < rect.top() + Self::HUD_ZONE {
            hud.min.y = hud.min.y.max(self.rect.bottom() + 2.0);
        }
        if self.rect.bottom() > rect.bottom() - Self::HUD_ZONE && self.rect.right() > rect.right() - Self::HUD_WIDTH {
            hud.max.y = hud.max.y.min(self.rect.top() - 2.0);
        }
        hud
    }
}

/// GPU handles the viewport needs to stream tiles into the atlas.
pub struct ViewportGpu<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub renderer: &'a mut egui_wgpu::Renderer,
}

/// What a dock tab needs from the GPU: the handles, or nothing. Only tabs that draw pixels
/// (the viewport, the scopes) ask; key-only tabs never touch it, so the real `build_ui` runs
/// in a headless test with [`NoGpu`] and its panel registrations are covered without an adapter.
pub trait TabGpu {
    /// The GPU handles. Production ([`ViewportGpu`]) always has them; a pixel tab given `None`
    /// says so on screen instead of drawing.
    fn gpu(&mut self) -> Option<ViewportGpu<'_>>;
}

impl TabGpu for ViewportGpu<'_> {
    fn gpu(&mut self) -> Option<ViewportGpu<'_>> {
        Some(ViewportGpu { device: self.device, queue: self.queue, renderer: &mut *self.renderer })
    }
}

/// The headless test stand-in: no adapter, so pixel tabs show their "no GPU" message.
#[cfg(test)]
pub struct NoGpu;

#[cfg(test)]
impl TabGpu for NoGpu {
    fn gpu(&mut self) -> Option<ViewportGpu<'_>> {
        None
    }
}

/// Draw the flat image viewport into `ui`.
///
/// Handles navigation + channel keys, runs the GPU colour pass via
/// [`ImageBackend::present`], and paints the result. Updates `controls`
/// (channel isolate) and `img` (pan/zoom/picked pixel) in place.
#[allow(clippy::too_many_arguments)]
pub fn show(
    ui: &mut egui::Ui,
    img: &mut ImageBackend,
    controls: &mut ViewControls,
    gpu: &mut dyn TabGpu,
    target: &mut DisplayTarget,
    seq_pos: Option<f32>,
    playhead_alpha: f32,
    seq_range: Option<(i32, i32)>,
    mut stage: Option<&mut (dyn ImageStage + '_)>,
    strip: Strip,
) -> ViewportOut {
    let mut out = ViewportOut::default();
    // A stage with an overlay takes the left button (its handles); pick and scrub are off then.
    let lmb_free = !stage.as_ref().is_some_and(|stage| stage.takes_pointer());

    let available = ui.available_size();
    if available.x <= 0.0 || available.y <= 0.0 {
        return out;
    }

    let (rect, response) = ui.allocate_exact_size(available, egui::Sense::click_and_drag());

    // The canvas we fit and anchor on is the DISPLAY window, not the data window:
    // a sequence whose data window changes per frame then stays put instead of
    // jumping as the image re-centres.
    let win = img.windows();
    #[cfg(feature = "ofx")]
    let worker = stage.as_ref().is_some_and(|stage| stage.uses_worker());
    #[cfg(not(feature = "ofx"))]
    let worker = false;
    let aspect = if worker { img.display_aspect() } else { 1.0 };
    // Recompute the fit factor EVERY frame from the live rect — it is the target
    // of `F`/`reset_view`, and a stale one (captured at first display) made `F`
    // wrong after every window resize. Only the FIRST computation also sets the
    // current zoom; after that the user owns it.
    if img.width > 0 && img.height > 0 {
        let fw = rect.width() / (win.display_size.0.max(1) as f32 * aspect);
        let fh = rect.height() / win.display_size.1.max(1) as f32;
        let first = img.fit_zoom <= 0.0;
        img.fit_zoom = fw.min(fh);
        if first {
            img.zoom = img.fit_zoom;
        }
    }

    // ── Navigation ──────────────────────────────────────────────────────────
    // Middle-button drag pans.
    if response.dragged_by(egui::PointerButton::Middle) {
        img.pan += response.drag_delta();
    }
    // Right-button drag zooms about the VIEWPORT CENTRE (drag up = in, down =
    // out). We zoom about the centre rather than the cursor on purpose: with a
    // cursor pivot the zoom point slid around as the mouse moved during the
    // drag, which felt unstable — so the pivot is the fixed centre.
    // (The wheel still zooms toward the cursor, where a moving pivot is natural.)
    //
    // Driven from the RAW pointer state (`secondary` button down + per-frame
    // `pointer.delta()`) rather than `dragged_by(Secondary)` alone: egui
    // arbitrates the secondary button between a click, a context-menu trigger
    // and a drag (`is_decidedly_dragging` gates on `could_any_button_be_click`),
    // so the claimed-drag signal is suppressed for small/slow right-drags — which
    // is exactly why the RMB zoom felt dead. We still OR-in `dragged_by` so a
    // fast drag that strays off the viewport rect keeps zooming.
    let (rmb_down, raw_dy) = ui.input(|i| {
        (
            i.pointer.button_down(egui::PointerButton::Secondary),
            i.pointer.delta().y,
        )
    });
    // Raw input is not hit-tested, so the toolbar strip's hover is gated here.
    if rmb_down
        && raw_dy != 0.0
        && !strip.hot
        && (response.contains_pointer() || response.dragged_by(egui::PointerButton::Secondary))
    {
        zoom_toward_center(img, 1.0 - raw_dy * ZOOM_DRAG_SPEED);
    }
    // Wheel zooms toward the cursor.
    if response.hovered() && !strip.hot {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll != 0.0 {
            zoom_toward_cursor(img, rect, ui, 1.0 + scroll * ZOOM_SPEED);
        }
    }
    // The sequence scrub (LMB) is handled below, mapped across the IMAGE bounds.

    // ── Navigation asked this frame (menu, key or toolbar; `commands`) ─────────
    match controls.nav.take() {
        Some(NavCmd::Fit) => img.reset_view(),
        Some(NavCmd::Home) => img.home_view(),
        Some(NavCmd::ZoomIn) => zoom_toward_center(img, ZOOM_KEY_STEP),
        Some(NavCmd::ZoomOut) => zoom_toward_center(img, 1.0 / ZOOM_KEY_STEP),
        None => {}
    }
    img.channel_display = controls.channel;
    // Apply the shared layer selection each frame (per-frame backends in a
    // sequence are rebuilt fresh; set_view only re-materializes on change).
    img.set_view(&controls.view);

    // ── Background ──────────────────────────────────────────────────────────
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, VIEW_BG);

    if img.width == 0 || img.height == 0 {
        // No image yet: a sequence scrub still works, mapped across the viewport
        // (there are no image bounds to map to until a frame lands).
        if lmb_free && seq_pos.is_some() && is_primary_scrub(&response)
            && let Some(p) = response.interact_pointer_pos()
        {
            out.scrub_to = Some(((p.x - rect.left()) / rect.width().max(1.0)).clamp(0.0, 1.0));
        }
        draw_hud(ui, strip.hud_rect(rect), None, img.zoom);
        return out;
    }

    // The display-window rect (the stable canvas) and, inside it, the data window
    // at its own offset. For an ordinary image the two coincide.
    let canvas = egui::Rect::from_center_size(
        rect.center() + img.pan,
        egui::vec2(
            win.display_size.0 as f32 * aspect * img.zoom,
            win.display_size.1 as f32 * img.zoom,
        ),
    );
    let img_rect = egui::Rect::from_min_size(
        canvas.min
            + egui::vec2(
                (win.data_min.0 - win.display_min.0) as f32 * img.zoom,
                (win.data_min.1 - win.display_min.1) as f32 * img.zoom,
            ),
        egui::vec2(img.width as f32 * img.zoom, img.height as f32 * img.zoom),
    );

    // The backdrop shows through the image's alpha: this clears the image's rect of the canvas
    // and the present pass composites it there, in linear light (a painted checker came out at
    // the wrong brightness under PQ/HLG). The tiles are drawn after, premultiplied, over it.
    let shown = if worker { canvas } else { img_rect };
    let view = egui_display::BackdropView::new(rect.min, ui.ctx().pixels_per_point()).with_image(shown.min, img.zoom);
    egui_display::show_backdrop(&painter, shown, &controls.backdrop, &view);

    // Plan and stream this frame's tiles, then draw them through a paint
    // callback clipped to the viewport. Magnified pixels are drawn nearest
    // (pixel-exact); minified ones filtered. An error draws its reason instead
    // of an image — the viewer stays alive and says why.
    let place = Placement { origin: if worker { canvas.min } else { img_rect.min }, zoom: img.zoom, clip: rect };
    // The frame's colour space resolves here, once per frame and choice (`ocio::State::frame`):
    // its OCIO transform, or the built-in decode of its tag.
    let src = controls.colour.frame(&img.frame, &controls.target);
    let colour = controls.colour.active().map(|t| &**t);
    let shown = match (gpu.gpu(), src) {
        (_, Err(e)) => Err(format!("{e} (OCIO off: no built-in decode for the frame's colour space)")),
        (Some(g), Ok(src)) => {
            let look = Look {
                exposure_ev: controls.exposure,
                gamma: controls.gamma,
                channel_mode: img.channel_display.to_channel_mode(),
                tonemap: controls.tonemap,
                nearest: img.zoom >= 1.0,
                hdr: controls.target.hdr,
                src,
            };
            img.present(g.device, g.queue, g.renderer, target, stage.as_deref_mut(), place, look, colour)
        }
        (None, Ok(_)) => Err("no GPU adapter".to_owned()),
    };
    match shown {
        Ok(tiles) => {
            painter.add(egui_wgpu::Callback::new_paint_callback(rect, tiles));
            // Tiles still being produced land on a later frame: keep painting.
            if target.tiles_pending() > 0 {
                ui.ctx().request_repaint();
            }
            // Overlay geometry and time follow the owner actually accepted by the GPU.
            #[cfg(feature = "ofx")]
            let mut overlay_rect = img_rect;
            #[cfg(not(feature = "ofx"))]
            let overlay_rect = img_rect;
            #[cfg(feature = "ofx")]
            let mut overlay_size = (img.width, img.height);
            #[cfg(not(feature = "ofx"))]
            let overlay_size = (img.width, img.height);
            #[cfg(feature = "ofx")]
            if worker {
                let shown = target.accepted_effect();
                if let Some(stage) = stage.as_deref_mut() { stage.accepted(shown); }
                if let Some(shown) = shown {
                    overlay_size = shown.size;
                    overlay_rect = egui::Rect::from_min_size(shown.place.origin, egui::vec2(
                        shown.size.0 as f32 * shown.aspect as f32 * shown.place.zoom,
                        shown.size.1 as f32 * shown.place.zoom));
                }
            }
            // The stage's overlay over the image; a change re-processes next frame.
            if let Some(stage) = stage
                && stage.overlay(&painter, &response, overlay_rect, overlay_size)
            {
                ui.ctx().request_repaint();
            }
        }
        Err(reason) => {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                format!("cannot display {}x{}: {reason}", img.width, img.height),
                theme::mono(ui.style()),
                egui::Color32::from_rgb(230, 140, 140),
            );
        }
    }

    // Sequence scrub (LMB): map the pointer X across the CANVAS bounds — not the
    // whole viewport — to a 0..=1 position the caller turns into a frame.
    if lmb_free && seq_pos.is_some() && is_primary_scrub(&response)
        && let Some(p) = response.interact_pointer_pos()
    {
        out.scrub_to = Some(((p.x - canvas.left()) / canvas.width().max(1.0)).clamp(0.0, 1.0));
    }

    // Acquire the accepted owner after presentation, so this readout cannot
    // prevent the promotion being attempted above.
    #[cfg(feature = "ofx")]
    let frozen = target.accepted_pixels();
    let (probe_origin, probe_scale, probe_size) =
        (img_rect.min, egui::vec2(img.zoom, img.zoom), (img.width, img.height));
    #[cfg(feature = "ofx")]
    let (probe_origin, probe_scale, probe_size) = target.accepted_effect()
        .map(|shown| (shown.place.origin,
            egui::vec2(shown.place.zoom * shown.aspect as f32, shown.place.zoom), shown.size))
        .unwrap_or((probe_origin, probe_scale, probe_size));
    let probe_rect = egui::Rect::from_min_size(probe_origin,
        egui::vec2(probe_size.0 as f32 * probe_scale.x, probe_size.1 as f32 * probe_scale.y));
    let probe_at = |pos: egui::Pos2| {
        if probe_scale.x == probe_scale.y {
            return screen_to_pixel(pos, probe_rect, probe_scale.y, probe_size.0, probe_size.1);
        }
        if probe_scale.x <= 0.0 || probe_scale.y <= 0.0 { return None; }
        let x = ((pos.x - probe_origin.x) / probe_scale.x).floor() as f64;
        let y = ((pos.y - probe_origin.y) / probe_scale.y).floor() as f64;
        (x.is_finite() && y.is_finite() && x >= 0.0 && y >= 0.0
            && x < f64::from(probe_size.0) && y < f64::from(probe_size.1))
            .then_some((x as u32, y as u32))
    };

    // LMB click → pin pixel readout — ONLY on a single image (for a sequence the
    // left button is the scrub, handled above).
    if lmb_free && seq_pos.is_none() && response.clicked_by(egui::PointerButton::Primary)
        && let Some(pos) = response.interact_pointer_pos()
    {
        img.picked_pixel = probe_at(pos);
    }

    // Hover readout HUD (hover takes priority, pinned pick shown otherwise).
    // Hovering OUTSIDE the data window is reported as such: the old code clamped
    // negative coordinates with `.max(0)`, so pointing left of / above the image
    // read out pixel (0,0) as if it were under the cursor.
    let hover = response.hover_pos();
    let readout: Option<(u32, u32, bool)> = match hover {
        Some(pos) => probe_at(pos).map(|(x, y)| (x, y, false)),
        None => img.picked_pixel.filter(|&(x, y)| x < probe_size.0 && y < probe_size.1)
            .map(|(x, y)| (x, y, true)),
    };

    let mut readout_line = None;
    if let Some((px, py, pinned)) = readout {
        #[cfg(not(feature = "ofx"))]
        let pixel = img.pixel_at(px, py);
        #[cfg(feature = "ofx")]
        let pixel = match &frozen {
            Some(pixels) => pixels.canvas_sample(px, py).map(|sample| sample.pixel),
            None => img.pixel_at(px, py),
        };
        if let Some([r, g, b, a]) = pixel {
            let coordinates = coords_label(&win, px, py);
            #[cfg(feature = "ofx")]
            let coordinates = frozen.as_ref().map_or(coordinates, |pixels| {
                let lineage = pixels.lineage();
                format!("canvas({px},{py}) frame:{} t:{}", lineage.primary_index, lineage.time)
            });
            let tag = if pinned { "[pin] " } else { "" };
            readout_line = Some(format!(
                "{tag}{}  R:{r:.4}  G:{g:.4}  B:{b:.4}  A:{a:.4}",
                coordinates
            ));
        }
        // Live readout also feeds the Pixel Info panel.
        controls.hovered_pixel = Some((px, py));
    } else {
        if hover.is_some() {
            readout_line = Some("outside the data window".to_owned());
        }
        controls.hovered_pixel = None;
    }

    // Crosshair at the pinned pixel.
    if let Some((px, py)) = img.picked_pixel {
        let center =
            probe_origin + egui::vec2((px as f32 + 0.5) * probe_scale.x, (py as f32 + 0.5) * probe_scale.y);
        let arm = (img.zoom * 0.5).clamp(3.0, 8.0);
        let col = egui::Color32::from_rgba_unmultiplied(255, 220, 60, 200);
        painter.line_segment(
            [center - egui::vec2(arm, 0.0), center + egui::vec2(arm, 0.0)],
            egui::Stroke::new(1.0_f32, col),
        );
        painter.line_segment(
            [center - egui::vec2(0.0, arm), center + egui::vec2(0.0, arm)],
            egui::Stroke::new(1.0_f32, col),
        );
    }

    draw_hud(ui, strip.hud_rect(rect), readout_line, img.zoom);

    // Optional on-image playhead: a thin vertical line at the current frame's
    // position across the IMAGE bounds (not the viewport). Off by default; the
    // colour is white and the opacity comes from the `playhead_alpha` setting. The
    // timeline itself lives in the transport bar — this is just a position marker.
    if let Some(pos) = seq_pos
        && playhead_alpha > 0.0
    {
        let a = (playhead_alpha.clamp(0.0, 1.0) * 255.0) as u8;
        let x = canvas.left() + pos.clamp(0.0, 1.0) * canvas.width();
        painter.line_segment(
            [egui::pos2(x, canvas.top()), egui::pos2(x, canvas.bottom())],
            egui::Stroke::new(1.0, egui::Color32::from_white_alpha(a)),
        );
    }

    // Flame-style scrub cursor: while LMB-scrubbing a sequence, a 0.5-opacity
    // white vertical line at the (clamped) pointer X with the target frame number
    // beside it. The X is clamped to the image bounds, so the line never runs past
    // the first/last frame.
    if let Some((first, last)) = seq_range
        && is_primary_scrub(&response)
        && let Some(p) = response.interact_pointer_pos()
    {
        draw_scrub_cursor(&painter, rect, canvas, p, first, last, theme::mono(ui.style()));
    }

    out
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// What the viewport interaction produced this frame.
#[derive(Default)]
pub struct ViewportOut {
    /// When the on-screen LMB scrubber was used over a sequence, the `0..=1`
    /// position to seek to (the caller maps it to a frame via
    /// [`crate::playback::Playback::scrub_fraction`]).
    pub scrub_to: Option<f32>,
}

/// Zoom by `factor` about the viewport centre (keyboard zoom — no cursor pivot).
fn zoom_toward_center(img: &mut ImageBackend, factor: f32) {
    let old = img.zoom;
    img.zoom = (img.zoom * factor).clamp(ZOOM_MIN, ZOOM_MAX);
    let scale = img.zoom / old;
    // Pivot is the viewport centre, so `pan` (already relative to centre) just scales.
    img.pan *= scale;
}

/// True when the primary button is scrubbing (dragged or tapped) — the gesture
/// that seeks a sequence in [`show`].
fn is_primary_scrub(response: &egui::Response) -> bool {
    response.dragged_by(egui::PointerButton::Primary)
        || response.clicked_by(egui::PointerButton::Primary)
}

/// Draw the Flame-style scrub cursor while LMB-scrubbing a sequence.
///
/// A half-opacity (0.5) white vertical line at the pointer X — clamped to the
/// image bounds (`img_rect`) so it never passes the first/last frame — spanning
/// the full viewport height (`rect`). The target frame number is drawn in a small
/// dark backdrop just to the RIGHT of the line at the pointer Y (flipped to the
/// left near the right edge so it stays on screen). `first`/`last` are the
/// sequence's frame-number range; the displayed frame is the nominal frame at the
/// scrub position (`first + round(frac * (last - first))`).
fn draw_scrub_cursor(
    painter: &egui::Painter,
    rect: egui::Rect,
    img_rect: egui::Rect,
    pointer: egui::Pos2,
    first: i32,
    last: i32,
    font: egui::FontId,
) {
    // Position along the scrub range (image X-extent), clamped to [first, last].
    let frac = ((pointer.x - img_rect.left()) / img_rect.width().max(1.0)).clamp(0.0, 1.0);
    let x = img_rect.left() + frac * img_rect.width();

    // Vertical line: white at 0.5 opacity, full viewport height.
    painter.line_segment(
        [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
        egui::Stroke::new(1.0, egui::Color32::from_white_alpha(128)),
    );

    // Target frame number, laid out so we can size + place a legible backdrop.
    let span = (last - first).max(0) as f32;
    let frame = first + (frac * span).round() as i32;
    let galley = painter.layout_no_wrap(
        format!("{frame}"),
        font,
        egui::Color32::WHITE,
    );
    let pad = egui::vec2(4.0, 2.0);
    let size = galley.size() + pad * 2.0;
    // Vertically centre on the pointer, kept inside the viewport.
    let y = (pointer.y - size.y * 0.5).clamp(rect.top(), (rect.bottom() - size.y).max(rect.top()));
    // Default to the RIGHT of the line; flip left if it would clip the edge.
    let right_x = x + 6.0;
    let bg_x = if right_x + size.x <= rect.right() {
        right_x
    } else {
        x - 6.0 - size.x
    };
    let bg = egui::Rect::from_min_size(egui::pos2(bg_x, y), size);
    painter.rect_filled(bg, 3.0, egui::Color32::from_black_alpha(128));
    painter.galley(bg.min + pad, galley, egui::Color32::WHITE);
}

fn zoom_toward_cursor(img: &mut ImageBackend, rect: egui::Rect, ui: &egui::Ui, factor: f32) {
    let old_zoom = img.zoom;
    img.zoom = (img.zoom * factor).clamp(ZOOM_MIN, ZOOM_MAX);
    let scale = img.zoom / old_zoom;
    let pivot = ui
        .input(|i| i.pointer.hover_pos())
        .unwrap_or_else(|| rect.center());
    let delta = pivot - rect.center();
    img.pan = delta + (img.pan - delta) * scale;
}

/// Map a screen position to a pixel index inside the data window, or `None` when
/// it falls outside it.
///
/// Both bounds matter: the previous version clamped with `.max(0)`, so anything
/// left of or above the image reported pixel 0 — a readout for a pixel the
/// pointer was nowhere near, which was also pinnable.
fn screen_to_pixel(
    pos: egui::Pos2,
    img_rect: egui::Rect,
    zoom: f32,
    width: u32,
    height: u32,
) -> Option<(u32, u32)> {
    if zoom <= 0.0 {
        return None;
    }
    let rel = pos - img_rect.min;
    let px = (rel.x / zoom).floor();
    let py = (rel.y / zoom).floor();
    // `floor()` of a non-finite (or huge) value is not representable as i64, so
    // range-check in f64 before the cast.
    if !(px.is_finite() && py.is_finite()) {
        return None;
    }
    let (px, py) = (px as f64, py as f64);
    (px >= 0.0 && py >= 0.0 && px < f64::from(width) && py < f64::from(height))
        .then_some((px as u32, py as u32))
}

/// The readout coordinate label for local pixel `(px, py)`.
///
/// EXR pixel coordinates are absolute (the data window's `min` need not be the
/// origin), and what a compositor cares about is usually the position on the
/// DISPLAY window. Both are shown when they differ; an ordinary image, whose two
/// windows coincide at the origin, keeps the plain `(x,y)` it always had.
fn coords_label(win: &Windows, px: u32, py: u32) -> String {
    let (x, y) = (
        win.data_min.0 as i64 + px as i64,
        win.data_min.1 as i64 + py as i64,
    );
    if win.data_min == win.display_min && win.display_min == (0, 0) {
        return format!("({x},{y})");
    }
    let (dx, dy) = (x - win.display_min.0 as i64, y - win.display_min.1 as i64);
    format!("({x},{y}) disp({dx},{dy})")
}

/// The corner HUD: the pixel readout (top left) and the zoom (bottom right), on
/// translucent plates so they stay legible over any image.
fn draw_hud(ui: &mut egui::Ui, rect: egui::Rect, readout: Option<String>, zoom: f32) {
    use egui_viewport_hud::{HudBlock, HudCorner, HudStyle};
    let mut blocks = vec![HudBlock::new(HudCorner::BottomRight, vec![format!("{:.0}%", zoom * 100.0)])];
    if let Some(line) = readout {
        blocks.push(HudBlock::new(HudCorner::TopLeft, vec![line]));
    }
    let style = HudStyle {
        text_color: egui::Color32::from_gray(230),
        font_size: theme::mono(ui.style()).size,
        ..HudStyle::default()
    };
    egui_viewport_hud::draw_hud(ui, rect, &blocks, &style);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x: f32, y: f32, w: f32, h: f32) -> egui::Rect {
        egui::Rect::from_min_size(egui::pos2(x, y), egui::vec2(w, h))
    }

    /// The HUD blocks keep clear of the toolbar strip wherever it is: below a strip (or the
    /// collapsed toggle) at the top, above one dragged to the bottom; a strip elsewhere, none, or
    /// a toggle in the bottom-LEFT corner (the zoom block is bottom-right) changes nothing.
    #[test]
    fn hud_rect_keeps_clear_of_the_strip_on_either_edge() {
        let view = rect(0.0, 0.0, 800.0, 600.0);
        let top = |r| Strip { rect: r, hot: false }.hud_rect(view);
        let strip_top = rect(4.0, 4.0, 792.0, 28.0);
        assert_eq!(top(strip_top).min.y, strip_top.bottom() + 2.0, "rolled out at the top");
        assert_eq!(top(strip_top).max, view.max);
        let toggle = rect(4.0, 4.0, 22.0, 28.0);
        assert_eq!(top(toggle).min.y, toggle.bottom() + 2.0, "the collapsed toggle at the top-left corner");
        let strip_bottom = rect(4.0, 568.0, 792.0, 28.0);
        let hud = top(strip_bottom);
        assert_eq!(hud.max.y, strip_bottom.top() - 2.0, "dragged to the bottom: the zoom block moves up");
        assert_eq!(hud.min, view.min);
        let toggle_bottom = rect(4.0, 568.0, 22.0, 28.0);
        assert_eq!(top(toggle_bottom), view, "a bottom-left toggle does not reach the zoom block");
        assert_eq!(top(rect(4.0, 300.0, 792.0, 28.0)), view, "a strip mid-view reaches neither block");
        assert_eq!(Strip::NONE.hud_rect(view), view);
    }

    /// The viewport's top band follows the strip's actual (animated) width, never a target flag,
    /// and is zero for anything but a strip docked at the top.
    #[test]
    fn band_tracks_the_roll_and_only_a_docked_strip() {
        let view = rect(0.0, 0.0, 800.0, 600.0);
        let band = |r| Strip { rect: r, hot: false }.band(view, 4.0, 22.0);
        let full = 4.0 + 28.0 + 2.0;
        assert_eq!(band(rect(4.0, 4.0, 792.0, 28.0)), full, "rolled out");
        assert_eq!(band(rect(4.0, 4.0, 22.0, 28.0)), 0.0, "the bare toggle takes none");
        let half = band(rect(4.0, 4.0, 22.0 + (792.0 - 22.0) / 2.0, 28.0));
        assert!((half - full / 2.0).abs() < 1e-3, "mid-roll: {half}");
        assert_eq!(band(rect(4.0, 4.3, 792.0, 28.0)), full + 0.3, "a sub-point offset is still docked");
        assert_eq!(band(rect(4.0, 40.0, 792.0, 28.0)), 0.0, "dragged down: it floats over the image");
        assert_eq!(band(egui::Rect::NOTHING), 0.0);
        let tiny = rect(0.0, 0.0, 20.0, 600.0);
        assert_eq!(Strip { rect: rect(4.0, 4.0, 22.0, 28.0), hot: false }.band(tiny, 4.0, 22.0), 0.0, "no room to roll out");
    }

    /// The readout must say "outside" rather than clamp to pixel 0. The old
    /// `px.max(0)` reported (0,0) for anything left of / above the image, and let
    /// that phantom pixel be pinned.
    #[test]
    fn screen_to_pixel_rejects_outside_instead_of_clamping() {
        // 4x2 image at 2x zoom, drawn at (100, 50): 8x4 screen pixels.
        let r = rect(100.0, 50.0, 8.0, 4.0);
        assert_eq!(
            screen_to_pixel(egui::pos2(100.0, 50.0), r, 2.0, 4, 2),
            Some((0, 0))
        );
        assert_eq!(
            screen_to_pixel(egui::pos2(107.9, 53.9), r, 2.0, 4, 2),
            Some((3, 1))
        );
        // Left of / above: NOT pixel 0.
        assert_eq!(screen_to_pixel(egui::pos2(99.0, 51.0), r, 2.0, 4, 2), None);
        assert_eq!(screen_to_pixel(egui::pos2(101.0, 49.0), r, 2.0, 4, 2), None);
        // Right of / below (already rejected before, still rejected).
        assert_eq!(screen_to_pixel(egui::pos2(108.0, 51.0), r, 2.0, 4, 2), None);
        assert_eq!(screen_to_pixel(egui::pos2(101.0, 54.0), r, 2.0, 4, 2), None);
        // Degenerate / non-finite input must not panic or cast out of range.
        assert_eq!(screen_to_pixel(egui::pos2(101.0, 51.0), r, 0.0, 4, 2), None);
        assert_eq!(
            screen_to_pixel(egui::pos2(f32::NAN, 51.0), r, 2.0, 4, 2),
            None
        );
        assert_eq!(
            screen_to_pixel(egui::pos2(f32::INFINITY, 51.0), r, 2.0, 4, 2),
            None
        );
    }

    /// Readout coordinates are EXR-absolute, and also display-window-relative
    /// whenever the two windows differ.
    #[test]
    fn coords_label_reports_data_and_display_window() {
        // Ordinary image: both windows at the origin -> the plain (x,y).
        let plain = Windows {
            display_min: (0, 0),
            display_size: (16, 16),
            data_min: (0, 0),
        };
        assert_eq!(coords_label(&plain, 3, 4), "(3,4)");

        // Data window offset inside a larger display window.
        let offset = Windows {
            display_min: (0, 0),
            display_size: (100, 50),
            data_min: (10, 20),
        };
        assert_eq!(coords_label(&offset, 1, 2), "(11,22) disp(11,22)");

        // A display window that does not start at the origin (a cropped comp).
        let shifted = Windows {
            display_min: (-8, -8),
            display_size: (32, 32),
            data_min: (0, 0),
        };
        assert_eq!(coords_label(&shifted, 0, 0), "(0,0) disp(8,8)");
    }
}
