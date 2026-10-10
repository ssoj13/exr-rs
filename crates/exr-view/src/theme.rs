//! Centralized theme — semantic colors + egui visuals.
//!
//! Image-viewer-relevant text + status color tokens only. The theme is stored
//! in egui memory so any panel can read it via [`current`] and apply via [`set`].

use egui::Color32;

/// Semantic color tokens for the viewer UI.
#[derive(Debug, Clone)]
pub struct ColorTokens {
    pub text_primary: Color32,
    pub text_secondary: Color32,
    pub text_ghost: Color32,

    pub status_error: Color32,
    pub status_warn: Color32,
    pub status_info: Color32,
    pub status_ok: Color32,
}

impl ColorTokens {
    pub fn dark() -> Self {
        Self {
            text_primary: Color32::from_rgb(200, 200, 200),
            text_secondary: Color32::from_rgb(160, 160, 160),
            text_ghost: Color32::from_rgb(120, 120, 120),
            status_error: Color32::from_rgb(220, 53, 69),
            status_warn: Color32::from_rgb(218, 165, 32),
            status_info: Color32::from_rgb(23, 162, 184),
            status_ok: Color32::from_rgb(120, 200, 120),
        }
    }

    pub fn light() -> Self {
        Self {
            text_primary: Color32::from_rgb(30, 30, 30),
            text_secondary: Color32::from_rgb(90, 90, 90),
            text_ghost: Color32::from_rgb(150, 150, 150),
            status_error: Color32::from_rgb(190, 30, 45),
            status_warn: Color32::from_rgb(180, 130, 0),
            status_info: Color32::from_rgb(0, 130, 160),
            status_ok: Color32::from_rgb(40, 160, 40),
        }
    }
}

/// A named theme = dark/light flag + its color tokens.
#[derive(Debug, Clone)]
pub struct Theme {
    pub name: &'static str,
    pub dark: bool,
    pub colors: ColorTokens,
}

impl Theme {
    pub fn dark_theme() -> Self {
        Self {
            name: "Dark",
            dark: true,
            colors: ColorTokens::dark(),
        }
    }

    pub fn light_theme() -> Self {
        Self {
            name: "Light",
            dark: false,
            colors: ColorTokens::light(),
        }
    }

    /// Build egui `Visuals` for this theme (egui's stock dark/light base).
    pub fn to_visuals(&self) -> egui::Visuals {
        if self.dark {
            egui::Visuals::dark()
        } else {
            egui::Visuals::light()
        }
    }
}

const THEME_ID: &str = "exr_view_dark";

/// Read the current theme from egui memory (defaults to Dark).
pub fn current(ctx: &egui::Context) -> Theme {
    let is_dark = ctx.data_mut(|d| {
        d.get_persisted::<bool>(egui::Id::new(THEME_ID))
            .unwrap_or(true)
    });
    if is_dark {
        Theme::dark_theme()
    } else {
        Theme::light_theme()
    }
}

/// Persist the theme in egui memory and apply it: visuals, the text sizes
/// (`persist::Fonts`, Settings > UI), and the shared widget tokens (`egui-widgets-config`)
/// derived from those sizes, so text laid out by egui and text painted directly (HUD,
/// timeline ruler) is sized from the one place. Call on a theme or size change, not every frame.
pub fn set(ctx: &egui::Context, theme: &Theme, fonts: &crate::persist::Fonts) {
    let fonts = fonts.clamped();
    ctx.data_mut(|d| d.insert_persisted(egui::Id::new(THEME_ID), theme.dark));
    ctx.set_visuals(theme.to_visuals());
    ctx.all_styles_mut(|style| {
        for (text_style, font) in style.text_styles.iter_mut() {
            font.size = match text_style {
                egui::TextStyle::Small => fonts.small,
                egui::TextStyle::Body => fonts.body,
                egui::TextStyle::Monospace => fonts.monospace,
                egui::TextStyle::Button => fonts.button,
                egui::TextStyle::Heading => fonts.heading,
                egui::TextStyle::Name(_) => fonts.body,
            };
        }
        // Labels are read-only text, not text fields: no I-beam cursor, no drag
        // selection over lists that are meant to be clicked.
        style.interaction.selectable_labels = false;
    });
    let fonts = egui_widgets_config::Fonts::from_text_styles(&ctx.global_style());
    egui_widgets_config::WidgetsConfig::install(
        ctx,
        egui_widgets_config::WidgetsConfig {
            fonts,
            ..Default::default()
        },
    );
}

/// Register the Phosphor icon font (the icon set of the shared widgets) next to
/// egui's default fonts. Once per context.
pub fn install_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    egui_phosphor::add_to_fonts(&mut fonts, egui_phosphor::Variant::Regular);
    ctx.set_fonts(fonts);
}

/// The monospace font of the current style (painted readouts).
pub fn mono(style: &egui::Style) -> egui::FontId {
    egui::TextStyle::Monospace.resolve(style)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persist::Fonts;

    /// Each text style takes its own size from `Fonts`, held to the allowed range.
    #[test]
    fn set_applies_each_text_size() {
        let ctx = egui::Context::default();
        let fonts = Fonts { heading: 14.0, body: 12.0, monospace: 11.0, button: 13.0, small: 99.0, ..Fonts::default() };
        set(&ctx, &Theme::dark_theme(), &fonts);
        let size = |s: egui::TextStyle| s.resolve(&ctx.global_style()).size;
        assert_eq!(size(egui::TextStyle::Heading), 14.0);
        assert_eq!(size(egui::TextStyle::Body), 12.0);
        assert_eq!(size(egui::TextStyle::Monospace), 11.0);
        assert_eq!(size(egui::TextStyle::Button), 13.0);
        assert_eq!(size(egui::TextStyle::Small), *Fonts::SIZE.end());
    }
}
