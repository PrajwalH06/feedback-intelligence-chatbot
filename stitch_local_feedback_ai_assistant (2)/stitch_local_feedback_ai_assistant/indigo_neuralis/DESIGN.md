# Design System Document

## 1. Overview & Creative North Star: "The Cognitive Architect"
This design system moves away from the "SaaS-generic" dashboard look toward a sophisticated, editorial "Deep Tech" aesthetic. The Creative North Star is **The Cognitive Architect**: a visual philosophy that treats data not as static charts, but as a living, layered intelligence. 

We break the "template" feel through **Intentional Asymmetry** and **Tonal Depth**. Instead of rigid, equal-width columns, we use varying container widths to guide the eye toward the most critical insights. We emphasize high-contrast typography scales—pairing the technical precision of *Inter* with the architectural elegance of *Manrope*—to ensure the interface feels like a high-end command center rather than a simple reporting tool.

---

## 2. Colors: Tonal Depth & The "No-Line" Rule
The palette is rooted in deep indigos (`#0b1326`) and tech-focused tertiary cyans. The goal is to create an immersive environment where information "glows" against a dark, stable foundation.

- **Primary (`#bac3ff`):** Use for active states and critical data paths.
- **Tertiary (`#00daf3`):** Reserved exclusively for AI-driven insights and the "Bot" persona in chat interfaces to signify high-tech intelligence.
- **The "No-Line" Rule:** 1px solid borders for sectioning are strictly prohibited. Boundaries must be defined through background shifts. For example, a `surface-container-low` component should sit on a `surface` background to create a "carved" effect rather than a "boxed" one.
- **Surface Hierarchy & Nesting:** Treat the UI as physical layers. 
    - Base: `surface` (`#0b1326`)
    - Sections: `surface-container-low` (`#131b2e`)
    - Elements: `surface-container-high` (`#222a3d`)
- **The "Glass & Gradient" Rule:** Floating modals or chat bubbles must use Glassmorphism. Apply `surface-container-highest` at 70% opacity with a `24px` backdrop blur. Use a subtle linear gradient (Top-Left: `primary`, Bottom-Right: `primary-container`) for main Action buttons to provide a "lit from within" feel.

---

## 3. Typography: The Editorial Tech Scale
The system uses a dual-sans-serif approach to balance functionality with a premium brand voice.

- **Display & Headlines (Manrope):** Chosen for its geometric, architectural proportions. Use `display-lg` and `headline-lg` for high-level data summaries. The generous x-height of Manrope commands authority.
- **Body & Labels (Inter):** The workhorse for data. Use `body-md` for general dashboard content and `label-sm` for chart axes. Inter's technical precision ensures legibility in dense data environments.
- **Intentional Contrast:** Always pair a `headline-sm` (Manrope) with a `label-md` (Inter, All Caps, 0.05em letter spacing) for card headers to create an editorial, "report-style" feel.

---

## 4. Elevation & Depth: Tonal Layering
Traditional drop shadows are too "heavy" for a Deep Tech aesthetic. We use light and tone to imply height.

- **The Layering Principle:** Stack containers to create lift. A `surface-container-lowest` card placed atop a `surface-container-low` background creates a natural recession, perfect for secondary background tasks.
- **Ambient Shadows:** For floating AI chat bubbles or tooltips, use an ultra-diffused shadow: `0px 12px 32px rgba(0, 0, 0, 0.4)`. The shadow should feel like a soft glow rather than a dark stain.
- **The "Ghost Border" Fallback:** If accessibility requires a stroke (e.g., in high-contrast mode), use `outline-variant` at 15% opacity. Never use 100% opaque borders.
- **Backdrop Blurs:** Any element that overlaps data (like a dropdown or a chat modal) must use a `12px` to `20px` backdrop blur to maintain the sense of "layered transparency."

---

## 5. Components: Precision Primitives

### Cards (Stats & Charts)
- **Styling:** No borders. Use `surface-container-high` with a `xl` (0.75rem) corner radius.
- **Layout:** Forbid divider lines. Separate the "Stat Value" from the "Trend Label" using 1.5rem of vertical whitespace (`spacing-6`).

### The Conversational Interface
- **Bot Bubbles:** Use `secondary-container` with a `lg` radius. The bottom-left corner should be `sm` to "point" to the avatar.
- **User Bubbles:** Use a "Ghost" style—`outline-variant` at 20% opacity with white text. This emphasizes that the AI (Bot) is the "active" assistant.
- **Input Field:** Use `surface-container-highest` with a `full` (pill) radius. No border; use a `primary` glow (2px outer spread) only when focused.

### Charts & Data Viz
- **Trend Lines:** Use `tertiary` (`#00daf3`) with a 2px stroke width. Apply a subtle gradient fill underneath the line (Tertiary to Transparent).
- **Bar Charts:** Use `primary` for historical data and `primary-fixed-dim` for projected/AI-predicted data.

### Buttons
- **Primary:** Gradient fill (`primary` to `primary-container`), `md` radius, `label-md` uppercase text.
- **Tertiary (Ghost):** No background. Use `on-surface-variant` text. On hover, shift background to `surface-bright`.

---

## 6. Do's and Don'ts

### Do:
- **Do** use `surface-container` shifts to group related data.
- **Do** use `manrope` for any number larger than 24px.
- **Do** allow for "white space" (negative space) between dashboard cards to let the dark background "breathe."

### Don't:
- **Don't** use 1px solid borders to separate list items; use 8px of padding and a `surface-container` hover state instead.
- **Don't** use pure black (`#000000`). Always use the deep indigo `surface` (`#0b1326`) to maintain the "Deep Tech" tonal range.
- **Don't** use standard "Warning Yellow." Use the `error_container` tokens for a more sophisticated, muted alert state.