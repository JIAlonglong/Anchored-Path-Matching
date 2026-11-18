# README Link Badges Design

## Intent

Replace the plain hero navigation links with a compact, recognizable badge row
so readers can scan the project resources immediately.

## Design

- Use four linked Shields.io badges: Project Page, Paper, GitHub, and Models.
- Use a distinct relevant icon for each destination while keeping one coherent
  flat badge style and restrained colors.
- Preserve the current URLs and centered hero layout.
- Do not add a video badge, video URL, or embedded media.

## Approaches considered

- **Selected:** Shields.io badges with icons. Best recognition with minimal
  repository maintenance.
- Emoji plus text links. Lighter, but visually inconsistent across platforms.
- Local SVG assets. More control, but unnecessary files and maintenance for a
  four-link navigation row.

## Verification

Extend the README contract test to require all four linked badge images and to
continue rejecting video links or embeds. Run formatting, lint, and the full
test suite before attempting the existing fast-forward push to `main` again.
