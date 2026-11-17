# Anchored Path Matching — project page

Academic project page for **Distilling World Action Models for Efficient Bimanual Manipulation via Anchored Path Matching**.

Adapted from [REAL_wb](https://github.com/JIAlonglong/REAL_wb) and [Roman Hauksson-Neill's academic project page template](https://research-template.roman.technology). Template attribution is retained in the website footer (CC BY-SA 4.0).

## Local development

Use Node.js 24 or newer:

```sh
cd docs
npm ci
npm run dev
```

The local URL includes `/Anchored-Path-Matching/`. Content is in `src/paper.mdx`; author affiliations and resource links are in `src/pages/index.astro`. Figures are in `public/images/`, and `public/paper.pdf` is the named preprint version.

```sh
npm run build
npm run preview
```

## Publishing

The repository-level GitHub Actions workflow builds `docs/` and deploys `docs/dist/` to GitHub Pages. Select **GitHub Actions** as the Pages source. The production URL is:

https://jialonglong.github.io/Anchored-Path-Matching/

The Paper button serves the local named PDF. Replace it with an arXiv URL once an identifier is available, and update the citation component. No arXiv identifier or acceptance status is assumed.

## Content provenance

All result values and figures come from the authors' manuscript. Flash-WAM's one-step RoboTwin result is attributed to DirectWAM; its other budgets are locally evaluated. The video examples are selected qualitative illustrations, as disclosed in their caption. There are no video assets yet: the website presents the actual paper's still images.

## Accessibility and behavior

Sampling-budget buttons expose their selected state to assistive technology. Without JavaScript, all three result tables remain visible. Image links open full-resolution figures. BibTeX copy has a manual-selection fallback. Horizontal table scrolling is contained within the table on small screens.
