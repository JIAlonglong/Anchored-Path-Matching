import { defineConfig } from 'astro/config';
import tailwindcss from '@tailwindcss/vite';
import icon from 'astro-icon';
import mdx from '@astrojs/mdx';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
export default defineConfig({
  site: 'https://jialonglong.github.io',
  base: '/Anchored-Path-Matching',
  vite: { plugins: [tailwindcss()] },
  markdown: { remarkPlugins: [remarkMath], rehypePlugins: [rehypeKatex] },
  integrations: [icon(), mdx()],
  image: { responsiveStyles: true },
});
