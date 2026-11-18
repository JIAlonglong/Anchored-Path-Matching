# README Link Badges Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the README hero's four plain links with four icon-bearing Shields.io badges while preserving destinations and excluding video media.

**Architecture:** Keep the change entirely in README Markdown. Strengthen the existing release-contract test so the badge image URLs, link destinations, labels, and no-video rule remain coupled.

**Tech Stack:** GitHub-flavored Markdown, Shields.io static badge URLs, pytest

**Spec:** `docs/superpowers/specs/2026-11-18-readme-link-badges-design.md`

## Global Constraints

- Preserve the Project Page, Paper, GitHub, and Hugging Face destinations exactly.
- Use four centered, flat Shields.io badges with relevant icons and restrained colors.
- Do not add a video badge, video URL, or embedded media.
- Do not modify the README body or training behavior for this task.

## Review Focus

- Badge link and destination accidentally diverge: test each complete Markdown link.
- Shields query parameters break Markdown: use URL-encoded values and assert the exact image URL.
- A badge is lost or duplicated: assert exactly four Shields badge images in the hero.
- Video media reappears under another syntax: retain URL-extension and HTML embed rejection.
- The centered hero layout breaks: keep all badges inside the existing `<div align="center">` block.

---

### Task 1: Hero resource badges

**Files:**
- Modify: `README.md`
- Modify: `tests/test_readme_release.py`
- Commit: `docs/superpowers/specs/2026-11-18-readme-link-badges-design.md`
- Commit: `docs/superpowers/plans/2026-11-18-readme-link-badges.md`

**Interfaces:**
- Consumes: the four canonical resource URLs already asserted by `test_readme_paper_release_contract`
- Produces: four linked Shields.io badge images in the centered README hero

- [ ] **Step 1: Strengthen the failing README contract test**

Assert the four complete badge Markdown links, their exact labels/icons/colors, exactly four Shields badge image URLs before the closing hero `</div>`, and the existing no-video restrictions.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python -m pytest -q tests/test_readme_release.py`

Expected: FAIL because the current hero contains plain text links rather than Shields badges.

- [ ] **Step 3: Replace the plain resource links**

In `README.md`, add centered flat badges for Project Page, Paper, GitHub, and Models using `simple-icons`-compatible symbols where available. Preserve the existing href destinations and add no video resource.

- [ ] **Step 4: Verify focused and full checks**

Run: `python -m pytest -q tests/test_readme_release.py tests/test_documented_commands.py`

Expected: `2 passed`.

Run: `PYTHONPATH=/tmp/apm-release-tools python -m ruff format --check . && PYTHONPATH=/tmp/apm-release-tools python -m ruff check . && python -m pytest -q`

Expected: formatting and lint pass; all tests pass.

- [ ] **Step 5: Commit and attempt fast-forward push**

Commit the README, test, spec, and plan. Fetch `origin/main`, verify the remote is only behind the local branch, then run `git push origin HEAD:main` without force.
