from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def test_readme_paper_release_contract() -> None:
    text = README.read_text(encoding="utf-8")
    lowered = text.lower()

    for url in (
        "https://jialonglong.github.io/Anchored-Path-Matching/",
        "https://jialonglong.github.io/Anchored-Path-Matching/paper.pdf",
        "https://github.com/JIAlonglong/Anchored-Path-Matching",
        "https://huggingface.co/JIAlonglong/Anchored-Path-Matching",
    ):
        assert url in text

    figure_paths = (
        "docs/public/images/teaser.webp",
        "docs/public/images/method.webp",
        "docs/public/images/video-quality.webp",
        "docs/public/images/real-robot.webp",
    )
    for relative in figure_paths:
        assert relative in text
        assert (ROOT / relative).is_file()

    assert "first intermediate state" in lowered
    assert "finite-interval" in lowered
    assert "instantaneous direction" not in lowered
    assert "1, 2, and 4" in text
    assert all(value in text for value in ("80.9%", "88.8%", "91.4%"))
    assert "32.317 dB" in text
    assert "21.3×" in text
    assert "MODEL_ZOO.md" in text
    assert "overview video" not in lowered
    assert "<video" not in lowered
