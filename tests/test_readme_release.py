from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def test_readme_paper_release_contract() -> None:
    text = README.read_text(encoding="utf-8")
    lowered = text.lower()
    prose = re.sub(r"\n>\s*", " ", lowered)

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
        assert re.search(rf"!\[[^\]]+\]\({re.escape(relative)}\)", text)
        assert (ROOT / relative).is_file()
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert tracked.returncode == 0

    assert "first intermediate state" in lowered
    assert "finite-interval" in lowered
    assert "current public stage ii implementation" in prose
    assert "instantaneous-field" in prose
    assert "do not yet implement" in prose
    assert "1, 2, and 4" in text
    assert (
        "| 1/1 | 82.3% | 76.1% | 79.2% | **84.3%** | **77.5%** | "
        "**80.9%** | **+1.7 pp** |" in text
    )
    assert (
        "| 2/2 | 89.6% | **87.7%** | 88.6% | **90.4%** | 87.1% | "
        "**88.8%** | **+0.1 pp** |" in text
    )
    assert (
        "| 4/4 | 90.5% | 89.1% | 89.8% | **92.5%** | **90.3%** | "
        "**91.4%** | **+1.6 pp** |" in text
    )
    assert "30.169 to 32.317 dB" in text
    assert "4323.6 ms" in text and "203.1 ms" in text and "21.3×" in text
    assert "MODEL_ZOO.md" in text
    assert "if `||ga|| = 0`" in lowered
    assert "computed before rounding" in lowered
    assert not re.search(r"<\s*(video|iframe)\b", lowered)
    assert not re.search(r"https?://[^\s)>]+\.(mp4|webm|mov)(?:[?#][^\s)>]*)?", lowered)
