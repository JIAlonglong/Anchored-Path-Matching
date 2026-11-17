#!/usr/bin/env bash
set -euo pipefail
python -m anchorstep.cli.train --config configs/model/lingbot_robotwin.yaml --config configs/train/stage2.yaml --config configs/ablation/apm.yaml --config configs/local.example.yaml "$@"
