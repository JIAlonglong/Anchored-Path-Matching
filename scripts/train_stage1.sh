#!/usr/bin/env bash
set -euo pipefail
python -m anchorstep.cli.train --config configs/model/lingbot_robotwin.yaml --config configs/train/stage1.yaml --config configs/local.example.yaml "$@"
