#!/usr/bin/env bash
set -euo pipefail
python -m anchorstep.cli.evaluate --config configs/model/lingbot_robotwin.yaml --config configs/eval/robotwin.yaml --config configs/local.example.yaml "$@"
