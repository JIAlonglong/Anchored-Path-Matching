# Training

## External assets

Provide compatible LingBot-VA code/model assets and RoboTwin data through a
local overlay based on `configs/local.example.yaml`. Do not commit that overlay.
Strict compatibility revisions are recorded in the model config and notices.

## Stage I preflight

```bash test=preflight
python -m anchorstep.cli.train --config configs/model/lingbot_robotwin.yaml --config configs/train/stage1.yaml --check-only
```

For a live run, add a private local config that supplies paths and a runtime
runner factory, then remove `--check-only` or run `scripts/train_stage1.sh`.
Stage I uses 4,000 optimization steps, batch size 32, and eight teacher
integration transitions.

## Stage II APM

```bash test=preflight
python -m anchorstep.cli.train --config configs/model/lingbot_robotwin.yaml --config configs/train/stage2.yaml --config configs/ablation/apm.yaml --check-only
```

Stage II adds 5,000 steps. The paper config samples training budgets from
`{1,2,4}`, uses finite local targets, and applies anchor-priority projection by
video/action parameter group after global averaging and before clipping.

To reproduce an ablation, replace `configs/ablation/apm.yaml` with another file
in `configs/ablation/`. Each states anchor source, local-field presence, and
projection explicitly.

## Guidance contract

The paper recipe records text dropout 0.1, teacher-target CFG 5/1, deployment
CFG 5/1, and `target_guidance_mode: inference_composed`. Text dropout is a
conditioning probability, not a guidance scale. A target-distilled run records
deployment CFG independently and normally deploys at 1/1.
