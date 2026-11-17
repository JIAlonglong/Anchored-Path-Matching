# Anchored Path Matching

Official implementation of **Distilling World Action Models for Efficient
Bimanual Manipulation via Anchored Path Matching**.

[Project page](https://jialonglong.github.io/Anchored-Path-Matching/) ·
[Paper](https://jialonglong.github.io/Anchored-Path-Matching/paper.pdf) ·
[Models](https://huggingface.co/JIAlonglong/Anchored-Path-Matching)

APM distills a joint video–action teacher into one student checkpoint that
runs at 1, 2, or 4 steps. Stage II restarts the teacher from a detached state
reached by the student. One continuation supplies an endpoint anchor and a
finite local-transition target. Modality-routed adapters update the video and
action paths, while anchor-priority projection preserves the endpoint gradient
when the local field gradient conflicts with it.

## Install

```bash
python -m pip install -e '.[dev]'
```

LingBot-VA and RoboTwin are optional external checkouts. They are not vendored.
The audited revisions and license notices are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Preflight

This command validates configuration without allocating CUDA memory, loading
model weights, or importing the simulator.

```bash test=preflight
python -m anchorstep.cli.train --config configs/model/lingbot_robotwin.yaml --config configs/train/stage2.yaml --config configs/ablation/apm.yaml --check-only
```

## Inference

`anchorstep.robotwin.inference.rollout_student` accepts `steps=1`, `2`, or `4`
and never queries the teacher. Download and materialization instructions for
verified weights are maintained in [MODEL_ZOO.md](MODEL_ZOO.md).

## Reproduction

- [Method and objective details](docs/method.md)
- [Stage I and Stage II training](docs/training.md)
- [RoboTwin evaluation](docs/evaluation.md)
- [Paper/ablation model map](MODEL_ZOO.md)

Every formal configuration records text-condition dropout, teacher-target
video/action CFG, deployment video/action CFG, and whether guidance is distilled
or composed at inference. `cfg_probability` is never interpreted as a guidance
scale. The paper recipe uses text dropout 0.1, teacher-target CFG 5/1,
deployment CFG 5/1, and explicit inference composition.

## Scope and safety

This is research code and simulator integration, not a production robot safety
controller. Validate action bounds, collision handling, emergency stops, and
task-specific constraints before physical deployment. No datasets, simulator
assets, teacher weights, or optimizer states are redistributed here. Code is
Apache-2.0; external assets retain their own terms.
