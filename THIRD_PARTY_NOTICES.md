# Third-party notices

Anchored Path Matching is distributed under the Apache License 2.0. This
notice records compatibility targets and source adaptations; it does not
relicense external model weights, datasets, assets, or dependencies.

## LingBot-VA

- Upstream repository: <https://github.com/Robbyant/lingbot-va>
- License: Apache License 2.0.
- Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
- Audited code revision: `58c2ae5bac46bd8114065bea9d7d256eb67c16c3`.
- Compatible model revision: `8c9dea8abbc5c91cc9e18bc3264b8915083bbe70`.

The optional backend adapts the public LingBot-VA model, conditioning, data,
and checkpoint interfaces behind a lazy compatibility boundary. LingBot-VA
source and model files are not vendored; users provide them separately under
their upstream terms.

## RoboTwin

- Upstream repository: <https://github.com/RoboTwin-Platform/RoboTwin>
- License: MIT License.
- Audited compatibility revision: `30954692d06ba7e89f7a6b76064f4062c488fa81`.

No RoboTwin source, task assets, demonstrations, or simulator files are
redistributed. The optional evaluator consumes a user-supplied checkout.

## Flash-WAM and prior AnchorStep implementation

- Flash-WAM repository license: Apache License 2.0.
- Copyright 2024-2025 The Flash-WAM Team Authors. All rights reserved.
- Audited distillation revision: `4cbecf208836739a86377c0efa4e7835d66b1dbb`.

The finite-interval flow-map objective and backend boundary were adapted from
the related Flash-WAM distillation implementation, then extended with paired
restart/local targets and modality-separated anchor-priority projection. The
public implementation removes environment-specific launch and experiment
overlay code.
