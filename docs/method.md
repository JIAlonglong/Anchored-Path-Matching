# Method

## Stage I: arbitrary finite intervals

Stage I samples two times `t > r`, interpolates one shared noisy video–action
state, and asks the student to match an eight-transition teacher continuation
from `t` to `r`. Video and action errors remain separate and use weights 1.0
and 0.1. The backend protocol keeps this mathematics independent of LingBot-VA
and RoboTwin.

## Stage II: paired targets at a student state

For a deployment budget `K ∈ {1, 2, 4}`, APM rolls out a detached student
prefix. `K=1` uses the prior; `K=2` uses its only intermediate state; `K=4`
selects a correction state with the documented Beta(5,2) bias. From that exact
state, one frozen teacher continuation provides the zero-noise restart endpoint.
The field objective separately matches the frozen teacher's instantaneous
direction and the student's equal-clock direction at that same detached state
and correction time.

The endpoint and field queries therefore start from the same detached state.
The Stage I objective is retained as the base loss.

## Routed updates and anchor priority

The shared base is frozen. Video and action FFN adapters use rank 64; temporal
adapters use rank 128. After gradient accumulation and distributed averaging,
endpoint and local gradients are grouped by modality. If their dot product is
negative, only the conflicting component of the local gradient is removed.
The endpoint gradient and retained Stage I gradient are unchanged. Projection
runs before clipping and AdamW.
