# RoboTwin evaluation

The release pins RoboTwin revision
`30954692d06ba7e89f7a6b76064f4062c488fa81` and the versioned LingBot RoboTwin
camera/action codec. Every record includes task, seed, Clean or Random domain,
student step budget, video CFG, action CFG, and success.

## Preflight

```bash test=preflight
python -m anchorstep.cli.evaluate --config configs/model/lingbot_robotwin.yaml --config configs/eval/robotwin.yaml --check-only
```

For live evaluation, add the private local overlay and use
`scripts/eval_robotwin.sh`. Episode IDs are deterministic, so an interrupted run
skips completed task/domain/seed tuples without duplicating them. Clean and
Random results are aggregated separately; the headline score is the macro
average of per-task success rates.

Use a new result directory whenever K or CFG changes. Report both each model's
selected deployment configuration and any matched-CFG comparison as distinct
protocols. Inference calls the student exactly K times and never calls the
teacher.
