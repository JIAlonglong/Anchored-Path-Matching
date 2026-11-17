# Model zoo

The canonical model repository is
<https://huggingface.co/JIAlonglong/Anchored-Path-Matching>. Only target/EMA
inference tensors are eligible for publication. Online, optimizer, scheduler,
trainer, recovery, and rank-local states are excluded.

| Public ID | Paper role | Anchor | Finite local field | Projection |
|---|---|---:|---:|---:|
| `shared` | Stage I student | — | — | — |
| `continued` | matched-budget continued Stage I | none | no | no |
| `original-no-field` | Original anchor | original | no | no |
| `original-field-projected` | Original anchor + field | original | yes | yes |
| `restart-no-field` | Restart anchor | restart | no | no |
| `restart-field-no-projection` | Restart anchor + field | restart | yes | no |
| `apm` | Anchored Path Matching | restart | yes | yes |

An artifact is called a paper checkpoint only after its resolved objective,
data domain, source initialization, CFG contract, topology, target type, and
projection state all match. The model manifest labels every candidate
`exact`, `unresolved`, or `auxiliary`; names and metric similarity are not
accepted as identity evidence.

Verified artifacts use one shared base plus either allowlisted adapter tensors
or replacement tensors. `delta.safetensors` means byte-for-byte replacement of
the listed tensors, not arithmetic subtraction. The final manifest includes
materialization commands and a logical tensor-state digest.
