# Results

1705 runs. Seven models, two prompt modes, three difficulty strata,
two repeats. GSM8K test split, stratified by the number of calculator
annotations in the reference solution. Temperature 0, `num_ctx` 4096,
`num_predict` 1024, batch size 1. GeForce RTX 4060 (8 GB), measured idle
draw 52.1 W. Raw data in `grid.csv`, per-cell aggregates in
`per_cell.csv`, environment in `environment.txt`.

## 1. Joules per token and tokens per correct answer rank models in
## opposite orders

Mean across cells:

| model | J/token | tokens per correct |
| --- | --- | --- |
| qwen2.5:0.5b | 0.20 (best) | 1668 (worst) |
| granite3-moe:3b | 0.31 | 1107 |
| qwen2.5:1.5b | 0.69 | 461 |
| qwen2.5:7b | 2.43 (worst) | 218 (best) |

The cheapest model per token is the most expensive per solved task by a
factor of eight. Both quantities are energy-derived and both are in
current use, so the choice between them is not a technicality: it
reverses the recommendation.

## 2. Four fifths of the measured energy is the card idling

`qwen2.5:0.5b`, CoT, easy tasks: 8809 J raw, 1840 J after subtracting
the measured idle draw — 79% idle. `granite3-moe:3b`, direct, easy:
6344 J raw, 1089 J subtracted — 83%.

Rankings under `j_per_correct` change with that single methodological
decision: `mistral:latest` moves from 3rd to 7th, `qwen2.5:1.5b` from
6th to 2nd, `granite3-moe:3b` from 2nd to 4th. Published inference
energy figures do not state which convention they use.

## 2b. The idle-subtraction decision destroys the ranking it produces

Ranking under `j_per_correct`, raw against idle-subtracted:

| model | raw | idle-subtracted |
| --- | --- | --- |
| granite3-moe:3b | 1 | 4 |
| mistral:latest | 2 | 7 |
| qwen2.5:0.5b | 3 | 2 |
| qwen2.5:3b | 4 | 3 |
| qwen2.5:7b | 5 | 5 |
| qwen2.5:1.5b | 6 | 1 |
| llama3.1:8b | 7 | 6 |

Spearman rho between the two orderings is -0.107: the same metric over
the same runs produces essentially unrelated rankings depending on
whether idle draw counts as part of the cost of inference. Published
figures do not state which convention they follow.

## 3. Token thrift is a capability that scales with model size

Mean output tokens, with the direct-mode prompt asking for the final
number and no explanation:

| model | CoT | direct | ratio |
| --- | --- | --- | --- |
| granite3-moe:3b | 210.7 | 127.9 | 1.65 |
| qwen2.5:0.5b | 343.4 | 115.7 | 2.97 |
| mistral:latest | 218.5 | 65.0 | 3.36 |
| llama3.1:8b | 196.3 | 26.1 | 7.52 |
| qwen2.5:1.5b | 221.7 | 15.0 | 14.78 |
| qwen2.5:7b | 267.2 | 8.3 | 32.19 |
| qwen2.5:3b | 300.7 | 7.7 | 39.05 |

`qwen2.5:3b` complies in 7.7 tokens; `qwen2.5:0.5b` emits 115.7 despite
the same instruction, reasoning aloud where it was told not to. The
advice to pick a smaller model in order to emit fewer tokens is
therefore false at this end of the size range: the smaller model emits
more, because it cannot suppress the reasoning.

## 4. Cutting tokens costs almost all of the capability

`qwen2.5:7b` accuracy falls from 0.925 to 0.275 on easy tasks and from
0.825 to 0.050 on medium and hard when switched from CoT to direct.
`mistral:latest` scores 0.000 on hard direct, Wilson interval
[0.000, 0.088]. The token reduction that `j_per_token` rewards buys
essentially no useful output.

## 5. MoE at equal total size: cheaper per token, worse per hard task

`granite3-moe:3b` (3B total, ~800M active) against `qwen2.5:3b` (3B
dense), CoT mode:

| | granite3-moe:3b | qwen2.5:3b |
| --- | --- | --- |
| accuracy easy / medium / hard | 0.60 / 0.35 / 0.15 | 0.90 / 0.88 / 0.68 |
| J/token | 0.26 | 0.61 |
| J/correct, easy | 83.6 | 173.2 |
| J/correct, hard | 412.7 | 313.9 |

Reducing activated parameters lowers energy per token roughly in
proportion and lowers accuracy more than in proportion, so the advantage
inverts as difficulty rises. The MoE model's accuracy tracks its active
parameter count rather than its total: it scores below the 1.5B dense
model. This qualifies rather than confirms the claim that architecture
substitutes for scale.

## 6. Tokens per megawatt adds nothing to joules per token

`tps_per_mw` as computed here equals 1e6 / `j_per_token` exactly —
verified against the data (`j_per_token` 0.162 against `tps_per_mw`
6 163 536). The vertical axis of the published throughput-versus-latency
frontier restates energy per token with the sign reversed so that higher
reads as better.

## Limitations

Batch size 1 on a consumer card is the far-right end of the
throughput-latency trade-off; none of these figures extrapolate to
batched serving, where energy per token is far lower.

GSM8K appears in the training data of every model tested, so accuracy
here measures the energy cost of producing an answer of known
correctness, not reasoning ability.

Energy is measured at the GPU board through NVML's buffered power
samples. CPU, memory and power-supply losses are outside the boundary.
The instantaneous power query and the cumulative energy counter both
return NVML_ERROR_NOT_SUPPORTED on this card, and
`nvidia-smi --query-gpu=power.draw` returns N/A; tooling built on that
field alone would treat the card as unmeasurable.

Truncation at `num_predict` 1024 affected 2.6% of `qwen2.5:0.5b` runs
and under 1% elsewhere, all in CoT mode — itself an instance of the
same mechanism.

Model tags are floating; digests are recorded in `environment.txt`.
`mistral:latest` resolved to a 7.2B build, `Q4_K_M`.

`gemma4:12b` was excluded from the grid: at 8.9 GB it does not fit the
8 GB card and Ollama splits it 31%/69% between CPU and GPU, putting part
of the work outside the measurement boundary.