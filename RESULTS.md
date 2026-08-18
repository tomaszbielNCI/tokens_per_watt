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

 

## 6. Difficulty-conditioned routing: oracle headroom

Routing the query to the cheapest model that clears a given accuracy
floor, against the cheapest single model that clears the same floor on
every difficulty stratum. Energy per correct answer, idle-subtracted,
CoT mode.

| accuracy floor | routed | best fixed | saving |
| --- | --- | --- | --- |
| 0.50 | 386.9 J | 399.3 J (qwen2.5:1.5b) | 3.1% |
| 0.60 | 598.7 J | 699.1 J (qwen2.5:3b) | 14.4% |
| 0.70 | 777.1 J | 1164.9 J (qwen2.5:7b) | 33.3% |
| 0.80 | 865.1 J | 1164.9 J (qwen2.5:7b) | 25.7% |

The saving is not monotonic in the floor. It peaks where the
single-model policy is forced onto a larger model while routing can
still serve the easy stratum from a smaller one — a discontinuity
imposed by the granularity of the available model sizes rather than by
anything continuous in the workload.

Two qualifications. The policy is selected on the same runs it is
evaluated on and assumes difficulty is known before inference, so these
figures bound the headroom available to difficulty-aware routing rather
than describing a deployable saving; classifying difficulty at inference
time would itself consume energy. And the 33.3% figure at a 0.70 floor
rests on excluding qwen2.5:3b, whose hard-stratum accuracy is 0.675 with
a Wilson interval of [0.520, 0.799] that contains 0.70 — the threshold
crossing that produces the largest apparent saving is one this data
cannot resolve. The 25.7% figure at a 0.80 floor does not depend on an
unresolved comparison: the same interval's upper bound falls below 0.80.

## 7. The measurement boundary also moves the recommendation

The same routing policy, evaluated on the same runs, saves 25.7% on
idle-subtracted energy and 21.7% on raw energy at a 0.80 accuracy floor.
The model selection is identical under both conventions — 3b, 3b, 7b —
so the policy is robust to the boundary while the figure quoted for it
is not.

In absolute terms the ordering reverses: 549 J saved raw against 300 J
idle-subtracted. Both are correct under different deployment
assumptions. If the card remains powered regardless of load, as it does
on a desktop serving a single user, only the incremental 300 J is
avoided. If finishing sooner allows the device to be powered down or
reallocated, the full 549 J is. The measurement does not contain the
information needed to choose, and no published efficiency figure states
which assumption it carries.

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

