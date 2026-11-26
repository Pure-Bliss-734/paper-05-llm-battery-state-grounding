# State Sensitivity Is Not Grounding

Offline reproduction of *State Sensitivity Is Not Grounding: Dual Replay for LLM Battery Scheduling*, by Jiangwei Xue, Zhida Qin and Yuda Bi. [Paper and DOI](https://doi.org/10.5281/zenodo.22707487).

## Run

Use Python 3.12 and the versions in `requirements.txt`. Create the virtual environment outside the extracted package. After installing dependencies, all analyses run offline.

```sh
python3.12 -m venv ../battery-env
source ../battery-env/bin/activate
python -m pip install -r requirements.txt
python reproduce.py --output ../battery-tables.json --rows ../battery-rows
```

The output contains Tables 1–9, matched temporal comparisons, E2 parser and projection denominators, F1 robustness and independent physical checks. The optional row directory must be new. It receives actions for all 3,060 planned calls and dual-replay verdicts for 900 complete-contract branches. The default analysis reconstructs deterministic proposal baselines and replays saved solver trajectories to evaluate their constraints, costs and distances. To also solve the battery-only and expanded gate problems again:

```sh
python reproduce.py --refit --output ../battery-refit.json
```

Table 6 intervals reproduce when resampling branch rows. Resampling the 60 independent scenario clusters as described in the caption gives different limits. Both computations and all four pairs of intervals are reported in [Table 6 comparisons](TABLE_6.md).

This command verifies the downloaded archive and runs the full analysis in a temporary extraction. It uses the pinned NumPy and SciPy versions from the active interpreter when available. If they are missing, it can use `uv` to run the same replay offline from previously cached wheels in an isolated environment. If the wheels are unavailable, use the installation instructions above. Substitute the ZIP path for the placeholder.

<!-- RELEASE_COMMAND:verify -->
```sh
python3.12 tools/verify_public.py --archive <ARCHIVE_PATH>
```
<!-- END_RELEASE_COMMAND:verify -->

## Experimental workflow and returns

Run the historical protocol without contacting a provider. The optional output, opened as a new file, contains regenerated requests in frozen block queue and stage order:

```sh
python scaffold.py --output ../battery-schedule.jsonl
```

The scaffold constructs the E2 prefix, parses its saved return, and forks C, M, S, K and C2. M receives the parsed model carrier and skips when there is no usable prefix state. The five E2 suffixes can run concurrently. F1 and F0 reconstruct four complete-contract messages in their original per-block order. The two Qwen3.7 strata share their original interleaved block queue. Queue and fork order describe experimental dependencies; parallel wall-clock arrival can differ. All 3,058 issued bodies are compared with the frozen requests. Every default table reproduction also runs this check.

| Tier | Planned logical calls | Physical attempts | Additional attempts | Returned choices | Attempts ending in transport failure | Skips |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E2 | 2,160 | 2,201 | 43 | 2,146 | 55 | 2 |
| F1 | 480 | 482 | 2 | 480 | 2 | 0 |
| F0 | 120 | 120 | 0 | 120 | 0 | 0 |
| Qwen3.7 F1 | 240 | 240 | 0 | 240 | 0 | 0 |
| Qwen3.7 F0 subset | 60 | 60 | 0 | 60 | 0 | 0 |
| Total | 3,060 | 3,103 | 45 | 3,046 | 57 | 2 |

Twenty-eight logical positions had more than one attempt. Sixteen eventually returned a choice; twelve exhausted their attempts. All recorded successful choices contained nonempty assistant text, choice index zero and finish reason `stop`. The twelve exhausted E2 positions carry the recorder's empty-string placeholder. The two skipped positions have null content. Neither represents a returned empty message. No successful response was recorded for the 57 failed tries.

In `data/attempt_outcomes.jsonl` each slot carries its ordered attempt sequence. A choice hash identifies the scientific choice object formed from its index, finish reason, role and exact content. Recompute it with the UTF-8 canonical JSON rule below. The 3,046 verbatim outputs are stored once in `data/responses.jsonl`. The queue, per-block stages and randomization order are in `data/call_plan.jsonl`. `analysis/prompt_protocol.py` and `analysis/schedule-prompt.txt` construct stimuli from the frozen scene. `analysis/carrier_parser.py` and `analysis/schedule_parser.py` implement the two output contracts. `analysis/plant.py` implements distance, replay and optimization. `analysis/physics_check.py` evaluates physical constraints independently. `reproduce.py` aggregates these results, applies the published failure and sampling rules and reports the two Table 6 interval calculations.

## Study inputs

| File | Observation and role |
| --- | --- |
| `data/admission.jsonl`, `data/scenarios.json` | The admission population has 280 candidates, 205 admitted and 60 selected. Scenario keys join to the dispatch problems and matched pairs. Frozen 15-minute load, forecast PV, price, active limits, two initial SOCs and MPC witness vectors support replay. Rejected candidates lack complete inputs for counterfactual reselection. |
| `data/requests.jsonl`, `data/responses.jsonl` | One entry per unique scientific request body and one per planned prefix or branch, including failed and skipped calls. Request, returned-content and parsed-output digests bind each representation. Research messages and model names have their original content. Start and end times in microseconds use one common relative origin. The last F1 to first F0 request interval is 154.65 hours. |
| `data/E2_projection.jsonl` | The archived E2 projection outcomes include 53 hard timeouts among 1,800 suffix branches. F1 gate analyses use their own complete population. |
| `data/solutions.jsonl`, `data/witnesses.jsonl` | Battery and expanded gate actions, curtailment, SOC paths, objective evidence and 60 deterministic witnesses support 2,760 gate and 60 witness cross-checks. Independent physics additionally checks 1,800 raw trajectories and 12 analytic fixtures. |
| `analysis/`, `reproduce.py`, `paper_values.json` | Frozen parsers, plant equations, optimizer, independent checker, reductions and reported cells. Scientific object hashes use SHA-256 of UTF-8 JSON with sorted keys and compact separators. Returned strings are hashed verbatim as UTF-8. |

Historical E2, F1 and F0 and both Qwen3.7-Plus sensitivity tiers correspond to the five rows of Table 1. The 2,160 E2 positions include shared prefixes and five suffix branches. Twelve transport failures and two skipped calls remain in the denominator. E2 parser failures follow the paper's zero-distance rule. F1/F0 and Qwen3.7 use their sparse-action parser. The 71 uncertified F1 minimum distances remain feasible upper bounds. Hosted responses are historical observations.

## Data sources and licenses

Author-written analysis and verification code uses the [MIT License](licenses/MIT.txt). Original documentation and project-derived table values use [CC BY 4.0](licenses/CC-BY-4.0.txt). Third-party inputs and hosted records are governed by their applicable upstream terms.

The 2024 solar series use NREL NSRDB GOES CONUS v4.0.0 for Los Angeles, San Diego, Fresno and Sacramento. NREL OEDI ComStock AMY2018 release 1 supplies the California large-office load profile. The day-ahead CAISO OASIS LMP series uses TH_NP15_GEN-APND. These synchronized inputs from separate sources serve a controlled benchmark. [NSRDB](https://nsrdb.nrel.gov/) · [ComStock](https://comstock.nrel.gov/) · [CAISO OASIS](https://oasis.caiso.com/).

Historical DeepSeek and Qwen requests and responses remain subject to the applicable providers' terms. NumPy and SciPy retain their upstream licenses.
