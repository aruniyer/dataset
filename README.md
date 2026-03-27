# c-CRAB Public Release

This repository is the public release of the c-CRAB benchmark and experiment pipeline. It contains:

- released benchmark data derived from `inclusionAI/SWE-CARE`
- stage-wise funnel files used in the paper experiments
- compressed raw experiment outputs
- scripts for comment filtering, Docker environment building, test generation, agent resolution, and tool evaluation

## Setup

```bash
uv sync
```

LLM-backed scripts use LiteLLM. Set `OPENAI_API_KEY` for OpenAI models or `ANTHROPIC_API_KEY` for Anthropic models. The Claude Code based scripts also expect Claude credentials at `~/.claude/.credentials.json`, which are mounted into Docker containers by default.

## Repository Structure

```text
pipeline/                  Core pipeline logic and prompts
execution/                 Docker image builders, runtime helpers, and image tests
results_preprocessed/      Released benchmark subset for this repo
results_pipeline_funnel/   Released stage0-stage4 JSONL files and funnel summary
raw_results_compressed/    Compressed raw outputs from released experiments

run_batch_filter.py            Batch HIGH/LOW comment filtering
preprocess_swe_care.py         Build the released benchmark subset from filter outputs
run_batch_testgen.py           Batch executable test generation
run_testgen_full.sh            Convenience wrapper for full test-split test generation
run_batch_baselines.py         Collect raw reviews from baseline tools/agents
run_batch_agent_resolution.py  Run agent-based comment resolution on verified instances
run_batch_tool_eval.py         Evaluate tool findings by applying them and rerunning tests
```

## Benchmark Data

The checked-in benchmark data is in [`results_preprocessed/`](./results_preprocessed):

- `preprocess_dataset.jsonl`: 410 released benchmark instances
- `instance-ids.txt`: the same 410 instances in normalized `reviewbench/...` form, used by Docker/image tooling

Each JSONL row is still a SWE-CARE instance, but `reference_review_comments` has already been reduced to the retained benchmark comments. Use this file as the main benchmark input if you want the released subset directly.

If you want to rebuild that subset from the original SWE-CARE test split, run:

```bash
uv run python run_batch_filter.py --split test --output-dir results_filter_test
uv run python preprocess_swe_care.py \
  --split test \
  --filter-dir results_filter_test \
  --output-dir results_preprocessed_regen \
  --jsonl-name preprocess_dataset.jsonl \
  --instance-ids-name instance-ids.txt
```

The public release also includes stage-wise funnel files in [`results_pipeline_funnel/`](./results_pipeline_funnel):

| Stage | File | Instances | Comments |
| --- | --- | ---: | ---: |
| Stage 0 | `stage0_full_test_split.jsonl` | 671 | 1313 |
| Stage 1 | `stage1_comment_filter.jsonl` | 410 | 595 |
| Stage 2 | `stage2_docker_image.jsonl` | 410 | 595 |
| Stage 3 | `stage3_testgen_verified.jsonl` | 339 | 485 |
| Stage 4 | `stage4_agent_resolved.jsonl` | 184 | 234 |

## Replicating Experiments

The main released experiments can be rerun with the scripts in the repo root.

### 1. Build Docker environments

Use the released instance list:

```bash
uv run python -m execution.build_swe_care \
  --split test \
  --instance results_preprocessed/instance-ids.txt \
  --max-workers 4
```

### 2. Regenerate executable tests

```bash
./run_testgen_full.sh \
  --instances-file results_preprocessed/instance-ids.txt \
  --workers 4 \
  --output-dir results_testgen
```

### 3. Collect baseline reviews

```bash
uv run python run_batch_baselines.py \
  --split test \
  --instances-file results_preprocessed/instance-ids.txt \
  --tools pr-agent devin claude-code codex \
  --output-dir baselines_output \
  --workers 4
```

Configure the corresponding external tool credentials before running this step.

### 4. Run agent resolution on Stage 3 instances

Downstream scripts consume the released Stage 3 file directly:

```bash
uv run python run_batch_agent_resolution.py \
  --stage3-file results_pipeline_funnel/stage3_testgen_verified.jsonl \
  --testgen-dir results_testgen \
  --output-dir results_agent_resolution \
  --workers 4
```

### 5. Evaluate tool findings with executable tests

Repeat once per tool:

```bash
uv run python run_batch_tool_eval.py \
  --tool pr-agent \
  --stage3-file results_pipeline_funnel/stage3_testgen_verified.jsonl \
  --testgen-dir results_testgen \
  --tool-results-dir baselines_output \
  --output-dir results_eval_pr-agent \
  --workers 4
```

The evaluation scripts have older local default paths baked in, so pass explicit directories when reproducing from this public release.

## Released Result Data

The repository ships released artifacts in [`raw_results_compressed/`](./raw_results_compressed):

- `reviews_combined.zip`: raw review outputs and parsed findings from baseline tools
- `testgen_combined.zip`: generated tests and per-instance `result.json`
- `agent_resolution_combined.zip`: agent-resolution outputs on Stage 3 instances
- `results_eval_pr-agent.zip`
- `results_eval_devin.zip`
- `results_eval_claude-code.zip`
- `results_eval_codex.zip`

Use [`results_pipeline_funnel/`](./results_pipeline_funnel) for released stage membership, and the zip files in [`raw_results_compressed/`](./raw_results_compressed) for raw per-instance outputs.

## Prompt Locations

The main prompts used in the released pipeline are inline in [`pipeline/`](./pipeline):

- `pipeline/comment_filter.py`: HIGH/LOW comment-quality filtering prompt
- `pipeline/test_generator.py`: test generation prompt and repair-feedback loop prompt
- `pipeline/agent_resolver.py`: agent-resolution prompt and tool-finding application prompt

For baseline-review collection, the review prompts for PR-Agent, CodeRabbit, Devin, Claude Code, and Codex are defined in `run_batch_baselines.py`.
