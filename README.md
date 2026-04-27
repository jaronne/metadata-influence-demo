# Metadata Influence Demo

This repository is a minimal research framework for studying black-box agent tool selection under metadata variation. It starts with a small runnable metadata influence demo, then extends that demo into a lightweight attribution and defense pipeline for risky tool routing behavior.

At the core, the project asks two linked questions:

1. Can wording changes in tool metadata flip tool selection on the same task?
2. Can simple interventions and post-selection defenses help attribute and reduce those flips?

The current codebase stays intentionally simple:

- small synthetic query and tool sets
- lightweight selector providers
- no retrieval
- no heavy orchestration framework
- directly inspectable CSV and Markdown outputs

The repository currently supports two selector providers:

- `mock`: the default offline keyword-based selector
- `zhipu`: a provider slot for Zhipu-style experiments, kept compatible with the same CLI workflow

## Project Structure

```text
.
├─ AGENTS.md
├─ README.md
├─ .env.example
├─ requirements.txt
├─ data/
│  ├─ queries_demo.jsonl
│  ├─ tools_clean.json
│  ├─ tools_attacked.json
│  └─ tools_canonical.json
├─ src/
│  ├─ provider_factory.py
│  ├─ selector.py
│  ├─ run_metadata_demo.py
│  ├─ plot_metadata_demo.py
│  └─ providers/
│     ├─ mock_provider.py
│     └─ zhipu_provider.py
└─ outputs/
   ├─ results_metadata_demo.csv
   ├─ report_metadata_demo.md
   └─ metadata_demo_bar.png
```

## Install

```bash
pip install -r requirements.txt
```

## Method Overview

The repository is organized as a black-box tool selection experiment with three layers:

1. Metadata influence demo
   This compares clean, attacked, and canonicalized tool metadata to show that wording alone can redirect tool choice.
2. Intervention layer
   This perturbs the input query through controlled rewrites and perturbs the candidate list through tool-order shuffling.
3. Risk scoring and defense layer
   This summarizes sensitivity into a small set of scores and maps them to simple post-selection actions.

The intent is not to claim full runtime verification. Instead, the project provides a compact framework for attribution: when a selector changes its chosen tool, we can ask whether the change appears tied to metadata reliance, weak intent support, or instability under simple perturbations.

## Configure Zhipu API

The Zhipu provider reads these environment variables:

- `ZHIPU_API_KEY`
- `ZHIPU_BASE_URL`
- `ZHIPU_MODEL`

Recommended defaults:

- `ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4/`
- `ZHIPU_MODEL=glm-4-flash`

You have two configuration options:

1. Set environment variables in your shell.
2. Copy `.env.example` to a local `.env` file and fill in your own values.

Do not commit your real `.env` or API key.

### PowerShell example

```powershell
$env:ZHIPU_API_KEY="your_key_here"
$env:ZHIPU_BASE_URL="https://open.bigmodel.cn/api/paas/v4/"
$env:ZHIPU_MODEL="glm-4-flash"
```

### Local `.env` example

Create a file named `.env` in the project root based on `.env.example`:

```text
ZHIPU_API_KEY=your_key_here
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
ZHIPU_MODEL=glm-4-flash
```

## Intervention Protocol

The intervention data lives under `data/interventions/`.

For the current risk scoring experiment, each original query has three controlled semantic rewrites in `data/interventions/query_rewrites.jsonl`. These rewrites are designed to preserve:

- task goal
- key entities
- time and location constraints
- numeric constraints
- output granularity

Only surface form is changed, such as sentence structure, wording, politeness, or redundancy. This lets the experiment estimate whether a prediction is robust to harmless restatements of the same intent.

The experiment also applies tool-order perturbations by shuffling the candidate list multiple times with deterministic seeds.

## Run The Demo

### Mock mode

```bash
python src/run_metadata_demo.py --provider mock
python src/plot_metadata_demo.py --provider mock
```

This keeps the original offline demo behavior and writes:

- `outputs/results_metadata_demo.csv`
- `outputs/report_metadata_demo.md`
- `outputs/metadata_demo_bar.png`

### Zhipu mode

```bash
python src/run_metadata_demo.py --provider zhipu
python src/plot_metadata_demo.py --provider zhipu
```

This writes separate files so you can compare providers:

- `outputs/results_metadata_demo_zhipu.csv`
- `outputs/report_metadata_demo_zhipu.md`
- `outputs/metadata_demo_bar_zhipu.png`

If `ZHIPU_API_KEY` is missing, the script exits with a clear configuration hint instead of asking you to paste the key into the chat.

## Risk Scoring Experiment

The risk scoring experiment keeps `run_metadata_demo.py` unchanged and adds a separate entry point:

```bash
python src/run_risk_scoring_experiment.py --provider mock
python src/run_risk_scoring_experiment.py --provider zhipu
```

Optional arguments:

```bash
python src/run_risk_scoring_experiment.py --provider mock --num-perturbations 5 --seed 42
```

Outputs:

- `outputs/risk_scores_mock.csv`
- `outputs/risk_report_mock.md`
- `outputs/risk_scores_zhipu.csv`
- `outputs/risk_report_zhipu.md`

For each query, the experiment:

1. selects a tool under attacked metadata
2. re-runs the selection on the three query rewrites
3. re-runs the selection under canonical metadata
4. re-runs the selection under several tool-order shuffles
5. computes risk scores and a defense action

### Score Definitions

- `ISS`:
  Intent Support Score. This is the proportion of query rewrites that keep the original attacked-metadata prediction. Higher `ISS` means the prediction is more stable under semantic paraphrase.
- `MRS`:
  Metadata Reliance Score. This is `1 - canonical_keep_rate`, where `canonical_keep_rate` is the proportion of canonical-metadata runs that preserve the original attacked-metadata prediction. Higher `MRS` means the original choice appears more dependent on attacked wording.
- `INS`:
  Instability Score. This is `1 - perturbation_keep_rate`, where `perturbation_keep_rate` is the proportion of shuffled candidate-order runs that preserve the original attacked-metadata prediction. Higher `INS` means the choice is more fragile under simple candidate perturbations.
- `risk_score`:
  A capped aggregate risk score that increases with metadata reliance and instability, and decreases with intent support.

### Defense Policy

The current defense policy is intentionally simple:

- low risk: `allow`
- medium risk: `canonical_reselect`
- high risk: `user_confirm`

This makes the repository more than a single metadata demo. It becomes a compact experiment harness for attributing black-box tool selection behavior and testing whether lightweight defenses improve robustness without introducing a large systems stack.
