# Metadata Influence Demo

This project is a minimal runnable research demo for showing how tool metadata wording can influence tool selection.

It demonstrates two points:

1. Changing metadata wording can flip tool selection on the same query set.
2. Canonicalized or neutralized metadata can recover part of the flipped cases.

The demo supports two selector providers:

- `mock`: the default offline keyword-based selector
- `zhipu`: a real-model selector using the OpenAI Python SDK compatible interface for Zhipu

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
