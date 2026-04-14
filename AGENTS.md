# Project context

This project is a minimal research demo for tool selection security in closed-source LLM agents.

## Goal

Show two things clearly:

1. Metadata wording changes can flip tool selection
2. Canonicalized / neutralized metadata can recover some flipped cases

## Priorities

- Simple and runnable code
- Clear outputs for advisor review
- Interpretable CSV / Markdown / PNG artifacts
- No complex framework
- No retrieval
- No full runtime verification yet

## Expected outputs

- outputs/results_metadata_demo.csv
- outputs/report_metadata_demo.md
- outputs/metadata_demo_bar.png

## Implementation style

- Prefer simple Python
- Prefer clear logic over cleverness
- Use a small mock selector instead of real model APIs
- Make the demo easy to explain in a meeting