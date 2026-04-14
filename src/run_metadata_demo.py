import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from provider_factory import create_provider
from providers.zhipu_provider import MissingZhipuConfigurationError


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"


def load_local_env_if_present():
    env_path = ROOT / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and key not in __import__("os").environ:
            __import__("os").environ[key] = value


def load_queries():
    queries = []
    with (DATA_DIR / "queries_demo.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            queries.append(json.loads(line))
    return queries


def load_tools(filename):
    with (DATA_DIR / filename).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_condition(queries, tools, condition_name, provider):
    rows = []
    for item in queries:
        result = provider.select_tool(item["query"], tools)
        rows.append(
            {
                "query_id": item["query_id"],
                "query": item["query"],
                "gold_tool": item["gold_tool"],
                f"{condition_name}_prediction": result["selected_tool"],
                f"{condition_name}_correct": int(result["selected_tool"] == item["gold_tool"]),
                f"{condition_name}_provider_used": result.get("provider_used", provider.provider_name),
                f"{condition_name}_fallback_used": int(bool(result.get("fallback_used", False))),
            }
        )
    return pd.DataFrame(rows)


def accuracy(df, prefix):
    return float(df[f"{prefix}_correct"].mean())


def pick_cases(df):
    recovered = df[
        (df["clean_prediction"] == df["gold_tool"])
        & (df["attacked_prediction"] != df["gold_tool"])
        & (df["canonical_prediction"] == df["gold_tool"])
    ]
    still_flipped = df[
        (df["clean_prediction"] == df["gold_tool"])
        & (df["attacked_prediction"] != df["gold_tool"])
        & (df["canonical_prediction"] != df["gold_tool"])
    ]
    stable = df[
        (df["clean_prediction"] == df["gold_tool"])
        & (df["attacked_prediction"] == df["gold_tool"])
        & (df["canonical_prediction"] == df["gold_tool"])
    ]

    chosen = []
    if not recovered.empty:
        row = recovered.iloc[0]
        chosen.append(
            (
                row,
                "Attacked metadata makes a broad tool look attractive, while canonical wording restores the task-specific match.",
            )
        )
    if not still_flipped.empty:
        row = still_flipped.iloc[0]
        chosen.append(
            (
                row,
                "This case shows that canonicalization helps but does not fix every attack-induced error.",
            )
        )
    additional_recovered = recovered.iloc[1:3]
    for _, row in additional_recovered.iterrows():
        chosen.append(
            (
                row,
                "This is another recovered sample where broad attacked wording overclaims capability but canonical wording narrows it back down.",
            )
        )
    if not stable.empty and len(chosen) < 3:
        row = stable.iloc[0]
        chosen.append(
            (
                row,
                "This stable case helps show the demo is not failing everywhere; some queries remain robust across metadata variants.",
            )
        )
    return chosen[:3]


def build_report(df, metrics, attacked_extra_counts, provider_name):
    cases = pick_cases(df)
    if metrics["canonical_recovery_rate"] == 1.0 and metrics["attack_flip_rate"] > 0:
        recovery_sentence = (
            "In this toy setup, canonical metadata recovers all of the cases that were flipped by attacked metadata."
        )
    else:
        recovery_sentence = (
            "The canonical metadata removes broad wording and recovers part of the lost accuracy, though not every flipped case is fixed."
        )

    lines = [
        "# Metadata Influence Demo Report",
        "",
        "## Summary",
        "",
        f"- Provider: {provider_name}",
        f"- Number of queries: {len(df)}",
        f"- Clean accuracy: {metrics['clean_accuracy']:.2%}",
        f"- Attacked accuracy: {metrics['attacked_accuracy']:.2%}",
        f"- Canonical accuracy: {metrics['canonical_accuracy']:.2%}",
        f"- Attack flip rate: {metrics['attack_flip_rate']:.2%}",
        f"- Canonical recovery rate: {metrics['canonical_recovery_rate']:.2%}",
        "",
        "The attacked metadata makes several tools sound broader and more generally helpful, which pulls the selector away from the gold tool on some queries.",
        recovery_sentence,
        "",
        "## Extra Selections Under Attacked Metadata",
        "",
    ]

    if attacked_extra_counts:
        for tool_id, count in attacked_extra_counts.items():
            lines.append(f"- {tool_id}: {count}")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Representative Cases",
            "",
        ]
    )

    for index, (row, explanation) in enumerate(cases, start=1):
        lines.extend(
            [
                f"### Case {index}",
                f"- Query: {row['query']}",
                f"- Gold tool: {row['gold_tool']}",
                f"- Clean prediction: {row['clean_prediction']}",
                f"- Attacked prediction: {row['attacked_prediction']}",
                f"- Canonical prediction: {row['canonical_prediction']}",
                f"- Explanation: {explanation}",
                "",
            ]
        )

    return "\n".join(lines).strip() + "\n"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        default="mock",
        choices=["mock", "zhipu"],
        help="Selector provider to use.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)
    load_local_env_if_present()
    provider = create_provider(args.provider)

    queries = load_queries()
    clean_tools = load_tools("tools_clean.json")
    attacked_tools = load_tools("tools_attacked.json")
    canonical_tools = load_tools("tools_canonical.json")

    try:
        clean_df = run_condition(queries, clean_tools, "clean", provider)
        attacked_df = run_condition(queries, attacked_tools, "attacked", provider)
        canonical_df = run_condition(queries, canonical_tools, "canonical", provider)
    except MissingZhipuConfigurationError as exc:
        raise SystemExit(
            f"{exc}\nHint: copy .env.example to a local .env file or export the variables in your shell, then rerun with --provider zhipu."
        )

    df = clean_df.merge(
        attacked_df[
            [
                "query_id",
                "attacked_prediction",
                "attacked_correct",
                "attacked_provider_used",
                "attacked_fallback_used",
            ]
        ],
        on="query_id",
    ).merge(
        canonical_df[
            [
                "query_id",
                "canonical_prediction",
                "canonical_correct",
                "canonical_provider_used",
                "canonical_fallback_used",
            ]
        ],
        on="query_id",
    )

    df["attack_flipped"] = (
        (df["clean_prediction"] == df["gold_tool"])
        & (df["attacked_prediction"] != df["clean_prediction"])
    ).astype(int)
    df["canonical_recovered"] = (
        (df["clean_prediction"] == df["gold_tool"])
        & (df["attacked_prediction"] != df["gold_tool"])
        & (df["canonical_prediction"] == df["gold_tool"])
    ).astype(int)

    clean_accuracy = accuracy(df, "clean")
    attacked_accuracy = accuracy(df, "attacked")
    canonical_accuracy = accuracy(df, "canonical")

    attack_flip_mask = (df["clean_prediction"] == df["gold_tool"]) & (
        df["attacked_prediction"] != df["clean_prediction"]
    )
    total_clean_correct = max(int((df["clean_prediction"] == df["gold_tool"]).sum()), 1)
    total_attack_flips = int(attack_flip_mask.sum())
    total_recoveries = int(df["canonical_recovered"].sum())

    metrics = {
        "clean_accuracy": clean_accuracy,
        "attacked_accuracy": attacked_accuracy,
        "canonical_accuracy": canonical_accuracy,
        "attack_flip_rate": total_attack_flips / total_clean_correct,
        "canonical_recovery_rate": (
            total_recoveries / total_attack_flips if total_attack_flips else 0.0
        ),
    }

    extra_selected_counter = Counter()
    for _, row in df.iterrows():
        if row["attacked_prediction"] != row["clean_prediction"]:
            extra_selected_counter[row["attacked_prediction"]] += 1

    results_filename = "results_metadata_demo.csv" if args.provider == "mock" else f"results_metadata_demo_{args.provider}.csv"
    report_filename = "report_metadata_demo.md" if args.provider == "mock" else f"report_metadata_demo_{args.provider}.md"
    results_path = OUTPUT_DIR / results_filename
    df.to_csv(results_path, index=False)

    report_text = build_report(df, metrics, dict(extra_selected_counter), args.provider)
    report_path = OUTPUT_DIR / report_filename
    report_path.write_text(report_text, encoding="utf-8")

    print("Saved:", results_path)
    print("Saved:", report_path)
    print("")
    print("Metrics")
    print(f"- provider: {args.provider}")
    for key, value in metrics.items():
        print(f"- {key}: {value:.4f}")
    print("- attacked_extra_selected:", dict(extra_selected_counter))


if __name__ == "__main__":
    main()
