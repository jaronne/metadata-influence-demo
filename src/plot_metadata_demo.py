import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        default="mock",
        choices=["mock", "zhipu"],
        help="Provider suffix to read the correct CSV output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_filename = "results_metadata_demo.csv" if args.provider == "mock" else f"results_metadata_demo_{args.provider}.csv"
    image_filename = "metadata_demo_bar.png" if args.provider == "mock" else f"metadata_demo_bar_{args.provider}.png"
    df = pd.read_csv(OUTPUT_DIR / results_filename)
    accuracy_values = [
        df["clean_correct"].mean(),
        df["attacked_correct"].mean(),
        df["canonical_correct"].mean(),
    ]
    labels = ["Clean", "Attacked", "Canonical"]
    colors = ["#4C956C", "#C75146", "#2E86AB"]

    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(labels, accuracy_values, color=colors)
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Tool Selection Accuracy by Metadata Condition")

    for bar, value in zip(bars, accuracy_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.0%}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    output_path = OUTPUT_DIR / image_filename
    plt.savefig(output_path, dpi=160)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
