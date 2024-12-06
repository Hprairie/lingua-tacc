import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from typing import List, Tuple


def collect_token_counts(file_path: str) -> List[int]:
    """
    Collect all token counts from the JSONL file.

    Args:
        file_path (str): Path to the JSONL file

    Returns:
        List[int]: List of token counts
    """
    token_counts = []

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    record = json.loads(line.strip())
                    if "metadata" in record and "token_count" in record["metadata"]:
                        token_counts.append(record["metadata"]["token_count"])
                except (json.JSONDecodeError, KeyError) as e:
                    continue
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []

    return token_counts


def create_distribution_plot(token_counts: List[int], num_buckets: int = 100) -> None:
    """
    Create and display a distribution plot of token counts.

    Args:
        token_counts (List[int]): List of token counts
        num_buckets (int): Number of buckets for the histogram
    """
    if not token_counts:
        print("No data to plot")
        return

    plt.figure(figsize=(12, 6))

    # Create the histogram
    counts, bins, patches = plt.hist(token_counts, bins=num_buckets, edgecolor="black", color="skyblue", alpha=0.7)

    # Calculate statistics
    mean_tokens = np.mean(token_counts)
    median_tokens = np.median(token_counts)

    # Add vertical lines for mean and median
    plt.axvline(mean_tokens, color="red", linestyle="dashed", linewidth=2, label=f"Mean: {mean_tokens:.1f}")
    plt.axvline(median_tokens, color="green", linestyle="dashed", linewidth=2, label=f"Median: {median_tokens:.1f}")

    # Customize the plot
    plt.title("Distribution of Token Counts", pad=20, fontsize=14)
    plt.xlabel("Token Count", fontsize=12)
    plt.ylabel("Number of Documents", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add summary statistics as text
    stats_text = (
        f"Total Documents: {len(token_counts):,}\n"
        f"Mean: {mean_tokens:.1f}\n"
        f"Median: {median_tokens:.1f}\n"
        f"Min: {min(token_counts):,}\n"
        f"Max: {max(token_counts):,}"
    )
    plt.text(
        0.95,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Calculate and display bucket ranges
    bucket_ranges = []
    for i in range(len(bins) - 1):
        bucket_ranges.append((bins[i], bins[i + 1], counts[i]))

    # Print bucket information
    print("\nBucket Distribution:")
    print(f"{'Range':>20} | {'Count':>8} | {'Percentage':>10}")
    print("-" * 45)
    total_docs = len(token_counts)
    for start, end, count in bucket_ranges:
        percentage = (count / total_docs) * 100
        print(f"{f'{start:.0f}-{end:.0f}':>20} | {f'{count:.0f}':>8} | {f'{percentage:.1f}%':>10}")

    plt.tight_layout()
    plt.savefig("token_analysis.png", format="png")
    plt.close()  # Close the plot to free up memory


# Example usage
if __name__ == "__main__":
    file_path = "/home1/09753/hprairie/scratch/data-shuffled/fineweb_edu_10bt/fineweb_edu_10bt.val.jsonl"
    token_counts = collect_token_counts(file_path)

    if token_counts:
        create_distribution_plot(token_counts)
    else:
        print("No valid records were found")
