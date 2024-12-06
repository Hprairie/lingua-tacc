import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from typing import List, Tuple


def collect_token_counts(file_path: str) -> Tuple[List[int], List[int]]:
    """
    Collect token counts from the JSONL file, separating counts within and outside the 0-2048 range.

    Args:
        file_path (str): Path to the JSONL file

    Returns:
        Tuple[List[int], List[int]]: (counts within range, counts outside range)
    """
    token_counts_in_range = []
    token_counts_overflow = []

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    record = json.loads(line.strip())
                    if "metadata" in record and "token_count" in record["metadata"]:
                        count = record["metadata"]["token_count"]
                        if 0 <= count <= 2048:
                            token_counts_in_range.append(count)
                        else:
                            token_counts_overflow.append(count)
                except (json.JSONDecodeError, KeyError) as e:
                    continue
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return [], []

    return token_counts_in_range, token_counts_overflow


def create_distribution_plot(token_counts: List[int], overflow_counts: List[int], num_buckets: int = 32) -> None:
    """
    Create and display a distribution plot of token counts between 0 and 2048.

    Args:
        token_counts (List[int]): List of token counts within range
        overflow_counts (List[int]): List of token counts outside range
        num_buckets (int): Number of buckets for the histogram
    """
    if not token_counts:
        print("No data to plot within the 0-2048 range")
        return

    plt.figure(figsize=(15, 8))

    # Create the histogram with fixed range
    counts, bins, patches = plt.hist(
        token_counts, bins=num_buckets, range=(0, 2048), edgecolor="black", color="skyblue", alpha=0.7  # Fixed range
    )

    # Calculate statistics for in-range tokens
    mean_tokens = np.mean(token_counts)
    median_tokens = np.median(token_counts)

    # Add vertical lines for mean and median
    plt.axvline(mean_tokens, color="red", linestyle="dashed", linewidth=2, label=f"Mean: {mean_tokens:.1f}")
    plt.axvline(median_tokens, color="green", linestyle="dashed", linewidth=2, label=f"Median: {median_tokens:.1f}")

    # Customize the plot
    plt.title("Distribution of Token Counts (0-2048 tokens)", pad=20, fontsize=14)
    plt.xlabel("Token Count", fontsize=12)
    plt.ylabel("Number of Documents", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add summary statistics as text
    total_docs = len(token_counts) + len(overflow_counts)
    overflow_percentage = (len(overflow_counts) / total_docs * 100) if total_docs > 0 else 0

    stats_text = (
        f"Documents (0-2048): {len(token_counts):,}\n"
        f"Documents (>2048): {len(overflow_counts):,} ({overflow_percentage:.1f}%)\n"
        f"Mean (in range): {mean_tokens:.1f}\n"
        f"Median (in range): {median_tokens:.1f}\n"
        f"Min: {min(token_counts):,}\n"
        f"Max (in range): {max(token_counts):,}"
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

    # Calculate bucket size
    bucket_size = 2048 / num_buckets

    # Print bucket information
    print("\nBucket Distribution (0-2048 tokens):")
    print(f"{'Range':>20} | {'Count':>8} | {'Percentage':>10}")
    print("-" * 45)
    for i in range(len(counts)):
        start = i * bucket_size
        end = (i + 1) * bucket_size
        percentage = (counts[i] / len(token_counts)) * 100 if len(token_counts) > 0 else 0
        if counts[i] > 0:  # Only show non-empty buckets
            print(f"{f'{start:.0f}-{end:.0f}':>20} | {f'{counts[i]:.0f}':>8} | {f'{percentage:.1f}%':>10}")

    # Print overflow information
    if overflow_counts:
        print("\nOverflow Statistics (>2048 tokens):")
        print(f"Total documents: {len(overflow_counts):,}")
        print(f"Min: {min(overflow_counts):,}")
        print(f"Max: {max(overflow_counts):,}")
        print(f"Mean: {np.mean(overflow_counts):.1f}")
        print(f"Median: {np.median(overflow_counts):.1f}")

    plt.tight_layout()
    plt.savefig("token_analysis2.png", format="png")
    plt.close()  # Close the plot to free up memory


# Example usage
if __name__ == "__main__":
    file_path = "/home1/09753/hprairie/scratch/data-shuffled/fineweb_edu_10bt/fineweb_edu_10bt.val.jsonl"
    token_counts_in_range, token_counts_overflow = collect_token_counts(file_path)

    if token_counts_in_range or token_counts_overflow:
        create_distribution_plot(token_counts_in_range, token_counts_overflow)
    else:
        print("No valid records were found")
