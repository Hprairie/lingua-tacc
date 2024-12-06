import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime


def read_jsonl(file_path):
    """
    Reads a JSONL file and returns a DataFrame with relevant columns.
    """
    data = []
    with open(file_path, "r") as f:
        for line in f:
            record = json.loads(line)
            record["created_at"] = datetime.fromisoformat(record["created_at"])
            data.append(record)
    return pd.DataFrame(data)


def plot_loss_over_time(file_paths, save_path="loss_over_time.png"):
    """
    Reads multiple JSONL files, extracts 'created_at' and 'loss/out', and plots loss over time.
    """
    plt.figure(figsize=(10, 6))

    for file_path in file_paths:
        df = read_jsonl(file_path)
        # Sort data by 'created_at' to ensure time progression in the plot
        df = df.sort_values(by="global_step")

        plt.plot(df["global_step"], df["loss/out"], label=file_path)

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss Over Time")
    plt.legend()
    plt.grid(True)
    # Save the plot to the specified path
    plt.savefig(save_path, format="png")
    plt.close()  # Close the plot to free up memory


# Example usage
file_paths = [
    "/home1/09753/hprairie/scratch/mamba/130m-run2/base3/metrics.jsonl",
    "/home1/09753/hprairie/scratch/mamba/130m-run2/full/metrics.jsonl",
]
plot_loss_over_time(file_paths, save_path="loss_over_step_2.png")
