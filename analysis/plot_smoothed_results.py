import pandas as pd
import matplotlib.pyplot as plt
import json


def read_jsonl(file_path):
    """
    Reads a JSONL file and returns a DataFrame with relevant columns.
    """
    data = []
    with open(file_path, "r") as f:
        for line in f:
            record = json.loads(line)
            data.append(record)
    return pd.DataFrame(data)


def smooth_series(series, method="rolling", window_size=10, alpha=0.3):
    """
    Smooths a pandas Series using a rolling average or EMA.

    Args:
        series (pd.Series): Series to smooth.
        method (str): Smoothing method - 'rolling' or 'ema'.
        window_size (int): Window size for rolling average.
        alpha (float): Smoothing factor for EMA.

    Returns:
        pd.Series: Smoothed series.
    """
    if method == "rolling":
        return series.rolling(window=window_size, min_periods=1).mean()
    elif method == "ema":
        return series.ewm(alpha=alpha, adjust=False).mean()
    else:
        raise ValueError("Unsupported smoothing method. Use 'rolling' or 'ema'.")


def plot_smoothed_loss_over_steps(
    file_paths, save_path="loss_over_steps.png", smooth_method="rolling", window_size=10, alpha=0.3
):
    """
    Reads multiple JSONL files, smooths 'loss/out' over 'global_step', and saves a plot of smoothed loss over steps.

    Args:
        file_paths (list): List of file paths to JSONL files.
        save_path (str): Path to save the plot image.
        smooth_method (str): Smoothing method - 'rolling' or 'ema'.
        window_size (int): Window size for rolling average.
        alpha (float): Smoothing factor for EMA.
    """
    plt.figure(figsize=(10, 6))

    for file_path in file_paths:
        df = read_jsonl(file_path)
        # Sort data by 'global_step' to ensure step progression in the plot
        df = df.sort_values(by="global_step")

        # Smooth the loss series
        smoothed_loss = smooth_series(df["loss/out"], method=smooth_method, window_size=window_size, alpha=alpha)

        plt.plot(df["global_step"], smoothed_loss, label=file_path)

    plt.xlabel("Global Step")
    plt.ylabel("Smoothed Loss")
    plt.title("Smoothed Loss Over Global Steps")
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
plot_smoothed_loss_over_steps(
    file_paths, save_path="smoothed_loss_over_steps.png", smooth_method="rolling", window_size=10
)
