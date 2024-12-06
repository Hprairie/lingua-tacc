import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class TokenStats:
    """Class to hold token count statistics"""

    average: float
    minimum: int
    maximum: int
    total_count: int
    record_count: int
    min_example: Optional[str] = None  # Store text with min tokens
    max_example: Optional[str] = None  # Store text with max tokens


def calculate_token_stats(file_path: str) -> TokenStats:
    """
    Calculate token count statistics from a JSONL file.

    Args:
        file_path (str): Path to the JSONL file

    Returns:
        TokenStats: Object containing token statistics
    """
    total_tokens = 0
    record_count = 0
    min_tokens = float("inf")
    max_tokens = 0
    min_example = None
    max_example = None

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    # Parse each line as JSON
                    record = json.loads(line.strip())

                    # Extract token count from metadata
                    if "metadata" in record and "token_count" in record["metadata"]:
                        token_count = record["metadata"]["token_count"]
                        total_tokens += token_count
                        record_count += 1

                        # Update min tokens
                        if token_count < min_tokens:
                            min_tokens = token_count
                            min_example = record.get("text", "")[:200]  # Store first 200 chars

                        # Update max tokens
                        if token_count > max_tokens:
                            max_tokens = token_count
                            max_example = record.get("text", "")[:200]  # Store first 200 chars

                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON line: {e}")
                    continue
                except KeyError as e:
                    print(f"Missing expected field in JSON: {e}")
                    continue

        if record_count == 0:
            return TokenStats(0, 0, 0, 0, 0)

        average_tokens = total_tokens / record_count
        return TokenStats(
            average=average_tokens,
            minimum=min_tokens,
            maximum=max_tokens,
            total_count=total_tokens,
            record_count=record_count,
            min_example=min_example,
            max_example=max_example,
        )

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return TokenStats(0, 0, 0, 0, 0)
    except Exception as e:
        print(f"An error occurred: {e}")
        return TokenStats(0, 0, 0, 0, 0)


def format_example(text: Optional[str], token_count: int) -> str:
    """Format the example text for display"""
    if not text:
        return "No example available"
    return f"{text[:200]}{'...' if len(text) > 200 else ''}\n[Token count: {token_count}]"


# Example usage
if __name__ == "__main__":
    file_path = "/home1/09753/hprairie/scratch/data-shuffled/fineweb_edu_10bt/fineweb_edu_10bt.val.jsonl"
    stats = calculate_token_stats(file_path)

    if stats.record_count > 0:
        print("\nToken Count Statistics:")
        print(f"{'=' * 50}")
        print(f"Total records processed: {stats.record_count:,}")
        print(f"Average token count: {stats.average:.2f}")
        print(f"Minimum token count: {stats.minimum:,}")
        print(f"Maximum token count: {stats.maximum:,}")
        print(f"\nExample with minimum tokens:")
        print(f"{'-' * 50}")
        print(format_example(stats.min_example, stats.minimum))
        print(f"\nExample with maximum tokens:")
        print(f"{'-' * 50}")
        print(format_example(stats.max_example, stats.maximum))
    else:
        print("No valid records were processed")
