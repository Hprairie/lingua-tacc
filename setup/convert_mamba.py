import os
import json
import yaml
import torch
import argparse
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Optional

from torch.distributed.checkpoint.format_utils import torch_save_to_dcp


@dataclass
class InitArgs:
    dt_max: float = 0.1
    dt_min: float = 0.001

    dt_init_floor: float = 1e-4

    A_init_min: float = 1
    A_init_max: float = 16


@dataclass
class BaseMambaArgs:

    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8

    state_dim: int = 128
    n_groups: int = 1
    group_rank: Optional[int] = None
    conv_size: Optional[int] = 4

    bias: bool = False
    conv_bias: bool = True

    dt_bias: bool = False
    D_has_head_dim: bool = False
    learnable_init_states: bool = False

    ssm_chunk_size: int = 256

    vocab_size: int = -1

    ffn_dim_multiplier: Optional[float] = 2

    multiple_of: int = 256
    """
    Enforces that the SwiGLU hidden layer size is a multiple
    of large power of 2.
    """

    norm_eps: float = 1e-5

    init_use_depth: bool = False
    init_base_std: Optional[float] = None
    init_std_factor: str = "disabled"

    init_args: InitArgs = field(default_factory=InitArgs)
    seed: int = 42


def clone_repo(repo_path, repo_dir):
    """Clone the Hugging Face repository."""
    repo_url = f"git@hf.co:{repo_path}"
    subprocess.run(["git", "clone", repo_url, repo_dir], check=True)


def load_defaults():
    """Load default values from BaseMambaArgs dataclass."""
    return asdict(BaseMambaArgs())


def create_config_from_json(json_path, yaml_output_path, key_mapping, ssm_mapping, init_mapping):
    """
    Create a YAML config based on JSON config, key mapping, and dataclass defaults.

    Args:
        json_path (str): Path to the JSON config file.
        yaml_output_path (str): Path to save the output YAML file.
        key_mapping (dict): Dictionary mapping YAML keys to JSON keys.
    """
    # Load JSON config if available
    try:
        with open(json_path, "r") as json_file:
            json_config = json.load(json_file)
    except FileNotFoundError:
        json_config = {}

    # Get defaults from BaseMambaArgs
    defaults = load_defaults()

    # Map and merge values from JSON to final config
    final_config = {}
    for yaml_key, json_key in key_mapping.items():
        if json_key in json_config:
            final_config[yaml_key] = json_config[json_key]
        else:
            # Use default if JSON key is missing
            final_config[yaml_key] = defaults.get(yaml_key)

    if "ssm_cfg" in json_config:
        for yaml_key, json_key in ssm_mapping.items():
            if yaml_key == "n_heads":
                if "expand" in json_config["ssm_cfg"]:
                    d_inner = json_config["ssm_cfg"]["expand"] * final_config["dim"]
                else:
                    d_inner = 2 * final_config["dim"]
                if "headdim" in json_config["ssm_cfg"]:
                    final_config[yaml_key] = d_inner // json_config["ssm_cfg"]["headdim"]
                else:
                    final_config[yaml_key] = d_inner // 64
            elif yaml_key == "dt_bias":
                final_config[yaml_key] = json_key
            elif yaml_key == "learnable_init_states":
                final_config[yaml_key] = json_key
            elif json_key in json_config["ssm_cfg"]:
                final_config[yaml_key] = json_config["ssm_cfg"][json_key]
            else:
                # Use default if JSON key is missing
                final_config[yaml_key] = defaults.get(yaml_key)

        for yaml_key, json_key in init_mapping.items():
            if "init_args" not in final_config:
                final_config["init_args"] = {}
            if yaml_key == "A_init_min":
                if "A_init_range" in json_config["ssm_cfg"]:
                    final_config["init_args"][yaml_key] = json_config["ssm_cfg"].get("A_init_range")[0]
                else:
                    final_config["init_args"][yaml_key] = 1
            elif yaml_key == "A_init_max":
                if "A_init_range" in json_config["ssm_cfg"]:
                    final_config["init_args"][yaml_key] = json_config["ssm_cfg"].get("A_init_range")[1]
                else:
                    final_config["init_args"][yaml_key] = 16
            elif json_key in json_config["ssm_cfg"]:
                final_config["init_args"][yaml_key] = json_config["ssm_cfg"][json_key]
            else:
                final_config["init_args"][yaml_key] = defaults["init_args"].get(yaml_key)

    # Handle nested InitArgs separately if present in defaults
    if "init_args" in defaults and "init_args" in final_config:
        final_config["init_args"] = {**defaults["init_args"], **(json_config.get("init_args", {}))}

    # Save the final config as YAML
    final_config = {"model": final_config}
    with open(yaml_output_path, "w") as yaml_out:
        yaml.dump(final_config, yaml_out, default_flow_style=False)


def convert_mamba(state_dict_path, save_path):
    """Transform state_dict keys and save as a new file."""
    state_dict = torch.load(state_dict_path, map_location=torch.device("cpu"))
    new_state_dict = {}

    # Transform keys in the state dictionary
    for key in state_dict.keys():
        newkey = key.replace("backbone.", "")
        newkey = newkey.replace("mixer.", "ssm.")
        newkey = newkey.replace("norm.", "ssm_norm.")
        newkey = newkey.replace("norm_f.", "norm.")
        newkey = newkey.replace("lm_head.", "output.")
        newkey = newkey.replace("embedding.", "tok_embeddings.")
        new_state_dict[newkey] = state_dict[key]

    # Save the transformed state_dict
    torch.save({"model": new_state_dict}, save_path)


def save_to_dcp(file_path, dcp_path):
    """Save the model in a distributed format."""
    torch_save_to_dcp(file_path, dcp_path)


def main(repo_path, save_dir):
    repo_dir = repo_path.split("/")[1]
    repo_dir = os.path.join(save_dir, repo_dir)
    # Clone the repo
    clone_repo(repo_path, repo_dir)

    # Convert config.json to YAML
    os.makedirs(repo_dir, exist_ok=True)
    config_json = os.path.join(repo_dir, "config.json")
    config_yaml = os.path.join(repo_dir, "lingua_config.yaml")

    # Convert and save new config.yaml
    key_mapping = {
        "dim": "d_model",
        "n_layer": "n_layer",
    }
    ssm_mapping = {
        "n_heads": None,
        "ffn_dim_multiplier": "expand",
        "state_dim": "d_state",
        "conv_size": "d_conv",
        "n_groups": "ngroups",
        "dt_bias": True,
        "bias": "bias",
        "conv_bias": "conv_bias",
        "D_has_head_dim": "D_has_hdim",
        "learnable_init_states": False,
        "ssm_chunk_size": "chunk_size",
    }
    init_mapping = {
        "dt_max": "dt_max",
        "dt_min": "dt_min",
        "dt_init_floor": "dt_init_floor",
        "A_init_min": None,
        "A_init_max": None,
    }
    create_config_from_json(config_json, config_yaml, key_mapping, ssm_mapping, init_mapping)

    # Convert the model's state_dict
    model_bin = os.path.join(repo_dir, "pytorch_model.bin")
    new_model_bin = os.path.join(repo_dir, "lingua_pytorch_model.bin")
    convert_mamba(model_bin, new_model_bin)

    # Save the model in distributed format
    dcp_model_bin = os.path.join(repo_dir, "lingua_distributed_pytorch_model")
    save_to_dcp(new_model_bin, dcp_model_bin)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_path",
        type=str,
        required=True,
        help="Hugging Face model repository in the format <username>/<model-name>",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="The directory to save the model weights.",
    )
    args = parser.parse_args()

    main(args.repo_path, args.save_dir)
