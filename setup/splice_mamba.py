import torch
import argparse
from typing import Dict
from omegaconf import OmegaConf
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InitArgs:
    dt_max: float = 0.1
    dt_min: float = 0.001
    dt_init_floor: float = 1e-4
    A_init_min: float = 1
    A_init_max: float = 16


@dataclass
class ModelConfig:
    dim: int
    n_layer: int
    n_heads: int
    state_dim: int
    n_groups: int = 1
    conv_size: Optional[int] = None
    bias: bool = False
    conv_bias: bool = True
    dt_bias: bool = True
    D_has_head_dim: bool = False
    learnable_init_states: bool = False
    ssm_chunk_size: int = 256
    ffn_dim_multiplier: Optional[float] = None
    init_args: InitArgs = field(default_factory=InitArgs)


def modify_state_dict(
    state_dict: Dict[str, torch.Tensor],
    target_n_groups: int,
    config: ModelConfig,
) -> Dict[str, torch.Tensor]:
    """
    Modify a Mamba model state dict to increase n_groups from 1 to target_n_groups.
    Only works with models that originally had n_groups=1.

    Args:
        state_dict: Original model state dict
        target_n_groups: Target number of groups to expand to
        config: Model configuration

    Returns:
        Modified state dict with expanded groups
    """
    if target_n_groups < 1:
        raise ValueError("target_n_groups must be >= 1")

    if target_n_groups == 1:
        return state_dict

    if config.n_groups != 1:
        raise ValueError("This script only supports modifying models with n_groups=1")

    new_state_dict = {}

    # Calculate intermediate dimensions based on config
    hidden_dim = config.dim
    state_dim = config.state_dim
    n_heads = config.n_heads

    for key, tensor in state_dict.items():
        if "in_proj.weight" in key:
            # Input projection format is [z, x, B, C, dt] where:
            # z: hidden_dim
            # x: hidden_dim
            # B: n_groups * state_dim
            # C: n_groups * state_dim
            # dt: n_heads

            # Verify dimensions match expected size
            out_features, in_features = tensor.shape
            expected_out_features = 2 * hidden_dim + 2 * state_dim + n_heads  # For n_groups=1
            if out_features != expected_out_features:
                raise ValueError(
                    f"Weight matrix size {out_features} doesn't match expected size {expected_out_features} "
                    f"for hidden_dim={hidden_dim}, state_dim={state_dim}, n_heads={n_heads}"
                )

            # Split the weight matrix into its components
            z_end = hidden_dim
            x_end = z_end + hidden_dim
            B_end = x_end + state_dim
            C_end = B_end + state_dim
            dt_end = C_end + n_heads

            z_weights = tensor[:z_end]
            x_weights = tensor[z_end:x_end]
            B_weights = tensor[x_end:B_end]
            C_weights = tensor[B_end:C_end]
            dt_weights = tensor[C_end:dt_end]

            # Create new expanded weights
            expanded_B = B_weights.repeat(target_n_groups, 1)
            expanded_C = C_weights.repeat(target_n_groups, 1)

            # Concatenate all parts
            new_weights = torch.cat([z_weights, x_weights, expanded_B, expanded_C, dt_weights], dim=0)

            new_state_dict[key] = new_weights
        else:
            # Keep other weights unchanged
            new_state_dict[key] = tensor

    return new_state_dict


def load_config(config_path: str) -> ModelConfig:
    """Load and validate the model configuration."""
    # Load raw config
    raw_config = OmegaConf.load(config_path)

    # Extract model section
    if "model" not in raw_config:
        raise ValueError("Config must contain a 'model' section")

    model_config = raw_config.model

    # Handle init_args separately
    init_args = InitArgs(**model_config.get("init_args", {}))
    model_config_dict = OmegaConf.to_container(model_config, resolve=True)
    model_config_dict["init_args"] = init_args

    # Convert to structured config
    config = ModelConfig(**model_config_dict)

    return config


def main():
    parser = argparse.ArgumentParser(description="Modify Mamba model state dict to increase n_groups")
    parser.add_argument("input_path", type=str, help="Path to input state dict file")
    parser.add_argument("output_path", type=str, help="Path to save modified state dict")
    parser.add_argument("config_path", type=str, help="Path to model config YAML file")
    parser.add_argument("target_n_groups", type=int, help="Target number of groups")

    args = parser.parse_args()

    # Load and validate config
    print(f"Loading config from {args.config_path}")
    config = load_config(args.config_path)

    # Load state dict
    print(f"Loading state dict from {args.input_path}")
    state_dict = torch.load(args.input_path)

    # Modify state dict
    print(f"Modifying state dict to have {args.target_n_groups} groups")
    new_state_dict = modify_state_dict(state_dict, args.target_n_groups, config)

    # Save modified state dict
    print(f"Saving modified state dict to {args.output_path}")
    torch.save(new_state_dict, args.output_path)
    print("Done!")


if __name__ == "__main__":
    main()
