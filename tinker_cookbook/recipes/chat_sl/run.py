"""TOML config runner for chat_sl training."""
import argparse
import tomllib

from tinker_cookbook.recipes.chat_sl.train import CLIConfig, cli_main


def main():
    parser = argparse.ArgumentParser(description="Run chat_sl training with TOML config")
    parser.add_argument("--config", required=True, help="Path to TOML config file")
    args, remaining = parser.parse_known_args()

    # Load TOML config
    with open(args.config, "rb") as f:
        toml_config: dict = tomllib.load(f)

    # Parse CLI overrides (key=value format)
    cli_overrides: dict = {}
    for arg in remaining:
        if "=" in arg:
            key, value = arg.split("=", 1)
            cli_overrides[key] = value

    # Merge: CLI overrides take precedence
    final_config = {**toml_config, **cli_overrides}

    # Print final configuration
    print("=" * 60)
    print(f"Configuration {args.config}")
    print("=" * 60)
    for key, value in final_config.items():
        # Mark overridden values
        suffix = " (CLI override)" if key in cli_overrides and key in toml_config else ""
        # Truncate long values for display
        value_str = str(value)
        if "\n" in value_str:
            value_str = value_str.replace("\n", "\\n")
        if len(value_str) > 60:
            value_str = value_str[:57] + "..."
        print(f"  {key}: {value_str}{suffix}")
    print("=" * 60)
    print()

    # Directly instantiate CLIConfig and run training
    cli_config = CLIConfig(**final_config)
    cli_main(cli_config)


if __name__ == "__main__":
    main()

