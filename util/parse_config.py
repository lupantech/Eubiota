import yaml
import sys
import argparse
from typing import List, Any, Tuple

def get_values_from_yaml(config_path: str, keys: List[str]) -> Tuple[Any, ...]:
    """
    Parses a YAML file and returns the values for a list of specified keys.
    It first searches in the 'python_args' section, then falls back to 'env'.

    Args:
        config_path (str): The path to the YAML configuration file.
        keys (List[str]): A list of keys to retrieve values for.

    Returns:
        Tuple[Any, ...]: A tuple of the values corresponding to the keys.
                         None is used for keys that are not found.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at '{config_path}'", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}", file=sys.stderr)
        sys.exit(1)

    values = []
    for key in keys:
        value = None
        try:
            # First, try to get the value from the 'python_args' section.
            value = config['python_args'].get(key)
            if value is not None:
                values.append(value)
                continue
        except KeyError:
            # If 'python_args' section doesn't exist, ignore and proceed to the next step.
            pass

        try:
            # If not found in 'python_args', try to get the value from the 'env' section.
            value = config['env'].get(key)
            if value is not None:
                values.append(value)
                continue
        except KeyError:
            # If 'env' section doesn't exist, ignore.
            pass

        # If the key was not found in either section, append None.
        if value is None:
            print(f"Warning: Key '{key}' not found in either 'python_args' or 'env' section.", file=sys.stderr)
            values.append(None)
    
    return tuple(values)

# The `main` function remains unchanged as it handles command-line parsing and calls `get_values_from_yaml`.
def main():
    """
    Main function to handle command-line arguments and run the parser.
    """
    parser = argparse.ArgumentParser(
        description="Retrieve values for specified keys from a YAML config file."
    )
    parser.add_argument(
        '-c', '--config', 
        type=str, 
        default='config.yaml',
        help="Path to the YAML configuration file. Defaults to 'config.yaml'."
    )
    parser.add_argument(
        'keys', 
        nargs='+', 
        help="A space-separated list of keys to retrieve from the YAML file."
    )
    
    args = parser.parse_args()
    
    result_tuple = get_values_from_yaml(args.config, args.keys)
    
    # Print the resulting tuple to standard output
    print(result_tuple)

if __name__ == "__main__":
    main()