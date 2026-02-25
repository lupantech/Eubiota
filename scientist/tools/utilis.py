import os
import json
import numpy as np
import re

def print_json(result):
    print(json.dumps(result, indent=4))

def save_result(result, output_file_name, output_dir):
    safe_query = re.sub(r'[^a-zA-Z0-9]', '_', output_file_name)
    safe_query = safe_query[:150]
    
    os.makedirs(output_dir, exist_ok=True)
    output_file_name = os.path.join(output_dir, f"result_{safe_query}.json")

    with open(output_file_name, "w") as f:
        json.dump(result, f, indent=4)

    print(f"Save the result to {output_file_name}")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))   

def remove_duplicates_nested(data):
    """
    Remove duplicate elements from lists in a nested data structure.
    Handles dictionaries, lists, and nested combinations while preserving order.
    
    Args:
        data: The data structure (dict, list, or primitive) to process
        
    Returns:
        The data structure with duplicates removed from all lists
    """
    if isinstance(data, dict):
        # Recursively process all values in the dictionary
        return {key: remove_duplicates_nested(value) for key, value in data.items()}
    elif isinstance(data, list):
        # Remove duplicates from the list while preserving order
        seen = set()
        unique_list = []
        for item in data:
            # Recursively process each item
            processed_item = remove_duplicates_nested(item)
            # For hashable items, check if we've seen them
            try:
                if processed_item not in seen:
                    seen.add(processed_item)
                    unique_list.append(processed_item)
            except TypeError:
                # If item is unhashable (like dict or list), just append it
                # We could convert to JSON string for comparison, but that might be expensive
                unique_list.append(processed_item)
        return unique_list
    else:
        # Return primitive types as-is
        return data

def remove_keys_nested(data, keys_to_remove):
    """
    Recursively remove specified keys from nested dictionaries and lists.
    
    Args:
        data: Dictionary, list, or other data structure
        keys_to_remove: List of keys to remove from dictionaries
        
    Returns:
        The data structure with specified keys removed at all nesting levels
    """
    if isinstance(data, dict):
        # Remove specified keys and recursively process remaining values
        return {
            key: remove_keys_nested(value, keys_to_remove)
            for key, value in data.items()
            if key not in keys_to_remove
        }
    elif isinstance(data, list):
        # Recursively process each element in the list
        return [remove_keys_nested(item, keys_to_remove) for item in data]
    else:
        # Return other types as-is
        return data
