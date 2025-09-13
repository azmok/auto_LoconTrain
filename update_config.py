import json
import os
import sys

def main(rank_value, path_config_json):
    """
    Modifies a kohya_ss JSON config file with new conv_dim and conv_alpha values.
    
    Args:
        rank_value (int): The new value for conv_dim and conv_alpha.
        path_config_json (str): The path to the base JSON configuration file.
    
    Returns:
        str: The path to the newly created, modified JSON file.
    """
    try:
        # Load the base configuration from the provided file
        with open(path_config_json, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Fix: Add the missing 'distributed_type' key required by accelerate.
        # "NO" is used for single-machine, non-distributed training.
        config['distributed_type'] = "NO"

        # Update the convolution rank and alpha values
        config['conv_dim'] = rank_value
        config['conv_alpha'] = rank_value
        
        # get values for new filename
        newtwork_dim = config['network_dim']
        network_alpha = config['network_alpha']
        schedular = config['lr_scheduler'][:5]
        optimizer = config['optimizer']
        epochs = config['epoch']
        lr = f"{config['learning_rate']:.1e}".replace("e-0", "e-").replace("e+0", "e+")

        
        
        # We will dynamically replace "conv20,al20" with the new value.
        new_filename = f"locon,d{newtwork_dim},al{network_alpha},conv{rank_value},conval{rank_value},epo{epochs},{schedular},{optimizer},{lr}"
        
        config['output_name'] = new_filename

        # Update the output and logging directories to be more descriptive
        # We assume the user runs the script from a location where these paths are valid.
        output_parent_dir = os.path.dirname(config['output_dir'])
        output_dir = os.path.join(output_parent_dir, new_filename)
        
        new_logging_dir = os.path.join(output_dir, "log")
        config['logging_dir'] = new_logging_dir
        
        # Create a new file name for the updated config
        new_config_path = os.path.join(output_dir, f"{new_filename}.json")

        # Ensure the new output and logging directories exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(new_logging_dir, exist_ok=True)

        # Save the modified configuration to the new file
        with open(new_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
            
        return new_config_path

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        return ""

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python3 update_config.py <rank_value> <path_config_json>", file=sys.stderr)
        sys.exit(1)
    
    # Get command-line arguments and convert rank to integer
    try:
        rank_value = int(sys.argv[1])
        path_config_json = sys.argv[2]
    except (ValueError, IndexError):
        print("Invalid arguments. <rank_value> must be an integer.", file=sys.stderr)
        sys.exit(1)

    new_config_path = main(rank_value, path_config_json)
    if new_config_path:
        print(new_config_path)
