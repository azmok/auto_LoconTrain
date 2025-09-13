#!/bin/bash

# This script automates multiple LoCon training runs with incrementing
# convolution rank and alpha values.
#
# Before running:
# 1. Make sure you are in the directory where `train_network.py` is located.
# 2. Update `PATH_CONFIG_JSON` to the correct name of your JSON file.
# 3. Update `PATH_TRAIN_NETWORK` if your `train_network.py` is not in the current directory.

# --- Configuration ---
# Your base JSON config file from the kohya_ss GUI
PATH_CONFIG_JSON="_baseConfig.json"
PATH_CONFIG_TOML=""
# The path to the update script
PATH_UPDATE_SCRIPT="./update_config.py"
# The path to the main training script
PATH_TRAIN_NETWORK="kohya_ss/sd-scripts/sdxl_train_network.py"
# The total number of training runs you want to perform
NUM_RUNS=3

# --- User Input ---
read -p "Enter the starting convolution rank/alpha value: " start_rank_alpha
read -p "Enter the step value for each training run: " step

# Validate user input
if ! [[ "$start_rank_alpha" =~ ^[0-9]+$ ]] || ! [[ "$step" =~ ^[0-9]+$ ]]; then
    echo "Error: Both inputs must be non-negative integers."
    exit 1
fi

echo "Starting automated training loop..."

# --- Training Loop ---
for (( i=0; i<$NUM_RUNS; i++ ))
do
    # Calculate the current rank for this iteration
    current_rank=$((start_rank_alpha + i * step))
    echo "--- Training Run $((i + 1)) of $NUM_RUNS: Rank/Alpha = $current_rank ---"

    # Step 1: Call the Python script to update the JSON config
    # The output of the Python script (the new file path) is captured by `new_config_path`
    new_config_path=$(python3 "$PATH_UPDATE_SCRIPT" "$current_rank" "$PATH_CONFIG_JSON")
    
    if [ -z "$new_config_path" ]; then
        echo "Failed to create new config file. Aborting."
        exit 1
    fi
    echo "Generated new config file: $new_config_path"
    
    # Step 2: Launch the training using the newly created config file
    # We pass the new config file path to the --config_file argument.
    accelerate launch --config_file="$new_config_path" "$PATH_TRAIN_NETWORK"
    
    # Check the exit status of the `accelerate launch` command
    if [ $? -eq 0 ]; then
        echo "Training for rank $current_rank completed successfully."
    else
        echo "Training for rank $current_rank failed. Continuing to the next run."
    fi

done

echo "Automated training loop finished."
