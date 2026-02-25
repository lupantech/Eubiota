#!/bin/bash

# Read CUDA_VISIBLE_DEVICES from config.yaml
if [ -f "trainer/train_scientist/config.yaml" ]; then
  CUDA_DEVICES=$(grep -m1 "CUDA_VISIBLE_DEVICES:" trainer/train_scientist/config.yaml | sed "s/.*CUDA_VISIBLE_DEVICES: *['\"]\\([^'\"]*\\)['\"].*/\\1/")
  if [ -n "$CUDA_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
    echo "Setting CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES from config.yaml"
  fi
fi

# Pub IP
PUBLIC_IP=$(curl -s ifconfig.me)

if [ -z "$PUBLIC_IP" ]; then
  echo "No valid IP get!"
  exit 1
fi

LAST_TWO_OCTETS=$(echo "$PUBLIC_IP" | awk -F'.' '{print $3"."$4}')

# --- Configuration Section ---
# 1. Define the log directory
LOG_DIR="./task_logs/${PUBLIC_IP}/scientist_serve_log"

# 2. Define the prefix for output files
LOG_PREFIX="scientist_serving_output_"

# 3. Define the maximum size of a single log file (1MB)
LOG_SIZE='1M'

# 4. Define the maximum number of log files to keep (mimics backupCount behavior)
MAX_LOG_FILES=5000

# 5. The Python command you want to run (enclosed in quotes)
PYTHON_COMMAND=".venv/bin/python trainer/train_scientist/rollout_scientist.py"

# --- Function: calculate digit length of a number ---
suffix_length() {
    # This is a more concise way to get the length of a number in bash
    echo "${#1}"
}

# Remove and recreate the log directory
rm -rf $LOG_DIR
mkdir -p $LOG_DIR

# Calculate suffix digit length: we need to represent up to (MAX_LOG_FILES - 1)
SUFFIX_DIGITS=$(suffix_length $((MAX_LOG_FILES - 1)))

echo "Starting the Scientist rollout task... Log files will use $SUFFIX_DIGITS-digit suffixes (000... to $((MAX_LOG_FILES - 1)))"

# Use split with dynamic suffix length
PYTHONUNBUFFERED=1 $PYTHON_COMMAND 2>&1 | \
    split -b "$LOG_SIZE" -d -a "$SUFFIX_DIGITS" - "$LOG_DIR/$LOG_PREFIX"

SPLIT_EXIT_CODE=${PIPESTATUS[1]}

if [ $SPLIT_EXIT_CODE -eq 0 ]; then
    echo "Task completed successfully."
else
    echo "Error: The task or log splitting failed with exit code $SPLIT_EXIT_CODE."
    exit $SPLIT_EXIT_CODE
fi

# Clean up: keep only the newest MAX_LOG_FILES files (by modification time)
echo "Cleaning up old log files, keeping the latest $MAX_LOG_FILES..."
ls -1t "$LOG_DIR"/"$LOG_PREFIX"* 2>/dev/null | \
    tail -n +$((MAX_LOG_FILES + 1)) | \
    xargs rm -f

echo "Log files are saved in: $LOG_DIR"
