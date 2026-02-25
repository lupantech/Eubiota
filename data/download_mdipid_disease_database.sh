#!/usr/bin/env bash
# Get the directory of this script and cd into it
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Install gdown if not already installed
if ! command -v gdown &> /dev/null; then
    pip install gdown
fi

# https://drive.google.com/file/d/1G7fVCFhyGhBm51rEtHrfR7-OiwSriZiP/view?usp=drive_link
TARGET_DIR=mdipid_disease_database
FILE_ID=1G7fVCFhyGhBm51rEtHrfR7-OiwSriZiP

# Error if TARGET_DIR already exists
if [ -d $TARGET_DIR ]; then
    echo "Error: $TARGET_DIR already exists. Please remove it manually."
    exit 1
fi

# Download the file from Google Drive if it doesn't exist
if [ ! -f mdipid_disease_database.zip ]; then
    gdown https://drive.google.com/uc?id=$FILE_ID -O mdipid_disease_database.zip
fi

# Unzip the file
unzip mdipid_disease_database.zip -d ./

# Remove the zip file after extraction
rm mdipid_disease_database.zip

echo "Done: contents of mdipid_disease_database are ready."
