#!/usr/bin/env bash
# Get the directory of this script and cd into it
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Install gdown if not already installed
if ! command -v gdown &> /dev/null; then
    pip install gdown
fi

TARGET_DIR=micro_genotypes
FILE_ID=1N5JU73eRcQUxeKTdrtUr1kOw4kA-DkqZ

# Download the file from Google Drive
gdown https://drive.google.com/uc?id=$FILE_ID -O fna_parsed.zip

# Unzip the file
unzip fna_parsed.zip -d $TARGET_DIR

# Remove the zip file after extraction
rm fna_parsed.zip

echo "Done: contents of $TARGET_DIR are ready."
