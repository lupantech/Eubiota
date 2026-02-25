#!/usr/bin/env bash
# Get the directory of this script and cd into it
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Install gdown if not already installed
if ! command -v gdown &> /dev/null; then
    pip install gdown
fi

REMOTE_DIR=kegg_disease_database_250906
TARGET_DIR=kegg_disease_database

FILE_ID=1YqwF8nLxaqSq1170paQWoYfgb0tRmdqb

# Error if TARGET_DIR already exists
if [ -d $TARGET_DIR ]; then
    echo "Error: $TARGET_DIR already exists. Please remove it manually."
    exit 1
fi

# Download the file from Google Drive if it doesn't exist
if [ ! -f $REMOTE_DIR.zip ]; then
    gdown https://drive.google.com/uc?id=$FILE_ID -O $REMOTE_DIR.zip
fi

# Unzip the file and create the target directory
unzip $REMOTE_DIR.zip -d ./

# Remove the zip file after extraction
rm $REMOTE_DIR.zip

echo "Done: contents of $TARGET_DIR are ready."

# Rename the directory
mv $REMOTE_DIR $TARGET_DIR
