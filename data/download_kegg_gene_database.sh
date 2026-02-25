#!/usr/bin/env bash
# Get the directory of this script and cd into it
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Install gdown if not already installed
if ! command -v gdown &> /dev/null; then
    pip install gdown
fi

TARGET_DIR=kegg_gene_database
FILE_ID=1NNCxQY9nJyQi4Ak87zEektnv6xwlohca

# Error if TARGET_DIR already exists
if [ -d $TARGET_DIR ]; then
    echo "Error: $TARGET_DIR already exists. Please remove it manually."
    exit 1
fi

# Download the file from Google Drive if it doesn't exist
if [ ! -f kegg_gene_database.zip ]; then
    gdown https://drive.google.com/uc?id=$FILE_ID -O kegg_gene_database.zip
fi

# Unzip the file
unzip kegg_gene_database.zip -d ./

# Remove the zip file after extraction
rm kegg_gene_database.zip

echo "Done: contents of $TARGET_DIR are ready."

# Rename the directory
mv kegg_gene_database_full_info_no_genes kegg_gene_database
