#!/bin/bash
set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate the project's venv (and ensure it can run pip-installed CLIs like gdown)
VENV_DIR="$SCRIPT_DIR/../.venv"
VENV_PY="$VENV_DIR/bin/python"

if [ ! -x "$VENV_PY" ]; then
  echo "Error: expected venv python at: $VENV_PY"
  echo "Create the venv first (see project setup instructions), then re-run."
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Ensure pip exists inside the venv (some envs may be created without pip)
"$VENV_PY" -m ensurepip --upgrade >/dev/null 2>&1 || true

# Ensure gdown is available for the download scripts (they call `gdown ...`)
if ! command -v gdown >/dev/null 2>&1; then
  "$VENV_PY" -m pip install --upgrade pip >/dev/null
  "$VENV_PY" -m pip install gdown >/dev/null
  hash -r
fi

echo "=========================================="
echo "Downloading ALL Databases"
echo "=========================================="

# KEGG Databases
echo "Downloading KEGG databases..."
bash "$SCRIPT_DIR/download_kegg_organism_database.sh"
bash "$SCRIPT_DIR/download_kegg_drug_database.sh"
bash "$SCRIPT_DIR/download_kegg_disease_database.sh"
bash "$SCRIPT_DIR/download_kegg_gene_database.sh"

# MDIPID Databases
echo "Downloading MDIPID databases..."
bash "$SCRIPT_DIR/download_mdipid_disease_database.sh"
bash "$SCRIPT_DIR/download_mdipid_gene_database.sh"
bash "$SCRIPT_DIR/download_mdipid_microbiota_database.sh"

# Genotype Data
echo "Downloading genotype data..."
bash "$SCRIPT_DIR/download_genotype_data.sh"

echo "=========================================="
echo "All databases downloaded successfully!"
echo "=========================================="
