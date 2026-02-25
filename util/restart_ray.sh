#!/bin/bash

# Restart Ray service
# This script stops any existing Ray instances and starts a new one

set -e

echo "Stopping existing Ray instances..."
ray stop --force 2>/dev/null || true

echo "Waiting for cleanup..."
sleep 2

echo "Starting new Ray instance..."
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265

echo "Ray cluster started successfully!"
echo "Dashboard available at: http://localhost:8265"
