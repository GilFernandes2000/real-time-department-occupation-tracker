#!/usr/bin/env bash

# This script is used to setup the environment for the RTDOT project.
# To be run from the root directory.

# Setup B.A.T.M.A.N.
cd batman_installation
./create_batman_interface.sh $1 $2

# Start Detection
cd ../RSAProj
if [[ -e Jetson_Master.py ]]; then
    echo "Found Jetson_Dashboard.py"
    echo "Starting detection..."
    python3 Jetson_Master.py
elif [[ -e Jetson_Worker.py ]]; then
    echo "Found Jetson_Worker.py"
    echo "Starting detection..."
    python3 Jetson_Worker_Down_Up.py
else
    echo "Jetson_Master.py and Jetson_Worker.py not found."
    echo "ERROR: Cannot start detection."
fi