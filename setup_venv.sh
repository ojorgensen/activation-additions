#!/bin/bash

# Name of the virtual environment
VENV_NAME="myenv"

# Check if the virtual environment already exists
if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment..."
    python -m venv $VENV_NAME
else
    echo "Virtual environment '$VENV_NAME' already exists."
fi

echo "Activating virtual environment..."
source $VENV_NAME/bin/activate

# Check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found!"
    deactivate
    exit 1
fi

echo "All done!"
