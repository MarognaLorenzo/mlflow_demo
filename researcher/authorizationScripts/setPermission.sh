#!/bin/bash

echo
echo $# arguments provided
echo

# Check if both arguments are provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 <experiment-id> <target-username> <permission>"
    exit 1
fi

# Check if environment variables are set
if [ -z "$MLFLOW_TRACKING_USERNAME" ]; then
    echo "Error: MLFLOW_TRACKING_USERNAME environment variable is not set"
    exit 1
fi

if [ -z "$MLFLOW_TRACKING_PASSWORD" ]; then
    echo "Error: MLFLOW_TRACKING_PASSWORD environment variable is not set"
    exit 1
fi


python_name="python3.11"
echo "Python version is set as: "$python_name, modify this script if necessary
# Store arguments in variables
experiment_id="$1"
target_username="$2"
permission="$3"

# Define allowed permissions
allowed_permissions=("READ" "EDIT" "MANAGE" "NO_PERMISSION")

# Check if permission is in the allowed list
if [[ ! " ${allowed_permissions[@]} " =~ " ${permission} " ]]; then
    echo "Error: Invalid permission. Allowed values are: ${allowed_permissions[*]}"
    exit 1
fi

# Run the Python script with the arguments
echo
$python_name core/_setPermission.py "$experiment_id" "$target_username" "$permission"

if [ $? -eq 0 ]; then
    echo "✅Permission set successfully!✅"
else
    echo "Failed to set permission"
    exit $?
fi

exit 0