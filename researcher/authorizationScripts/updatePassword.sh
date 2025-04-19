#!/bin/bash

echo
echo $# arguments provided
echo

# Check if both arguments are provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 <username> <old-password> <new-password>"
    exit 1
fi

python_name="python3.11"
echo "Python version is set as: "$python_name, modify this script if necessary
# Store arguments in variables
username="$1"
old_password="$2"
new_password="$3"

export MLFLOW_TRACKING_USERNAME=$username
export MLFLOW_TRACKING_PASSWORD=$old_password

# Run the Python script with the arguments
echo
$python_name core/_updatePassword.py "$new_password"

if [ $? -eq 0 ]; then
    echo "✅Password updated successfully!✅"
    export MLFLOW_TRACKING_PASSWORD=$new_password
    echo ""
    echo "⚠️  ⚠️  ⚠️  IMPORTANT REMINDER ⚠️  ⚠️  ⚠️"
    echo "=====================================>"
    echo "Update your environment variables with:"
    echo "-------------------------------------"
    echo "export MLFLOW_TRACKING_USERNAME=$username"
    echo "export MLFLOW_TRACKING_PASSWORD=$new_password"
    echo "=====================================>"
else
    echo "Failed to update password"
    exit $?
fi

exit 0