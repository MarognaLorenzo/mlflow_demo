from mlflow.server import get_app_client
import sys
import os
import warnings

try:
    tracking_uri = os.environ['MLFLOW_TRACKING_URI']
except KeyError:
    warnings.warn("MLFLOW_TRACKING_URI environment variable not set. Tracking URI is set to localhost:8080")
    tracking_uri = "http://localhost:8080/"

auth_client = get_app_client("basic-auth", tracking_uri=tracking_uri)

auth_client.create_experiment_permission(
    experiment_id=sys.argv[1], username=sys.argv[2], permission=sys.argv[3]
)