# MLflow Server Setup and Usage Guide

## Overview
This guide explains how to set up and use a local MLflow tracking server.

## Starting the Server
To start the MLflow tracking server, use the following command:

```bash
mlflow server \
    --host 127.0.0.1 \
    --port 8080
```

## Connecting to the Server
Set the following environment variable to connect to the server:
```bash
export MLFLOW_TRACKING_URI="http://localhost:8080"
```
alternatively, it is possible to call 
```python
mlflow.set_tracking_uri("http://localhost:8080")
```
inside python source code, and it will override the environment variable.

## Basic Usage
1. Access the MLflow UI: Open your browser and navigate to `http://localhost:8080`
2. View experiments, runs, and metrics through the web interface

## Remote Server
It will also be possible to log your mlflow data on a VM on the departemental cluster, which will facilitate team working and will allow for extra storage. You will need the departemental VPN to get access to it. 

IP address (future changes will be public here):
172.23.32.96

Port (future changes will be public here):
8080

```bash
# Replace with your virtual machine's IP and port
export MLFLOW_TRACKING_URI="http://vm-ip:port"
```

Or in Python:
```python
mlflow.set_tracking_uri("http://vm-ip:port")
```

## Additional Resources
- ‼️[Mlflow quickstart](https://www.mlflow.org/docs/latest/getting-started/intro-quickstart)
- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
- [MLflow GitHub Repository](https://github.com/mlflow/mlflow)