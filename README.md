# MLflow Demo

A demonstration repository showcasing MLflow, an open-source platform for managing the machine learning lifecycle.

## Structure
The repository is organized as follows:

```
mlflow_demo/
├── README.md          # Main documentation
├── serverFolder/      # MLflow server configuration
└── researcher/        # Example code for the demonstration
    ├── example1.py    # Single-run introductory training example on Fashion Mnist
    ├── example2.py    # Hierchical run with hyperparameter tuning(Optuna) training example on Fashion Mnist
    ├── helpers.py     # Helpers function for argument parsing and parameters retrieval
    ├── log_plot.py    # Python script for generating a plot of loss function for a given run
    └── cfg/           # Configuration files
```

## Overview

MLflow is a platform that helps to track experiments, package code into reproducible runs, and share and deploy models. It offers four main components:

1. **MLflow Tracking**: Records and queries experiments (parameters, metrics, code versions, etc.)
2. **MLflow Projects**: Packages ML code in a reusable and reproducible format
3. **MLflow Models**: Deploys ML models in diverse serving environments
4. **MLflow Registry**: Centrally manages models in a Model Registry

### Tracking Capabilities

MLflow Tracking is a powerful component that logs and tracks:

- **Parameters**: Hyperparameters and configuration settings used in training
- **Metrics**: Training and validation metrics over time (loss, accuracy, etc.)
- **Artifacts**: Files associated with runs (models, plots, data files)
- **Code Version**: Git commit hash and repo status
- **Tags**: Custom annotations for runs
- **Environment**: Python environment and dependencies

The tracking server provides a web UI to:
- Compare experiments visually
- Search and filter runs
- Organize experiments into hierarchies
- Export results for analysis

More info on [how to run a server](./serverFolder/README.md).

### Experiments and Runs
An experiment in MLflow is like a container or project folder that groups related machine learning runs together. Think of it as a highest-level organizational unit that might represent, for example an "Image Classification Model"

Runs are individual execution instances within an experiment. Each run represents a single training attempt. 

Each run automatically tracks:
- code version
- timestamp
- run status (running/completed/failed)

```py
import mlflow

# Start an experiment
mlflow.set_experiment("Image classification")

# Create a run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("batch_size", 32)
    
    # Train your model...

    # Log metrics
    mlflow.log_metric("accuracy", 0.85)
    
    # Save artifacts...
```

Every experiment and run will be assigned a unique ID. You can then use this ID to retrieve further information later from a specific run.