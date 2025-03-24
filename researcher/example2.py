#%% 
import mlflow.tracking
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from torchmetrics import Accuracy
from torchvision import datasets
from torchvision.transforms import ToTensor
import optuna

import mlflow



training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


print(f"Image size: {training_data[0][0].shape}")
print(f"Size of training dataset: {len(training_data)}")
print(f"Size of test dataset: {len(test_data)}")




class ImageClassifier(nn.Module):
    def __init__(self, hidden_channels: int = 8):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.model = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 16, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(10),  # 10 classes in total.
        )

    def forward(self, x):
        return self.model(x)

class MLP(nn.Module):
    def __init__(
            self,
            hidden_features: int, 
            out_features: int
    ):
        super().__init__()
        layer1 = nn.LazyLinear(out_features=hidden_features)
        layer2 = nn.Linear(in_features=hidden_features, out_features=out_features)
        self.sequential = nn.Sequential(
            layer1, 
            layer2
        )
        
    def forward(self, x: torch.Tensor):
        x = x.flatten()
        x = self.sequential(x)
        return x


# Get cpu or gpu for training.
device = "cuda" if torch.cuda.is_available() else "cpu"


def train(dataloader, model, loss_fn, metrics_fn, optimizer, epoch):
    """Train the model on a single pass of the dataloader.

    Args:
        dataloader: an instance of `torch.utils.data.DataLoader`, containing the training data.
        model: an instance of `torch.nn.Module`, the model to be trained.
        loss_fn: a callable, the loss function.
        metrics_fn: a callable, the metrics function.
        optimizer: an instance of `torch.optim.Optimizer`, the optimizer used for training.
        epoch: an integer, the current epoch number.
    """
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)
        accuracy = metrics_fn(pred, y)

        # Backpropagation.
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch
            step = batch // 100 * (epoch + 1)
            mlflow.log_metric("loss", f"{loss:2f}", step=step)
            mlflow.log_metric("accuracy", f"{accuracy:2f}", step=step)
            print(f"loss: {loss:2f} accuracy: {accuracy:2f} [{current} / {len(dataloader)}]")


def evaluate(dataloader, model, loss_fn, metrics_fn, epoch):
    """Evaluate the model on a single pass of the dataloader.

    Args:
        dataloader: an instance of `torch.utils.data.DataLoader`, containing the eval data.
        model: an instance of `torch.nn.Module`, the model to be trained.
        loss_fn: a callable, the loss function.
        metrics_fn: a callable, the metrics function.
        epoch: an integer, the current epoch number.
    """
    num_batches = len(dataloader)
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            eval_loss += loss_fn(pred, y).item()
            eval_accuracy += metrics_fn(pred, y)
    eval_loss /= num_batches
    eval_accuracy /= num_batches
    print(eval_loss, eval_accuracy)
    mlflow.log_metric("eval_loss", f"{eval_loss:2f}", step=epoch, synchronous=True)
    mlflow.log_metric("eval_accuracy", f"{eval_accuracy:2f}", step=epoch, synchronous=True)

    print(f"Eval metrics: \nAccuracy: {eval_accuracy:.2f}, Avg loss: {eval_loss:2f} \n")
    return eval_accuracy



mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")


mlflow.set_experiment("mnist-reference-v2")

loss_fn = nn.CrossEntropyLoss()
metric_fn = Accuracy(task="multiclass", num_classes=10).to(device)
epochs = 3

def objective(trial):
    with mlflow.start_run(nested=True) as nested_run:
        lr = trial.suggest_float("lr", 1e-10, 1e10, log=True)

        batch_size = trial.suggest_int("batch_size", 4, 256, log=True)

        train_dataloader = DataLoader(training_data, batch_size=batch_size)
        test_dataloader = DataLoader(test_data, batch_size=batch_size)

        # classifier = trial.suggest_categorical("classifier", ["mlp", "cnn"])
        classifier = "cnn"
        print(f"testing with a {classifier}")

        params = {
            "learning_rate": lr,
            "batch_size": batch_size,
            "classifier" : classifier,
        }
        print("params:  ", params)

        if classifier == "mlp":
            hidden_features = trial.suggest_int("hidden_features", 5, 2048, log=True)
            model_params = {
                "hidden_features":hidden_features,
                "out_features": 10, # 10 output classes
            }
            model = MLP(**model_params)
        else:
            hidden_channels = trial.suggest_int("hidden_channels", 4, 10, log= True)
            model_params = {
                "hidden_channels": hidden_channels,
            }
            model = ImageClassifier(**model_params)

        params.update(model_params)
        mlflow.log_params(params)

        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        best = 0
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, metric_fn, optimizer, epoch=t)
            best = max(best, evaluate(test_dataloader, model, loss_fn, metric_fn, epoch=0))
    return best


with mlflow.start_run(nested=True) as run:
    params = {
        "epochs":epochs,
        "loss_function": loss_fn.__class__.__name__,
        "metric_function": metric_fn.__class__.__name__,
        "optimizer": "SGD",
    }

    # Log training parameters.
    mlflow.log_params(params)
    mlflow.log_metric("mum",1.2)

    # Log model summary.
    with open("cnn_model_summary.txt", "w") as f:
        f.write(str(summary(ImageClassifier())))
    mlflow.log_artifact("cnn_model_summary.txt")
    # mlflow.pytorch.log_model(model, "pre_trained")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=40)

    trial = study.best_trial

    print("Accuracy: {}".format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))

    # Save the trained model to MLflow.
    # mlflow.pytorch.log_model(model, "post_trained")

# %%
# mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
# with mlflow.start_run(f"ef2537e567a640979d0dc792323fa4b5") as run:
#     print(run._data._params)
#     print(run._data.metrics["loss"])
# %%

mlflow_client = mlflow.tracking.MlflowClient("http://127.0.0.1:8080")
# run_data_dict = mlflow_client.get_run("ef2537e567a640979d0dc792323fa4b5").data.to_dictionary()
losses = mlflow_client.get_metric_history(run_id=run.info.run_id, key="loss")

values = list(enumerate(map(lambda metric :metric.value, losses)))
print(values)
# %%
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 6))
plt.plot(*zip(*values))
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.grid(True)
plt.savefig('loss_line_plot.png')
plt.close()

# %%
mlflow_client.log_artifact(run_id=run.info.run_id, local_path="loss_line_plot.png")
# %%