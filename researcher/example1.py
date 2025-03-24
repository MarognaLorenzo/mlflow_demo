import matplotlib.pyplot as plt
import mlflow.tracking
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from torchmetrics import Accuracy
from torchvision import datasets
from torchvision.transforms import ToTensor

import mlflow

import sys
import helpers

assert len(sys.argv) > 1, "Please pass the configuration file to the script."
hparams = helpers.parse_configuration_yaml(sys.argv[1])


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


train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


class ImageClassifier(nn.Module):
    def __init__(self, hidden_channels: int = 8):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels, 16, kernel_size=3),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(10),  # 10 classes in total.
        )

    def forward(self, x):
        for layer in [self.conv1, self.conv2, self.head]:
            # if x.isnan().any():
            #     breakpoint()
            x = layer(x)
        return x


# Get cpu or gpu for training.
device = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"


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
    mlflow.log_metric("eval_loss", f"{eval_loss:2f}", step=epoch)
    mlflow.log_metric("eval_accuracy", f"{eval_accuracy:2f}", step=epoch)

    print(f"Eval metrics: \nAccuracy: {eval_accuracy:.2f}, Avg loss: {eval_loss:2f} \n")


loss_fn = nn.CrossEntropyLoss()
metric_fn = Accuracy(task="multiclass", num_classes=10).to(device)
model = ImageClassifier().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
batch_size = 64


mlflow.set_tracking_uri(uri=hparams.tracking_uri)


mlflow.set_experiment("mnist-reference-v2")

with mlflow.start_run(run_name=hparams.run_name) as run:
    params = vars(hparams)
    params.update({
        "loss_function": loss_fn.__class__.__name__,
        "metric_function": metric_fn.__class__.__name__,
        "optimizer": "SGD",
    })

    # Log training parameters.
    mlflow.log_params(params)

    # Log model summary.
    with open("model_summary.txt", "w") as f:
        f.write(str(summary(model)))
    mlflow.log_artifact("model_summary.txt")


    # mlflow.pytorch.log_model(model, "pre_trained")

    for t in range(hparams.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, metric_fn, optimizer, epoch=t)
        evaluate(test_dataloader, model, loss_fn, metric_fn, epoch=0)

    # Save the trained model to MLflow.
    mlflow.pytorch.log_model(model, "final_model")

# %%
mlflow_client = mlflow.tracking.MlflowClient(hparams.tracking_uri)
losses = mlflow_client.get_metric_history(run_id=run.info.run_id, key="loss")
values = list(enumerate(map(lambda metric: metric.value, losses)))
print(values)
# %%

fig = plt.figure(figsize=(10, 6))
plt.plot(*zip(*values))
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.grid(True)
plt.savefig('loss_line_plot.png')
plt.close()

# %%
mlflow_client.log_artifact(run_id=run.info.run_id,
                           local_path="loss_line_plot.png")
# %%
