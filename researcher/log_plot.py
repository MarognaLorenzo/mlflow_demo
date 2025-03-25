
import matplotlib.pyplot as plt
import mlflow

import sys

import mlflow.tracking
import helpers


assert len(sys.argv) > 1, "Please pass the configuration file to the script."
hparams = helpers.parse_configuration_yaml(sys.argv[1])
print(vars(hparams))


mlflow.tracking.set_tracking_uri(hparams.tracking_uri)

mlflow_client = mlflow.tracking.MlflowClient()
losses = mlflow_client.get_metric_history(run_id=hparams.run_id, key="loss")


values = list(map(lambda metric :metric.value, losses))
print(values)

# Retrieve parameters from the selected run
prev_params = helpers.get_run_params(run_id=hparams.run_id)

fig = plt.figure(figsize=(10, 6))
plt.plot(values)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.grid(True)

epoch_counter = 0
for i, loss in enumerate(losses):
    if loss.step % (len(losses) // int(prev_params["epochs"])) == 0: 
        print(i)
        plt.axvline(x = i, color='red', linestyle='--')
        plt.text(i+1, plt.ylim()[1], f'Epoch {epoch_counter}', rotation=90, verticalalignment='top')
        epoch_counter += 1

plt.savefig('loss_line_plot.png')
plt.close()

mlflow_client.log_artifact(run_id=hparams.run_id, local_path="loss_line_plot.png")
