sudo apt-get update
sudo apt install python3-pip
pip install mlflow
export PATH=$PATH:/home/ubuntu/.local/bin
mkdir mlflow-server
cd mlflow-server
screen -S server
mlflow server --host 0.0.0.0 --port 8080

