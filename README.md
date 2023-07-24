# MLOps Project


Please do not fork this repository, but use this repository as a template for your MLOps project. Make Pull Requests to your own repository even if you work alone and mark the checkboxes with an x, if you are done with a topic in the pull request message.

## Project for today
The task for today you can find in the [project-description.md](project-description.md) file.

## Setup

### GCP-deployment

#### Prometheus
```bash
### Fast-API-Webservice:
#### Locally:
cd webservice

docker buildx build --no-cache --platform linux/amd64 --push -t europe-west3-docker.pkg.dev/<project id>/docker-registry/ml-service:latest .

#### On VM:
sudo docker --config /home/<user-name>/.docker pull europe-west3-docker.pkg.dev/<project id>/docker-registry/ml-service:latest

sudo docker run -d -p 8080:8080 --name=webservice europe-west3-docker.pkg.dev/<project id>/docker-registry/ml-service

### Prometheus
#### Locally:
cd deployment/prometheus_deployment

docker buildx build --no-cache --platform linux/amd64 --push -t europe-west3-docker.pkg.dev/<project id>/docker-registry/prometheus:latest .

#### On VM:
sudo docker --config /home/<user-name>/.docker pull europe-west3-docker.pkg.dev/<project_id>/docker-registry/prometheus:latest

sudo docker network create monitoring

sudo docker run -d -p 9090:9090 --name=prometheus --network=monitoring europe-west3-docker.pkg.dev/<project id>/docker-registry/prometheus

### Grafana
#### Locally:
cd grafana/grafana_deployment

docker buildx build --no-cache --platform linux/amd64 --push -t europe-west3-docker.pkg.dev/<project id>/docker-registry/grafana:latest .

#### On VM:
sudo docker --config /home/<user-name>/.docker pull europe-west3-docker.pkg.dev/<project_id>/docker-registry/grafana:latest

sudo docker run -d -p 3000:3000 --name=grafana --network=monitoring europe-west3-docker.pkg.dev/<project id>/docker-registry/grafana

### Evidently
#### Locally:
docker buildx build --no-cache --platform linux/amd64 --push -t europe-west3-docker.pkg.dev/<project id>/docker-registry/evidently_service:latest .

#### On VM Instance:
sudo docker --config /home/<user-name>/.docker pull europe-west3-docker.pkg.dev/positive-sector-383614/docker-registry/evidently_service:latest

sudo docker run -d -p 8085:8085 --name=evidently_service --network=monitoring europe-west3-docker.pkg.dev/<project id>/docker-registry/evidently_service
```

### Environment
```bash
pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```
