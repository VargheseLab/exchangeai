
# Installation
Either you clone this repository and run it with the local server, or you can use the docker image.

## Docker (recommended)

### Windows/Linux/Mac (GUI)
0. [Install Docker Desktop](https://docs.docker.com/desktop/).
1. Open the Docker Desktop Dashboard
2. Navigate to the Images view in the left-hand navigation menu
3. Use the search bar at the top to search `ghcr.io/vargheselab/exchangeai/full`
4. Select Pull to download the image
5. Once the image is pulled, select the Run button
6. Expand the Optional settings
7. In the Host port field, specify the port on your machine `(8000)`
8. Select Run to start your container 
9. Open http://127.0.0.1:8000/ with your Browser

### Linux/Mac (command line)
0. [Install Docker Engine](https://docs.docker.com/engine/install).
1. Open a terminal and use following commands:
```
    docker pull ghcr.io/vargheselab/exchangeai/full:latest
    docker run -it --rm -p 8000:8000 ghcr.io/vargheselab/exchangeai/full:latest
```
2. Open http://127.0.0.1:8000/ with your Browser


### Configuration:

#### Lite version
Just pull the lite version `exchangeai/lite` which does not support finetuning, but its package size is smaller and supports all other major functions.

#### Sampling rate
You can change the default unified sampling rate via the environment variable SAMPLING_RATE. For for Docker Desktop, you find these under the Optional settings.
```
docker run -it --rm -p 8080:8000 -e SAMPLING_RATE=500 ...
```

#### GPU Acceleration (Nvidia only)
If you have a Nvidia GPU you can accelerate finetuning. Please install 
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and use the following command to start the container:

    docker run -it --rm -p 8000:8000 --gpus all ghcr.io/vargheselab/exchangeai/full:latest

#### Windows installation
If you are using windows, the container should work out of the Box. If you are using a Nvidia GPU as well, you have to install the toolkit inside the WSL2. Please follow this guide: [Install CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)


## Python
1. Install [redis-server](https://redis.io/docs/latest/operate/oss_and_stack/install/install-redis/)
2. git clone https://github.com/VargheseLab/exchangeai.git
3. cd exchange/app
4. pip install -r requirements/requirements.txt
5. pip install -r requirements/optional-requirements.txt
6. sh startup.sh
7. Open http://127.0.0.1:8000/ with your Browser