# Tensorflow Dev Template

## Prerequisites
- Python 3.8.x or newer
- Python venv
- VSCode
- Dev Containers VSCode extension
- docker

## Preparation

### Fetch required resources
Create a virtual enviroment:
```
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```

## CPU or GPU container
Configure the container with CPU or GPU:
```
./config-container.sh cpu
```
Or
```
./config-container.sh gpu
```
If it asks for sudo rights run:
```
chmod +x config-container.sh
```
And rerun the config script.

## Nvidia Compatibility:
The GPU version works with:
 - RTX 3060
 - Cuda version: 12.3
 - Driver version: 545.23.06

You can fetch this from here:
```
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda_12.3.0_545.23.06_linux.run
```