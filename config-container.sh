#!/bin/bash
if [ "$1" == "cpu" ]; then
  cp ./dockerization/config-files/cpu/devcontainer.json ./.devcontainer/devcontainer.json
elif [ "$1" == "gpu" ]; then
  cp ./dockerization/config-files/gpu/devcontainer.json ./.devcontainer/devcontainer.json
else
  echo "Usage: $0 [cpu|gpu]"
fi