#!/bin/bash

# Start the SSH server in the background
# /usr/sbin/sshd -D -e &

source ./set-env.sh

echo "Started Running..."

echo "step 1: setup and fetch repos"
bash ./setup.sh

source env/bin/activate

echo "step 2: dataset fetch"
bash ./script/data_fetch/data-fetch-small.sh

echo "set 3: depth anything"
bash ./script/external_models/run-depth-anything.sh

# echo "step 4: ppd"
# bash ./script/external_models/run-ppd.sh

# Setup vast cli tool
vastai set api-key $CONTAINER_API_KEY # note that CONTAINER_API_KEY gets injected into container by vast.

vastai stop instance $CONTAINER_ID

