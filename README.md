# MVP-DEPTH-ESTIMATION-WITH-REFINEMENT


### Docker container setup

First, make sure you have a docker hub account and have docker cli installed.

Then, login into docker in the cli using `sudo docker login --u <username>`

Build a image using `sudo docker build -t <docker_username>/<image_name>:latest .` in the project root directory (don't forget the dot at the end!).

Verify the image is built and on your system using `sudo docker images`

Push image to docker hub using `sudo docker push <docker_username>/<image_name>:latest`


