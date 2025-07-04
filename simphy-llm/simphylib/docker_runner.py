# File: simphy-llm/simphylib/docker_runner.py
# This is for linux only 
# and requires docker to be installed and running.
# It also requires the user to be added to the docker group.
# You can do this by running the following command:
# sudo usermod -aG docker $USER

import docker
import docker.errors

import docker.models.containers

# Pull and run a container (example: Ubuntu, echo hello)
try:
    client = docker.from_env()
except docker.errors.DockerException as e:
    print(f"Error connecting to Docker: {e}")
    print("Make sure Docker is running and you have the correct permissions.")
    print("Run following command to add your user to the docker group:")
    print("sudo usermod -aG docker $USER")
    exit(1)
LLMSHEPRA_IMAGE = "ghcr.io/nlmatics/nlm-ingestor:latest"
try:
    img = client.images.list()
    if LLMSHEPRA_IMAGE not in [image.tags[0] for image in img if image.tags]:
        print(f"Image {LLMSHEPRA_IMAGE} not found. Pulling the image...")
        client.images.pull(LLMSHEPRA_IMAGE)
    else:
        print(f"Image {LLMSHEPRA_IMAGE} already exists.")

    container = client.containers.run(
        LLMSHEPRA_IMAGE, detach=True, ports={'5000/tcp': 5001}, name="llmsherpa_container"
        )
    print(f"Container {container.name} is running with ID {container.id}.")
    # print("You can access the LLMSherpa API at http://localhost:8000")
    # print("To stop the container, run: docker stop", container.id)
except docker.errors.DockerException as e:
    print(f"Error listing images: {e}")
    if isinstance(container, docker.models.containers.Container):
        print(f"Stopping container {container.name} due to error.")
        client.api.kill(container)
    print(f"Container {container.name} has been stopped.")
    exit(1)
    