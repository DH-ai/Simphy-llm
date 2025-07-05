# File: simphy-llm/simphylib/docker_runner.py
# This is for linux only 
# and requires docker to be installed and running.
# It also requires the user to be added to the docker group.
# You can do this by running the following command:
# sudo usermod -aG docker $USER

import docker
import docker.errors
import logging

try:
    from simphylib.config import LLMSHERPA_IMAGE, LLMSHERPA_CONTAINER_NAME, DOCKER_ERROR_MESSAGE
except ImportError:
    from config import LLMSHERPA_IMAGE, LLMSHERPA_CONTAINER_NAME, DOCKER_ERROR_MESSAGE
logger = logging.getLogger(__name__)


class DockerRunner:
    """A class to manage Docker containers for LLMSherpaFileLoader.
    It ensures the required image exists, runs the container, and provides methods to remove containers.
    """
    def __init__(self, start_clean = False, image_name=LLMSHERPA_IMAGE, container_name=LLMSHERPA_CONTAINER_NAME):
        # self.client = docker.from_env()
        self.image_name = image_name
        self.container_name = container_name
        if start_clean:
            self.remove_all_containers()
        
        try:
            self.client = docker.from_env()
        except docker.errors.DockerException as e:
            print(f"Error connecting to Docker: {e}")
            print(DOCKER_ERROR_MESSAGE)
            exit(1)
        if self.ensure_image_exists(self.image_name):
            try:
                self.run_container(
                    image=self.image_name,
                    ports={'5000/tcp': 5001},
                    name=self.container_name
                )
            except docker.errors.APIError as e:
                print(f"Error running container: {e}")
                print(DOCKER_ERROR_MESSAGE)
                

    def run_container(self, image, ports=None, name=None):
        """
            Run a Docker container with the specified image and ports.
        """
        try:
            # Check if the container already exists
            if name in self.client.containers.list(all=True, filters={"name": name}):
                print(f"Container with name {name} already exists. Removing it first.")
                self.remove_container(name)
                
            container = self.client.containers.run(
                image, detach=True, ports=ports, name=name
            )
            print(f"Container {container.name} is running with ID {container.id}.")
        except docker.errors.APIError as e:
            print(f"Error running container: {e}")
    
    def remove_all_containers(self):
        """
        Remove all containers.
        """
        try:
            for container in self.client.containers.list(all=True):
                print(f"Removing container {container.name} with ID {container.id}.")
                container.remove(force=True)
            print("All containers have been removed.")
        except docker.errors.APIError as e:
            print(f"Error removing containers: {e}")
    
    def remove_container(self, container_name):
        """
        Remove a specific container by name.
        """
        try:
            container = self.client.containers.get(container_name)
            print(f"Removing container {container.name} with ID {container.id}.")
            container.remove(force=True)
            print(f"Container {container_name} has been removed.")
        except docker.errors.NotFound:
            print(f"Container {container_name} not found.")
        except docker.errors.APIError as e:
            print(f"Error removing container: {e}")
    
    
    def ensure_image_exists(self, image):
        """
        Ensure the specified Docker image exists, pulling it if necessary.
        """
        try:   
            img = self.client.images.list()

            if image not in [image.tags[0] for image in img if image.tags]:
                print(f"Image {image} not found. Pulling the image...")
                self.client.images.pull(image)
                print(f"Image {image} has been pulled successfully.")
            else:
                print(f"Image {image} already exists.")
            return True
        except docker.errors.DockerException as e:
            print(f"Error listing images: {e}") 
            print(DOCKER_ERROR_MESSAGE)
            return False
        
    def container_status(self, container_name):
        """
        Check the status of a specific container.
        """
        try:
            container = self.client.containers.get(container_name)
            return container.status
        except docker.errors.NotFound:
            print(f"Container {container_name} not found.")
            return None
        except docker.errors.APIError as e:
            print(f"Error checking container status: {e}")
            return None
