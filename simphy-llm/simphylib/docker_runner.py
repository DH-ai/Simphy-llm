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
    def __init__(self, start_clean = False, image_name=LLMSHERPA_IMAGE, container_name=LLMSHERPA_CONTAINER_NAME, run_container_at_start=True):
        # self.client = docker.from_env()
        self.image_name = image_name
        self.container_name = container_name
        self.run_container_at_start = run_container_at_start
        if start_clean:
            self.remove_all_containers()
        
        try:
            self.client = docker.from_env()
        except docker.errors.DockerException as e:
            logger.error(f"Error connecting to Docker: {e}")
            logger.error(DOCKER_ERROR_MESSAGE)
            exit(1)
        if self.ensure_image_exists(self.image_name) and self.run_container_at_start:
            try: 
                # print(f"1 Running the container at start default: {run_container_at_start}")
                # If run_container_at_start is True, run the container
                if run_container_at_start:
                    self.run_container(
                        image=self.image_name,
                        ports={'5000/tcp': 5001},
                        name=self.container_name
                    )
            except docker.errors.APIError as e:
                logger.error(f"Error running container: {e}")
                logger.error(DOCKER_ERROR_MESSAGE)
                

    def run_container(self, image, ports=None, name=None):
        """
            Run a Docker container with the specified image and ports.
        """
        try:
            # Check if the container already exists
            container_list = self.client.containers.list(all=True, filters={"name": name})  
            
            if container_list  is not None and len(container_list) > 0:
                if container_list[0].name == name:
                    logger.warning(f"Container with name {name} already exists. Removing it first.")
                    # self.remove_container(name)
                    container_list[0].restart()
                    

                # if name in str([container.name for container in container_list]):
                #     logger.warning(f"Container with name {name} already exists. Removing it first.")
                #     self.remove_container(name)
                
            container = self.client.containers.run(
                image, detach=True, ports=ports, name=name
            )
            logger.info(f"Container {container.name} is running with ID {container.id}.")
        except docker.errors.APIError as e:
            logger.error(f"Error running container: {e}")

    def remove_all_containers(self):
        """
        Remove all containers.
        """
        try:
            for container in self.client.containers.list(all=True):
                logger.warning(f"Removing container {container.name} with ID {container.id}.")
                container.remove(force=True)
            logger.info("All containers have been removed.")
        except docker.errors.APIError as e:
            logger.error(f"Error removing containers: {e}")

    def remove_container(self, container_name):
        """
        Remove a specific container by name.
        """
        try:
            logger.info(f"Attempting to remove container {container_name}.")
            container = self.client.containers.get(container_name)
            logger.warning(f"Removing container {container.name} with ID {container.id}.")
            container.remove(force=True)
            logger.info(f"Container {container_name} has been removed.")
        except docker.errors.NotFound:
            logger.warning(f"Container {container_name} not found.")
        except docker.errors.APIError as e:
            logger.error(f"Error removing container: {e}")


    def ensure_image_exists(self, image):
        """
        Ensure the specified Docker image exists, pulling it if necessary.
        """
        try:   
            img = self.client.images.list()

            if image not in [image.tags[0] for image in img if image.tags]:
                logger.warning(f"Image {image} not found. Pulling the image...")
                self.client.images.pull(image)
                logger.info(f"Image {image} has been pulled successfully.")
            else:
                logger.info(f"Image {image} already exists.")
            return True
        except docker.errors.DockerException as e:
            logger.error(f"Error listing images: {e}")
            logger.error(DOCKER_ERROR_MESSAGE)
            return False
        
    def container_status(self, container_name):
        """
        Check the status of a specific container.
        """
        try:
            container = self.client.containers.get(container_name)
            return container.status
        except docker.errors.NotFound:
            logger.warning(f"Container {container_name} not found.")
            return None
        except docker.errors.APIError as e:
            logger.error(f"Error checking container status: {e}")
            return None

if __name__ == "__main__":
    docker_runner = DockerRunner(start_clean=False, run_container_at_start=False)
    # Example usage
    # docker_runner.remove_all_containers()
    print(f"Container status: {docker_runner.container_status(LLMSHERPA_CONTAINER_NAME)}")