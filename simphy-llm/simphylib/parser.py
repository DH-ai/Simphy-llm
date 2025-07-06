try:
    from simphylib.config import DEFAULT_LLMSHERPAURL, LLMSHERPA_CONTAINER_NAME
    from simphylib.docker_runner import DockerRunner
except Exception as e:
    from config import DEFAULT_LLMSHERPAURL, LLMSHERPA_CONTAINER_NAME
    from docker_runner import DockerRunner
    
from langchain_community.document_loaders.llmsherpa import LLMSherpaFileLoader
from langchain_core.documents import Document
from typing import List, Optional
import json
from urllib3.exceptions import NewConnectionError, MaxRetryError
import logging
logger = logging.getLogger(__name__)

class Parser:
    "A custom parsing class for modularity and extensibility."
    def __init__(self,document:List[Document]):
        self.document = document

    def parse(self) -> List[Document]:
        # Implement your parsing logic here
        return self.document
 


class SimphyFileLoader(LLMSherpaFileLoader):
    """
    A class to load PDF files using LLMSherpaFileLoader with specific configurations.
    Inherits from LLMSherpaFileLoader.
    """

    def __init__(self, file_path, new_indent_parser=False, apply_ocr=False, strategy="sections", llmsherpa_api_url=DEFAULT_LLMSHERPAURL):
        super().__init__(file_path=file_path, new_indent_parser=new_indent_parser, apply_ocr=apply_ocr, strategy=strategy, llmsherpa_api_url=llmsherpa_api_url)

    ### error handling if api not available using docer class
    def load(self):
        """
        Load the PDF file and return the documents.
        """
        try:
            # Attempt to load the file using LLMSherpaFileLoader
            
            docs = super().load()
        except MaxRetryError as e:
            # If an error occurs, log it and use the DockerRunner to load the file


            # logger.error(f"MaxRetryError: {e}. Attempting to reload Docker")
            docker_runner = DockerRunner()
            # print(f"{docker_runner.run_container_at_start} status, container status; {docker_runner.container_status(LLMSHERPA_CONTAINER_NAME)}")
            docs = super().load()
        except NewConnectionError as e:
            

            raise ConnectionError(f"Failed to connect to LLMSherpa API at {DEFAULT_LLMSHERPAURL}. Please check your LLMSherpa docker Container status or the API URL.") from e
        except Exception as e:
            

            logger.error(f"An unexpected error occurred: {e}. Attempting to reload Docker")
            docker_runner = DockerRunner(start_clean=True)
            docs = super().load()


        return Parser(docs).parse()
    

    



if __name__ == "__main__":
    # dockerrunner = DockerRunner(start_clean=False, run_container_at_start=False)

    # dockerrunner.container_status(LLMSHERPA_CONTAINER_NAME)
    # dockerrunner.remove_all_containers()
    
    
    
    loader = SimphyFileLoader(
        file_path="simphy-llm/docs/SimpScriptGPart4Ch4.pdf",
        new_indent_parser=True,
        apply_ocr=False,
        strategy="sections",
        llmsherpa_api_url=DEFAULT_LLMSHERPAURL,
    
    )
    docs = loader.load()
    # html = ""
    # for i, doc in enumerate(iterable=docs, start=0):
    #     # print("\n"+"-"*40 + f"{i}")
    #     html = html + str(doc.page_content)
    #     print(html)
    # with open("index.json","w") as f:
    #     json.dump(docs, f, indent=4)
        