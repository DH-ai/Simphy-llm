try:
    from simphylib.config import DEFAULT_LLMSHERPAURL
    from simphylib.docker_runner import DockerRunner
except ImportError:
    from config import DEFAULT_LLMSHERPAURL
    from docker_runner import DockerRunner
    
from langchain_community.document_loaders.llmsherpa import LLMSherpaFileLoader
from langchain_core.documents import Document
from typing import List, Optional
import json



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

        docs = super().load()

        return Parser(docs).parse()
    

    



if __name__ == "__main__":
    docker_runner = DockerRunner()
    loader = SimphyFileLoader(
        file_path="simphy-llm/docs/SimpScriptGPart4Ch4.pdf",
        new_indent_parser=True,
        apply_ocr=False,
        strategy="sections",
        llmsherpa_api_url=DEFAULT_LLMSHERPAURL,
    
    )
    docs = loader.load()
    html = ""
    for i, doc in enumerate(iterable=docs, start=0):
        # print("\n"+"-"*40 + f"{i}")
        html = html + str(doc.page_content)
        print(html)
    # with open("index.json","w") as f:
    #     json.dump(docs, f, indent=4)
        