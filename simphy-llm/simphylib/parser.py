try:
    import simphylib.config 
except:
    import config 
from langchain_community.document_loaders.llmsherpa import LLMSherpaFileLoader

class Parser:

    pass
DEFAULT_LLMSHERPAURL = "http://172.17.0.2:5001/api/parseDocument?renderFormat=all&useNewIndentParser=true"


loader = LLMSherpaFileLoader(
    file_path="simphy-llm/docs/SimpScriptGPart4Ch4.pdf",
    new_indent_parser=True,
    apply_ocr=False,
    strategy="chunks",
    llmsherpa_api_url=DEFAULT_LLMSHERPAURL,
)

if __name__ == "__main__":
    docs = loader.load()

    for i, doc in enumerate(iterable=docs, start=0):
        print("\n"+"-"*40 + f"{i}")
        print(doc.page_content)