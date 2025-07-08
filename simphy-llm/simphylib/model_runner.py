from  langchain.chat_models import init_chat_model
from langchain.chat_models.base import generate_from_stream
from langchain import hub

from typing import Union

class ModelRunner:
    """A class to manage the initialization and running of language models."""
    
    def __init__(self, model_name, model_provider, model_kwargs=None):
        """
        Initialize the ModelRunner with a specific model name and optional model arguments.
        
        :param model_name: The name of the language model to run.
        :param model_kwargs: Optional dictionary of keyword arguments for the model.
        """
        self.model_name = model_name
        self.model_provider = model_provider
        self.model_kwargs = model_kwargs or {}
        self.model = self._initialize_model()


    def _initialize_model(self):
        """Initialize the language model using the specified name and arguments."""
        return init_chat_model(self.model_name, model_provider=self.model_provider, **self.model_kwargs)
    
    def generate_content(self, msg:str):
        """
        Generate content using the initialized model.

        :param msg: The input content for the model, string for now for experimenting.
        :return: Generated content from the model.
        """
        return self.model.invoke(msg)

    # def _generate_from_stream(self, content: Union[types.ContentListUnion, types.ContentListUnionDict], config: dict):
    #     """Generate text from the model using a streaming approach."""
    #     return generate_from_stream(self.model, content, config)

    # def generate_content
