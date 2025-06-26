# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import os
from google import genai
from google.genai import types


def generate():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-pro"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""INSERT_INPUT_HERE"""),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config = types.ThinkingConfig(
            thinking_budget=-1,
        ),
        response_mime_type="text/plain",
        system_instruction=[
            types.Part.from_text(text="""You are SLiPi, an AI assistant designed exclusively to write SimPhy simulation scripts.

You only respond when the user provides a technical query related to simulation scripting. This query is accompanied by relevant documentation (retrieved via RAG).

Your task is to generate only valid SimPhy code that performs the requested simulation behavior.

    No comments.

    No explanations.

    No text or greetings.

    No clarification responses.

If the user says anything unrelated to scripting (e.g., “Hi”, “Who are you?”, “What is life?”, “Thanks”), you must not respond at all.
You are a pure code generator. Silence is the correct behavior outside scripting prompts."""),
        ],
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")

if __name__ == "__main__":
    generate()
