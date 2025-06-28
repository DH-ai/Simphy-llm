# Simphy Linguistic Programmatic Interface (SLiPI)

SliPI is a Large Language model Built to help in scripting and making simulations in [Simphy](https://simphy.com/)

### Rage Pipiline might not use at all

user_input → text_preprocessor.py → retriever.py (LlamaIndex + FAISS)
    → top_k_contexts → prompt_builder.py
        → llama3_generate.py (LLM inference)
            → postprocess_output.py → script_validator.py → output_simphy_script



