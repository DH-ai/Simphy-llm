## Thigns to do
### 1. Store the embeddings model locally, so that it can be reused without reloading - DONE
### 4  Store the embeddings in a vector store, such as Qdrant or FAISS for now qdrant - DOnE 

### 7. Add a method to update the vector store with new documents - GOOD IDEA
### 3. Add error handling for file loading and embedding processes - UHM DONE 
### 8. Add a method to delete documents from the vector store - NOT NEEDED FOR NOW - can be worked on later
### 9. Need to see the docker and internet version of the qdrant client, for now using in-memory for testing - MAYBE Lets keetp it
### 10. Add a method to clear the vector store - Good idea
### 11. Add a method to modify the metadata of individual chunks - Will work later
### 12. Add command-line argument parsing for PDF path - MUST HAI for later





Need to think about the API endpoints 
8. try quantization techniques Use float8 over float32 for 4x storage reduction with <0.3% performance drop.

UI forntend for better testing 

Hosting it somewhere for above point






### Physics aware model

1. Knowlede injection
    Embed Physics principles
    - Fine-tune/train the model
        * Classical Mehcanics- whatever is possible in simphy
        * API docs + examples seperately
        * Special Curated examples
        * Physics context maybe books with formulas need to think and discuss it with kurmi sir
    - Using curated Datasets like
        ```
        physics_qa = [
            {
                "question": "How to set perfect elastic collisions in SimPhy?",
                "answer": "body.setRestitution(1.0)"
            },
            {
                "question": "Best joint type for pendulum?",
                "answer": "addDistanceJoint(body, anchorPoint, frequency=30)"
            }
        ]
        ```
        
2. Physisc-Aware promt Engineering 
    - Basically, another prompting guide :lol
    - Point is, prompt should be elaborate, lesser the promt more the halucination, can use simplar model run locally then imporves the promt fine tuned with promting to make it more accessible 
    
    - Also Custom checks
        - Pre-generation checks:
            ```            
            def validate_joint_params(params):
                if params["joint_type"] == "distance" and params["frequency"] <= 0:
                    raise ValueError("Frequency must be positive")
            ```
            
3. Output Optimization
    - Post processing; Such that code is exceutable 


4. Simulated-Tested Examples
    Curate a Golden Dataset

    | Scenario | Code Snippet | Physics Key Points |
    |----------|-------------|-------------------|
    | Newton's Cradle | `World.addDistanceJoint(..., freq=30)` | Conservation of momentum |
    | Projectile Motion | `body.applyImpulse(0, 50)` | Parabolic trajectory |
    | Spring-Mass System | `addSpringJoint(..., stiffness=40)` | Hooke's Law (F=-kx) |

5. Feedback Loop
    1. Run generated code in simphy
    2. Detect Physics errors
    3. Fine-tune Based on failures ( LoRA/QLoRA)
    4. Validation
        - PyBullet/Mujoco for automated physics checks




A query rewriting model (e.g., self-ask, REACT, or just GPT-style prompt)

A loop (manual or rule-based) that:

- tracks current context

- decides when to stop querying

“query → rag → refine → better query → better rag → final result”

Batching queries, to send the llm if its still processing one request need to think about he dynamics here  

Imediate Thoughts 
0. arg parser
1. improvinng data by looking at metadata and reducing the tokens and using layoutpdfreader instead of pymupdf, writting json parser, extending decoder and encoder, -> splitter, parsed data -> stored locally
2. using hybrid search + my custom searching function + chunk coupling of near by chunks -> retriever
3. pooling or file system handling to change the behaviour if i change system instructions, need to think at system level -> gemini request
4. Running the Vectorstore as a server, for quick retrieval, creating api for it
5. TheBloke/phi-2-GGUF for symentic tagging 
6. GOal - document = page_content + Metadata(keypharserr frome rake algorithm) + Symentic tagging -> caching 
7. Also chaining implmentatino ? another lib file Needed, storing prev chats locally, encrypter?
8. improving rag, small llm fine tuned and iterative rag, might also do post filteringof finalized rag
9. implmenting loading bars, pretty printing, using rich
look into - https://www.instill-ai.dev/docs/artifact/upload-filesls



Can it be such that the query instead directly going to rag, the llm decides if it wants to go for the api, might use it for 