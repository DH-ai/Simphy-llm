## Thigns to do
### 1. Store the embeddings model locally, so that it can be reused without reloading - DONE
### 4  Store the embeddings in a vector store, such as Qdrant or FAISS for now qdrant - DOnE 
### 5. Add a method to save the vector store to disk for later use - DONE
### 6. Add a method to load the vector store from disk - DONE
### 2. Add a method to search the vector store with a query and return relevant documents that might be a method for the retrivar class - DONE
### 7. Add a method to update the vector store with new documents - GOOD IDEA
### 3. Add error handling for file loading and embedding processes - UHM DONE 
### 8. Add a method to delete documents from the vector store - NOT NEEDED FOR NOW - can be worked on later
### 9. Need to see the docket and internet version of the qdrant client, for now using in-memory for testing - MAYBE LEts keetp it
### 10. Add a method to clear the vector store - Good idea
### 11. Add a method to modify the metadata of individual chunks - Will work later
### 12. Add command-line argument parsing for PDF path - MUST HAI for later
### 13. Add seperate logic for case when vectorstore is created  - DONE

### 1. understand the usage of both rag_llmshepra.py and rag_newmodel.py
### 2. understand the reason behind the errors, look for smart chunking, 
### 3. read the blogs and guides,
### 4. understand how to run the rag locally 
### 5. use different models to test accuracy, sentence bert, openai embeddings google etc
### 6. Try different vector database 
### 7. Combine embeddings   from different modalities 
### 8. try quantization techniques Use float8 over float32 for 4x storage reduction with <0.3% performance drop.
### 9. Try differnt chunking strategy long with trying 
### 10. Build Multi-retriever Systems: Blend keyword-based and vector search for diverse content types. Augment LLM prompts with retrieved chunks for context-aware answers
### 11. Use perplexity Resources and guides

Need to think about the API endpoints 

Post filtering of rag output can help

chaining implmentation, 

Memory of past sessions 

UI forntend for better testing 

Hosting it somewhere for above point

RAG better with new model now - nuce option too of better pdf extraction 





Physics aware model

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
    
    ## Curate a Golden Dataset

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