At chunksize = 800 overlap =100 the accuracy is bad
At chunksize = 800 overlap =400 the accuracy is too close to 100% like its more of keyword search than symentics

Python version using = 3.10.12


# Things to do, 
1. understand the usage of both rag_llmshepra.py and rag_newmodel.py
2. understand the reason behind the errors, look for smart chunking, 
3. read the blogs and guides,
4. understand how to run the rag locally 
5. use different models to test accuracy, sentence bert, openai embeddings google etc
6. Try different vector database 
7. Combine embeddings   from different modalities 
8. try quantization techniques Use float8 over float32 for 4x storage reduction with <0.3% performance drop.
9. Try differnt chunking strategy long with trying 
10. Build Multi-retriever Systems: Blend keyword-based and vector search for diverse content types. Augment LLM prompts with retrieved chunks for context-aware answers
11. Use perplexity Resources and guides


Need to think about the API endpoints 

Post filtering of rag output can help

Creating a goals.md

Use transformers for using deepseek-ai/coder

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