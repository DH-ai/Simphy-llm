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