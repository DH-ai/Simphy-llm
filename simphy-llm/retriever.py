import embeddings_simphy



class RetrieverSimphy:
    def __init__(self, simphy):
        self.simphy = simphy

    def retrieve(self, query, top_k=5):
        """
        Retrieve the top_k most relevant documents for the given query.
        """
        results = self.simphy.search(query, top_k=top_k)
        return results