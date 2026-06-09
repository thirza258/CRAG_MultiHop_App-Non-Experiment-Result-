def vote_prompt(query: str, chunk: str, response: str) -> str:
    return f"""
    Question: {query}
    Context Chunk: {chunk}
    Proposed Answer: {response}
    
    Evaluate if the Proposed Answer makes sense given the Context Chunk.
    """

def rag_prompt(query: str, context: str) -> str:
    if not context:
        return f"""You are a helpful assistant. You were asked the following question but no relevant information was found in the document.

        Question: {query}

        Please respond with: "I don't have enough information in the provided document to answer this question." Then briefly suggest what kind of information would be needed."""

    return f"""You are a helpful assistant. Answer the question based only on the context below.

        Context:
        {context}

        Question: {query}

        Answer:"""

def prompt_generator(query: str) -> str:
    return f"""
    Answer the question: {query}
    """