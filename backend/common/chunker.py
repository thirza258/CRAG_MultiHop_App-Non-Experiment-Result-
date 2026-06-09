import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from typing import List, Literal
from sklearn.metrics.pairwise import cosine_similarity

class DocumentChunker:
    def __init__(self, 
                 strategy: Literal["fixed", "paragraph", "semantic", "recursive"] = "recursive", 
                 chunk_size: int = 500, 
                 overlap: int = 50,
                 embedding_client=None):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.client = embedding_client

    def chunk(self, text: str) -> List[str]:
        print(f"Chunking document with strategy '{self.strategy}'...")
        if self.strategy == "fixed":
            return self._chunk_fixed(text)
        elif self.strategy == "recursive":
            return self.recursive(text)
        elif self.strategy == "paragraph":
            return self._chunk_paragraph(text)
        elif self.strategy == "semantic":
            return self._chunk_semantic(text)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _chunk_fixed(self, text: str) -> List[str]:
        chunks =[]
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += (self.chunk_size - self.overlap)
        
        return chunks

    def _chunk_paragraph(self, text: str) -> List[str]:
        # FIX 1: Fall back to single newlines if double newlines don't exist
        delimiter = '\n\n' if '\n\n' in text else '\n'
        raw_paras = text.split(delimiter)
        raw_paras =[p.strip() for p in raw_paras if p.strip()]
        
        chunks =[]
        current_chunk = ""

        for para in raw_paras:
            if len(para) > self.chunk_size:
                # Flush existing chunk
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # Split the massive paragraph
                sub_chunks = self._chunk_fixed(para)
                
                # Add all sub-chunks except the last one directly to chunks
                chunks.extend(sub_chunks[:-1])
                # Keep the last piece as the new current_chunk to merge with upcoming text
                current_chunk = sub_chunks[-1] if sub_chunks else ""
            
            # Normal logic: merge until it hits chunk size
            elif len(current_chunk) + len(para) + len(delimiter) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para
            else:
                if current_chunk:
                    current_chunk += delimiter + para
                else:
                    current_chunk = para
        
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    def recursive(self, text: str) -> List[str]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,          
            chunk_overlap=self.overlap,     
            length_function=len,   
            is_separator_regex=False,
            separators=[
                "\n\n",  
                "\n",     
                ". ",     
                " ",      
                ""        
            ]
        )

        chunks = text_splitter.split_text(text)
        return chunks

    def _chunk_semantic(self, text: str) -> List[str]:
        if not self.client:
            raise ValueError("Semantic chunking requires an embedding_client.")

        sentences = re.split(r'(?<=[.?!])\s+', text)
        sentences =[s for s in sentences if s.strip()]
        
        if not sentences:
            return[]

        try:
            embeddings_resp = self.client.embeddings.create(
                input=sentences,
                model="text-embedding-3-small"
            )
            vecs =[d.embedding for d in embeddings_resp.data]
        except Exception as e:
            print(f"Embedding failed: {e}")
            return sentences

        distances =[]
        for i in range(len(vecs) - 1):
            v1 = np.asarray(vecs[i]).reshape(1, -1)
            v2 = np.asarray(vecs[i + 1]).reshape(1, -1)
            sim = cosine_similarity(v1, v2)[0][0]
            distances.append(sim)

       
        threshold = 0.75 
        chunks =[]
        current_group = [sentences[0]]
        
        current_tokens = len(sentences[0]) // 4

        for i, dist in enumerate(distances):
            next_sentence = sentences[i+1]
            next_tokens = len(next_sentence) // 4
            
            if dist > threshold and (current_tokens + next_tokens) < self.chunk_size:
                current_group.append(next_sentence)
                current_tokens += next_tokens
            else:
                chunks.append(" ".join(current_group))
                current_group =[next_sentence]
                current_tokens = next_tokens

        if current_group:
            chunks.append(" ".join(current_group))

        return chunks