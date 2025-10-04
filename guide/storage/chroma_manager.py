import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class ChromaManager:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_db")  # FIX: Persistent storage
        self.collection = None
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def store_chunks(self, chunks: List[Dict[str, Any]]):
        try:
            self.client.delete_collection("code_chunks")
        except:
            pass
        
        self.collection = self.client.create_collection("code_chunks")
        
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            searchable_text = f"Type: {chunk['type']}\nName: {chunk['name']}\nCode:\n{chunk['code']}"
            embedding = self.embedding_model.encode([searchable_text])[0].tolist()
            
            ids.append(chunk['id'])
            embeddings.append(embedding)
            documents.append(searchable_text)
            metadatas.append({
                'type': chunk['type'],
                'name': chunk['name'],
                'filename': chunk['filename'],
                'line_start': chunk['line_start'],
                'line_end': chunk['line_end'],
                'code': chunk['code']  # FIX: Store full code in metadata
            })
        
        self.collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.collection:
            return []
        
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['metadatas', 'distances']
        )
        
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return formatted_results