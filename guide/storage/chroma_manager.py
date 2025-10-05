import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class ChromaManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.client = chromadb.PersistentClient(path="./chroma_db")
            cls._instance.collection = None
            cls._instance.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        return cls._instance
    
    def store_chunks(self, chunks: List[Dict[str, Any]]):
        """Store chunks in ChromaDB with detailed logging"""
        
        print(f"\n=== CHROMADB STORAGE DEBUG ===")
        print(f"Storing {len(chunks)} chunks")
        
        try:
            self.client.delete_collection("code_chunks")
            print("Deleted existing collection")
        except:
            print("No existing collection to delete")
        
        self.collection = self.client.create_collection("code_chunks")
        print("Created new collection")
        
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            searchable_text = f"Type: {chunk['type']}\nName: {chunk['name']}\nCode:\n{chunk['code']}"
            embedding = self.embedding_model.encode([searchable_text])[0].tolist()
            
            chunk_id = chunk['id']
            ids.append(chunk_id)
            embeddings.append(embedding)
            documents.append(searchable_text)
            metadatas.append({
                'type': chunk['type'],
                'name': chunk['name'],
                'filename': chunk['filename'],
                'line_start': chunk['line_start'],
                'line_end': chunk['line_end'],
                'code': chunk['code']
            })
            
            # Log first 3 chunks
            if i < 3:
                print(f"  Chunk {i}: ID={chunk_id}, type={chunk['type']}, name={chunk['name']}")
        
        # Store in ChromaDB
        self.collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        print(f"Successfully stored {len(ids)} chunks in ChromaDB")
        
        # VERIFY: Query back to confirm IDs match
        stored_data = self.collection.get(limit=3, include=['metadatas'])
        print(f"\nVERIFICATION - First 3 stored IDs:")
        for stored_id in stored_data['ids'][:3]:
            print(f"  Stored ID: {stored_id}")
        print("=== END STORAGE DEBUG ===\n")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search ChromaDB with detailed logging"""
        
        print(f"\n=== CHROMADB SEARCH DEBUG ===")
        print(f"Query: {query}")
        
        if not self.collection:
            print("ERROR: No collection exists!")
            return []
        
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['metadatas', 'distances']
        )
        
        print(f"ChromaDB returned {len(results['ids'][0])} results")
        
        formatted_results = []
        for i in range(len(results['ids'][0])):
            result_id = results['ids'][0][i]
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            
            formatted_results.append({
                'id': result_id,
                'metadata': metadata,
                'distance': distance
            })
            
            # Log first 3 results
            if i < 3:
                print(f"  Result {i}: ID={result_id}, name={metadata.get('name')}, distance={distance}")
        
        print("=== END SEARCH DEBUG ===\n")
        return formatted_results
