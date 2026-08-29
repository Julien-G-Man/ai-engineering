import os
import itertools
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INDEX_NAME = 'datacamp-index'

vectors = [ 
    {
        'id': "0",
        'values': [],
        'metadata': {'genre': 'productivity', 'year': 2020}
    },
]

class VectorEngine:    
    def __init__(self):
        self.index = self.pc.Index(INDEX_NAME)
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), pool_threads=30)
    
    def create_index(self):
        self.pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric = 'dotproduct', # can also be cosine or euclidean
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

    def check_dimensionality(self, vectors: list[float]) -> bool:
        vector_dims = [len(vector['values']) == 1536 for vector in vectors]
        return all(vector_dims)
    
    def index_stats(self):
        return self.index.describe_index_stats()
    
    def upsert_index(self, vectors, namespace):
        return self.index.upsert(
            vectors=vectors,
            namespace=namespace
        )

    def list_indexes(self):
        return self.pc.list_indexes()

    def delete_index(self, INDEX_NAME: str):
        return self.pc.delete_index(INDEX_NAME)

    def fetch_vectors(self, ids: list[str] | str, namespace: str):
        """Retrive vectors based on their IDs"""
        if isinstance(ids, str):
            ids=[ids]
        return self.index.fetch(
                    ids=ids,
                    namespace=namespace
                )
    
    def get_metadata(self, id: str, namespace: str):
        fetched_vectors = self.fetch_vectors(self.index, namespace)
        return fetched_vectors('vectors')[id]['metadata']

    def query_vector(self, vector: list[float], metadatas: str, top_k: int):
        """Retrieve semantically similar vectors to an input vector"""
        return self.index.query(
            vector=vector,
            filter = metadatas,
            #eg. metadatas =  {
            #     "genre": {"$eq": "documentary"},
            #     "year": 2019
            # },
            top_k=top_k,
            include_metadatas=True,
            include_values=True # include vector embeddings in results
        )
        
    def update_vector(self, id: str, vector: list[float]):
        return self.index.update(
            id=id,
            values=vector,
        )
        
    def update_metadata(self, id: str, metadata: dict[str: dict]):
        return self.index.update(
            id=id,
            set_metadata=metadata
        )
        
    def delete_vector(self, param):
        try:
            if isinstance(param, str | list[str]):
                return self.delete_vector_by_id(param)
            return self.delete_vector_by_metadata(param)
        except ValueError as e:
            raise ValueError(f"Error deleting vector: {e}")
    
    def delete_vector_by_id(self, ids: list[str] | str):
        if isinstance(ids, str):
            ids=[ids]
        return self.index.delete(ids=ids)
    
    def delete_vector_by_metadata(self, metadata: dict[str: dict]):
        return self.index.delete(
            filter=metadata
        )
        
    def delete_vector_from_namespace(self, ids: list[str] | str, namespace: str):
        if isinstance(ids, int):
            ids=[ids]
        return self.index.delete(
            ids=ids,
            namespace=namespace
        )
    
    def delete_all_vectors(self, namespace: str):
        return self.index.delete(
            delete_all=True,
            namespace=namespace
        )
    
    def chunks(iterable, batch_size=100):
        it = iter(iterable)
        chunk = tuple(itertools.islice(it, batch_size))
        while chunk:
            yield chunk
            chunk = tuple(itertools.islice(it, batch_size))
 
    def batch_upsert(self, vectors: list[float]):
        for chunk in self.chunks(vectors):
            self.index.upsert(vectors=chunk)
         
    def parallel_batch(self):
        with self.pc.Index(INDEX_NAME,  pool_threads=30) as index:
            async_results = [index.upsert(vectors=chunk, async_req=True)
                for chunk in self.chunks(vectors, batch_size=100)]
            
            [async_result.get() for async_result in async_results]
                
       

if __name__ == "__main__":
    v_engine = VectorEngine()