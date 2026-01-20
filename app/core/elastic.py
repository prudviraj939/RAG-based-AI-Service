from elasticsearch import Elasticsearch

INDEX = "documents"

mapping = {
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "metadata": {"type": "object"},
            "embedding": {
                "type": "dense_vector",
                "dims": 1536,
                "index": True,
                "similarity": "cosine"
            }
        }
    }
}

_es_client = None


def get_es_client():
    """Get or create Elasticsearch client."""
    global _es_client
    if _es_client is None:
        _es_client = Elasticsearch("http://localhost:9200")
        
        # Initialize index if it doesn't exist
        if not _es_client.indices.exists(index=INDEX):
            _es_client.indices.create(index=INDEX, body=mapping)
    
    return _es_client