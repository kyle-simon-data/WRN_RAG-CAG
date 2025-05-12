import sys
import os

# Add root of project to path (WRN_RAG-CAG)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.scripts.rag_generate_2f import load_rag_components, generate_rag_answer

_tokenizer = _model = _embedding_model = _collection = None

def initialize_rag():
    global _tokenizer, _model, _embedding_model, _collection
    if _model is None:
        _tokenizer, _model, _embedding_model, _collection = load_rag_components()

def get_rag_response(query: str) -> str:
    initialize_rag()
    return generate_rag_answer(query, _tokenizer, _model, _embedding_model, _collection)