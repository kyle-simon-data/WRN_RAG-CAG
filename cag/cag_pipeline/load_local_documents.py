import os
import json
from langchain_community.document_loaders import PyPDFLoader, TextLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_local_documents(download_dir='data/downloads', chunk_size=1000, chunk_overlap=200, return_with_metadata=False):
    """
    Load and chunk documents from the downloads directory.
    
    Args:
        download_dir (str): Directory containing documents to load
        chunk_size (int): Size of each text chunk
        chunk_overlap (int): Overlap between consecutive chunks
        return_with_metadata (bool): If True, return Document objects with metadata
        
    Returns:
        If return_with_metadata=False: List[str] of document chunks
        If return_with_metadata=True: List[Document] with page_content and metadata
    """
    chunked_documents = []
    file_counts = {
        "pdf": 0,
        "txt": 0,
        "json": 0,
        "md": 0,
        "sh": 0,
        "other": 0
    }
    
    # Initialize text splitter for chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    for filename in os.listdir(download_dir):
        file_path = os.path.join(download_dir, filename)
        if not os.path.isfile(file_path):
            continue

        try:
            file_ext = filename.split('.')[-1].lower()
            loaded_docs = []
            
            # Handle different types of files
            if file_ext == 'pdf':
                loader = PyPDFLoader(file_path)
                loaded_docs = loader.load()
                file_counts["pdf"] += 1
                print(f"[INFO] Loaded {len(loaded_docs)} pages from PDF: {filename}")
            
            elif file_ext in ['txt', 'md', 'sh']:
                loader = TextLoader(file_path, encoding='utf-8')
                loaded_docs = loader.load()
                file_counts[file_ext] += 1
                print(f"[INFO] Loaded text file ({file_ext}): {filename}")
            
            elif file_ext == 'json':
                # For JSON files, we need to extract textual content
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # Convert JSON to string for simple JSON files
                if isinstance(json_data, dict) or isinstance(json_data, list):
                    json_text = json.dumps(json_data, indent=2)
                    doc = Document(page_content=json_text, metadata={"source": filename})
                    loaded_docs = [doc]
                    file_counts["json"] += 1
                    print(f"[INFO] Loaded JSON file: {filename}")
            
            else:
                print(f"[INFO] Skipping unsupported file type: {filename}")
                file_counts["other"] += 1
                continue

            # Apply chunking to the loaded documents
            chunks = text_splitter.split_documents(loaded_docs)
            
            # Track original source in metadata
            for i, chunk in enumerate(chunks):
                if not hasattr(chunk, 'metadata'):
                    chunk.metadata = {}
                chunk.metadata['source'] = filename
                chunk.metadata['chunk_index'] = i
                chunk.metadata['type'] = file_ext
                
                chunked_documents.append(chunk)
            
            print(f"[INFO] Created {len(chunks)} chunks from {filename}")

        except Exception as e:
            print(f"[ERROR] Failed to load {filename}: {e}")
                
    print(f"[INFO] Finished loading documents.")
    print(f"[INFO] Files loaded: PDF={file_counts['pdf']}, TXT={file_counts['txt']}, JSON={file_counts['json']}, " 
          f"MD={file_counts['md']}, SH={file_counts['sh']}")
    print(f"[INFO] Total files: {sum(file_counts.values())}, Total chunks: {len(chunked_documents)}")
    
    # Return either Document objects with metadata or just the text content
    if return_with_metadata:
        return chunked_documents
    else:
        return [doc.page_content for doc in chunked_documents]

if __name__ == "__main__":
    docs = load_local_documents()
    #print(f"[DEBUG] Sample doc:\n{docs[0].page_content[:200]}\nMetadata: {docs[0].metadata}" if docs else "[DEBUG] No documents loaded.")
"""
import json
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import os
from langchain_core.documents import Document


#loading cybersecurity docs local directory

def load_local_documents(directory='data/downloads'):
    documents = []
    counts = {
        'pdf': 0,
        'txt': 0,
        'md': 0,
        'sh': 0,
        'json': 0,
        'total_chunks': 0
    }

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if not os.path.isfile(file_path):
            continue

        ext = os.path.splitext(filename)[-1].lower()

        try:
            if ext == '.pdf':
                loader = PyPDFLoader(file_path)
                loaded_docs = loader.load()
                counts['pdf'] += 1
                doc_type = 'pdf'

            elif ext in {'.txt', '.md', '.sh'}:
                loader = TextLoader(file_path, encoding='utf-8')
                loaded_docs = loader.load()
                doc_type = ext.lstrip('.')
                counts[doc_type] += 1

            elif ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                question_id = data.get("id", filename)
                for title, sentences in data.get("context", []):
                    for idx, sentence in enumerate(sentences):
                        documents.append(
                            Document(
                                page_content=sentence.strip(),
                                metadata={
                                    "source": filename,
                                    "type": "json",
                                    "title": title,
                                    "sentence_id": idx,
                                    "question_id": question_id
                                }
                            )
                        )
                        counts['total_chunks'] += 1
                counts['json'] += 1
                continue

            else:
                continue  # skip unsupported file types

            # Wrap non-JSON docs
            for doc in loaded_docs:
                documents.append(
                    Document(
                        page_content=doc.page_content,
                        metadata={
                            "source": filename,
                            "type": doc_type
                        }
                    )
                )
                counts['total_chunks'] += 1

        except Exception as e:
            print(f"[ERROR] Failed to load {filename}: {e}")

    print(f"[INFO] Finished loading documents:")
    print(f"  PDFs loaded:   {counts['pdf']}")
    print(f"  TXTs loaded:   {counts['txt']}")
    print(f"  MDs loaded:    {counts['md']}")
    print(f"  JSONs loaded:  {counts['json']}")
    print(f"  SHs loaded:    {counts['sh']}")
    print(f"  Total chunks:  {counts['total_chunks']}")
    print(f"[INFO] Total Document objects: {len(documents)}")

    return documents

if __name__ == "__main__":
    docs = load_local_documents()
    print(f"[DEBUG] Sample doc:\n{docs[0].page_content[:200]}\nMetadata: {docs[0].metadata}" if docs else "[DEBUG] No documents loaded.")
"""
