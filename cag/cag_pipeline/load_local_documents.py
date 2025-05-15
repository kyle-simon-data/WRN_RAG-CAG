import os
import json
from langchain_community.document_loaders import PyPDFLoader, TextLoader
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