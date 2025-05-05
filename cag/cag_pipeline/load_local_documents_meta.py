import os
import json
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

def load_local_documents(hotpotqa_dir='data/hotpotqa/testdocs'):
    documents = []
    pdf_count = 0
    txt_count = 0
    json_count = 0
    chunk_count = 0

    # Load PDFs and TXTs
    for filename in os.listdir(hotpotqa_dir):
        file_path = os.path.join(hotpotqa_dir, filename)
        if not os.path.isfile(file_path):
            continue

        try:
            if filename.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                loaded_docs = loader.load()
                pdf_count += 1
                print(f"[INFO] Loaded {len(loaded_docs)} pages from PDF: {filename}")
            elif filename.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
                loaded_docs = loader.load()
                txt_count += 1
                print(f"[INFO] Loaded 1 text file: {filename}")
            else:
                continue

            for doc in loaded_docs:
                documents.append(
                    Document(
                        page_content=doc.page_content,
                        metadata={"source": filename, "type": "txt" if filename.endswith('.txt') else "pdf"}
                    )
                )

        except Exception as e:
            print(f"[ERROR] Failed to load {filename}: {e}")

    # Load HotpotQA JSON files
    for filename in os.listdir(hotpotqa_dir):
        if not filename.endswith('.json'):
            continue

        file_path = os.path.join(hotpotqa_dir, filename)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            question_id = data.get("id", filename)

            for title, sentences in data.get("context", []):
                for idx, sentence in enumerate(sentences):
                    documents.append(
                        Document(
                            page_content=sentence.strip(),
                            metadata={
                                "source": filename,
                                "type": "hotpotqa",
                                "question_id": question_id,
                                "title": title,
                                "sentence_id": idx
                            }
                        )
                    )
                    chunk_count += 1

            json_count += 1

        except Exception as e:
            print(f"[ERROR] Failed to load {filename}: {e}")

    print(f"[INFO] Finished loading documents.")
    print(f"[INFO] PDFs loaded: {pdf_count}, TXTs loaded: {txt_count}, JSON files loaded: {json_count}")
    print(f"[INFO] Total HotpotQA chunks: {chunk_count}")
    print(f"[INFO] Total documents returned: {len(documents)}")

    return documents

if __name__ == "__main__":
    docs = load_local_documents()
    print(f"[DEBUG] Loaded {len(docs)} documents.")
    if docs:
        print(f"[DEBUG] Sample doc:\n{docs[0].page_content}\nMetadata: {docs[0].metadata}")
