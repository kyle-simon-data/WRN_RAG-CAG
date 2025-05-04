import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_local_documents(download_dir='data/downloads'):
    documents = []
    pdf_count = 0
    txt_count = 0

    for filename in os.listdir(download_dir):
        file_path = os.path.join(download_dir, filename)
        if not os.path.isfile(file_path):
            continue

        try:
            #handle different types of files
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
                print(f"[INFO] Skipping unsupported file type: {filename}")
                continue

        #loaded_docs = loader.load()

             #each loaded_doc is a Document object with 'page_content' attribute
            for doc in loaded_docs:
                documents.append(doc.page_content)

        except Exception as e:
            print(F"[Error] Failed to load {filename}: {e}")
                
    print(f"[INFO] Finished loading documents.")
    print(f"[INFO] PDFs loaded: {pdf_count}, TXTs loaded: {txt_count}, Total documents: {len(documents)}")
    
    return documents