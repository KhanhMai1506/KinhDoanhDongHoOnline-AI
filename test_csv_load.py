from langchain_community.document_loaders import CSVLoader
import os

try:
    loader = CSVLoader(
        file_path="Data/Data_DongHo.csv", 
        encoding='utf-8', 
        csv_args={
            "delimiter": ";",
            "quotechar": '"',
        }
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} documents")
    if len(docs) > 0:
        print("First document content:")
        print(docs[0].page_content)
except Exception as e:
    print(f"Error: {e}")
