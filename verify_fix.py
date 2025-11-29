
import logging
import shutil
import os
from rag_chain import create_qa_chain

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_fix():
    try:
        chain = create_qa_chain()
        vectordb = chain["vectordb"]
        
        # Test queries
        queries = ["Seiko", "Casio"]
        
        for q in queries:
            logger.info(f"Testing query: {q}")
            search_result = vectordb.similarity_search_with_score(q, k=3)
            
            # Mimic app.py logic
            relevant_docs = [doc for doc, score in search_result if score < 1.8]
            
            if relevant_docs:
                logger.info(f"SUCCESS: Found {len(relevant_docs)} relevant docs for '{q}'")
                for doc in relevant_docs:
                    logger.info(f"  - Content: {doc.page_content[:50]}...")
            else:
                logger.error(f"FAILURE: No relevant docs found for '{q}' with threshold 1.8")
                
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    verify_fix()
