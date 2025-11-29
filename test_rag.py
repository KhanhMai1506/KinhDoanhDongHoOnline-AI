
import logging
import shutil
import os
from rag_chain import create_qa_chain

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_rag():
    # Force rebuild of vector db for testing
    if os.path.exists("vector_db"):
        shutil.rmtree("vector_db")
        logger.info("Removed existing vector_db")

    try:
        chain = create_qa_chain()
        vectordb = chain["vectordb"]
        
        # Test queries
        queries = ["Seiko", "Casio", "giá rẻ", "bảo hành"]
        for q in queries:
            logger.info(f"Testing query: {q}")
            results = vectordb.similarity_search_with_score(q, k=3)
            for doc, score in results:
                logger.info(f"  - Score: {score:.4f}, Content: {doc.page_content[:50]}...")
                
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    test_rag()
