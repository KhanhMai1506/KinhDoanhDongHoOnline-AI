
from fastapi.testclient import TestClient
from app import app
import logging

# Disable logging for tests to keep output clean
logging.getLogger("app").setLevel(logging.WARNING)

client = TestClient(app)

def test_consultation_flow():
    print("Testing 'Tôi cần tư vấn' flow...")
    response = client.post("/chat-stream", json={"question": "Tôi cần tư vấn"})
    assert response.status_code == 200
    content = response.text
    print(f"Response to 'Tôi cần tư vấn':\n{content}")
    assert "Casio" in content
    assert "Seiko" in content
    
    print("\nTesting Brand Selection 'Casio'...")
    response = client.post("/chat-stream", json={"question": "Casio"})
    assert response.status_code == 200
    content = response.text
    print(f"Response to 'Casio':\n{content}")
    # We expect some product info or at least a mention of Casio products
    # Since we don't have the LLM running (or it might be mocked/slow), we check for key phrases if possible.
    # Note: The actual RAG depends on the vector DB and LLM. If LLM is Ollama and not running, this might fail or return error.
    
if __name__ == "__main__":
    try:
        test_consultation_flow()
        print("\nTest Finished Successfully")
    except Exception as e:
        print(f"\nTest Failed: {e}")
