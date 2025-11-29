
from fastapi.testclient import TestClient
from app import app
import logging

# Disable logging for tests
logging.getLogger("app").setLevel(logging.WARNING)

client = TestClient(app)

def test_conciseness():
    print("Testing conciseness for 'Casio' query...")
    response = client.post("/chat-stream", json={"question": "Casio"})
    assert response.status_code == 200
    content = response.text
    print(f"Response:\n{content}")
    
    # Check for concise indicators
    if len(content.split()) > 100:
        print("WARNING: Response might be too long.")
    else:
        print("Response length looks good.")
        
    if "Sản phẩm này có" in content:
        print("WARNING: Found verbose phrase 'Sản phẩm này có'")

if __name__ == "__main__":
    try:
        test_conciseness()
    except Exception as e:
        print(f"Test Failed: {e}")
