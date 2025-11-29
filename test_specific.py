
from fastapi.testclient import TestClient
from app import app
import logging

# Disable logging for tests
logging.getLogger("app").setLevel(logging.WARNING)

client = TestClient(app)

def test_specific_questions():
    print("Testing specific attribute queries...")
    
    # Test Price
    print("\n1. Testing Price Query for Casio MTP-V002L...")
    response = client.post("/chat-stream", json={"question": "Casio MTP-V002L giá bao nhiêu?"})
    content = response.text
    print(f"Response: {content}")
    if "1.200.000" in content and len(content.split()) < 20:
        print("PASS: Price query returned concise price.")
    else:
        print("WARNING: Price query response might be too long or incorrect.")

    # Test Warranty
    print("\n2. Testing Warranty Query for Citizen BM8180...")
    response = client.post("/chat-stream", json={"question": "Citizen BM8180 bảo hành bao lâu?"})
    content = response.text
    print(f"Response: {content}")
    if "5 năm" in content and "giá" not in content.lower():
        print("PASS: Warranty query returned warranty info without price.")
    else:
        print("WARNING: Warranty query might contain extra info.")

if __name__ == "__main__":
    try:
        test_specific_questions()
    except Exception as e:
        print(f"Test Failed: {e}")
