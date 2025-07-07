from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
from rag_chain import create_qa_chain
from typing import Dict, Optional, List
import re
import uuid
import logging
import time
from datetime import datetime, timedelta
from collections import defaultdict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Smart Watch Chatbot API",
    description="AI-powered chatbot for watch consultation with context memory",
    version="1.0.0"
)

# Security
security = HTTPBearer(auto_error=False)

# Rate limiting
request_counts = defaultdict(list)
MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "60"))

# Khởi tạo QA chain
try:
    qa_chain = create_qa_chain()
    llm = qa_chain["llm"]
    prompt = qa_chain["prompt"]
    vectordb = qa_chain["vectordb"]
    logger.info("QA chain initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize QA chain: {e}")
    raise

# Lưu trữ ngữ cảnh theo session với cải tiến
session_contexts: Dict[str, Dict] = {}
conversation_history: Dict[str, List[Dict]] = {}

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:8080").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="Câu hỏi của người dùng")
    session_id: Optional[str] = Field(None, description="ID session để duy trì ngữ cảnh")
    user_id: Optional[str] = Field(None, description="ID người dùng (tùy chọn)")


class ChatResponse(BaseModel):
    session_id: str
    response: str
    context_info: Optional[Dict] = None


def validate_input(question: str) -> str:
    """Validate và sanitize input"""
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống")
    
    question = question.strip()
    if len(question) > 500:
        raise HTTPException(status_code=400, detail="Câu hỏi quá dài (tối đa 500 ký tự)")
    
    # Basic sanitization
    question = re.sub(r'<script.*?</script>', '', question, flags=re.IGNORECASE)
    return question


def check_rate_limit(session_id: str) -> bool:
    """Kiểm tra rate limit"""
    now = time.time()
    minute_ago = now - 60
    
    # Clean old requests
    request_counts[session_id] = [req_time for req_time in request_counts[session_id] if req_time > minute_ago]
    
    # Check limit
    if len(request_counts[session_id]) >= MAX_REQUESTS_PER_MINUTE:
        return False
    
    request_counts[session_id].append(now)
    return True


def is_relevant_question(question: str) -> bool:
    """Kiểm tra câu hỏi liên quan với cải tiến"""
    question = question.lower().strip()
    
    # Từ khóa về đồng hồ, thương hiệu, thông tin chung
    keywords = [
        'đồng hồ', 'watch', 'sản phẩm', 'mẫu mã', 'hàng', 'mua', 'bán',
        'casio', 'seiko', 'citizen', 'orient', 'tissot', 'omega', 'rolex',
        'giá', 'giá cả', 'bao nhiêu tiền', 'đặc điểm', 'tính năng',
        'bảo hành', 'chính hãng', 'đeo tay', 'thương hiệu', 'loại',
        'liên hệ', 'chính sách', 'đổi trả', 'thanh toán', 'trả góp',
        'địa chỉ', 'số điện thoại', 'hỗ trợ', 'showroom', 'hotline', 'contact',
        'tư vấn', 'giới thiệu', 'so sánh', 'khuyến mãi', 'giảm giá'
    ]
    
    brands = ['casio', 'seiko', 'citizen', 'orient', 'tissot', 'omega', 'rolex']
    
    # Nếu câu hỏi chỉ là tên thương hiệu
    if question in brands:
        return True
    
    # Kiểm tra từ khóa
    return any(keyword in question for keyword in keywords)


def extract_product_info(context: str) -> dict[str, Optional[str]]:
    """Trích xuất thông tin sản phẩm từ context với cải tiến"""
    info: dict[str, Optional[str]] = {
        "product_name": None,
        "price": None,
        "features": None,
        "brand": None,
        "warranty": None
    }

    # Tìm thương hiệu và tên sản phẩm (cải tiến regex)
    brand_patterns = [
        r"(Casio|Seiko|Citizen|Orient|Tissot|Omega|Rolex)\s+([\w\-]+(?:\s+[\w\-]+)*)",
        r"(\d+\.\s+)?(Casio|Seiko|Citizen|Orient|Tissot|Omega|Rolex)\s+([\w\-]+(?:\s+[\w\-]+)*)"
    ]
    
    for pattern in brand_patterns:
        brand_matches = re.finditer(pattern, context, re.IGNORECASE)
        for match in brand_matches:
            info["brand"] = match.group(1) if match.group(1) and not match.group(1).isdigit() else match.group(2)
            product_part = match.group(2) if match.group(1) and not match.group(1).isdigit() else match.group(3)
            info["product_name"] = f"{info['brand']} {product_part}"
            break
        if info["product_name"]:
            break

    # Tìm giá (cải tiến)
    price_patterns = [
        r"giá[:\s]+([\d,\.]+)\s*(VND|đ)",
        r"mức giá[:\s]+([\d,\.]+)\s*(VND|đ)",
        r"([\d,\.]+)\s*(VND|đ)"
    ]
    
    for pattern in price_patterns:
        price_match = re.search(pattern, context, re.IGNORECASE)
        if price_match:
            info["price"] = price_match.group(1)
            break

    # Tìm đặc điểm
    features_match = re.search(r"đặc điểm[:\s]+(.+?)(?=\n|$)", context, re.IGNORECASE)
    if features_match:
        info["features"] = features_match.group(1).strip()

    # Tìm bảo hành
    warranty_match = re.search(r"bảo hành[:\s]+(.+?)(?=\n|$)", context, re.IGNORECASE)
    if warranty_match:
        info["warranty"] = warranty_match.group(1).strip()

    return info


def enhance_context_with_history(session_id: str, current_context: str) -> str:
    """Tăng cường context với lịch sử hội thoại"""
    if session_id not in conversation_history:
        return current_context
    
    history = conversation_history[session_id]
    if not history:
        return current_context
    
    # Lấy 3 câu hỏi gần nhất
    recent_history = history[-3:]
    history_context = "\n".join([
        f"Q: {item['question']}\nA: {item['answer'][:200]}..." 
        for item in recent_history
    ])
    
    enhanced_context = f"""
Lịch sử hội thoại gần đây:
{history_context}

Thông tin hiện tại:
{current_context}
"""
    return enhanced_context


def handle_follow_up(question: str, context: Dict, session_id: str) -> Optional[str]:
    """Xử lý câu hỏi tiếp theo với cải tiến ngữ cảnh"""
    question = question.lower()
    current_product = context.get("current_product")
    conversation_ctx = context.get("conversation_context", "")

    logger.info(f"Follow-up check - Question: '{question}', Current product: {current_product}")

    if not current_product and not conversation_ctx:
        logger.info("No current product or conversation context found")
        return None

    # Xử lý đại từ (nó, cái này, sản phẩm này...)
    pronouns = ["nó", "cái này", "sản phẩm này", "đồng hồ này", "mẫu này", 
                "sản phẩm đó", "đồng hồ đó", "mẫu đó", "cái đó", "thứ đó"]
    if any(pronoun in question for pronoun in pronouns):
        logger.info(f"Pronoun detected in question: {question}")
        if current_product:
            if "giá" in question or "bao nhiêu tiền" in question:
                if context.get("price"):
                    logger.info(f"Returning price info for {current_product}")
                    return f"{current_product} có giá {context['price']} VND."
                logger.info(f"No price info found for {current_product}")
                return f"Xin lỗi, tôi chưa có thông tin giá cho {current_product}."

            if "đặc điểm" in question or "tính năng" in question:
                if context.get("features"):
                    logger.info(f"Returning features for {current_product}")
                    return f"{current_product} có đặc điểm: {context['features']}."
                logger.info(f"No features info found for {current_product}")
                return f"Xin lỗi, tôi chưa có thông tin chi tiết về {current_product}."
                
            if "bảo hành" in question:
                if context.get("warranty"):
                    logger.info(f"Returning warranty info for {current_product}")
                    return f"{current_product} có {context['warranty']}."
                logger.info(f"No warranty info found for {current_product}")
                return f"Xin lỗi, tôi chưa có thông tin bảo hành cho {current_product}."

            # Xử lý câu hỏi chung về thông tin sản phẩm
            if "thông tin" in question:
                info_parts = []
                if context.get("price"):
                    info_parts.append(f"Giá: {context['price']} VND")
                if context.get("features"):
                    info_parts.append(f"Đặc điểm: {context['features']}")
                if context.get("warranty"):
                    info_parts.append(f"Bảo hành: {context['warranty']}")
                
                if info_parts:
                    logger.info(f"Returning general info for {current_product}")
                    return f"Thông tin về {current_product}: " + ", ".join(info_parts) + "."
                else:
                    logger.info(f"No detailed info found for {current_product}")
                    return f"Xin lỗi, tôi chưa có thông tin chi tiết về {current_product}."

            # Xử lý câu hỏi dựa trên ngữ cảnh hội thoại
        if conversation_ctx:
            if "so sánh" in question or "khác biệt" in question:
                return "Dựa trên thông tin đã trao đổi, tôi có thể so sánh các sản phẩm cho bạn. Bạn muốn so sánh điểm gì cụ thể?"
            
            if "khuyến nghị" in question or "gợi ý" in question:
                return "Dựa trên sở thích bạn đã chia sẻ, tôi có thể đưa ra gợi ý phù hợp. Bạn quan tâm đến mức giá nào?"

    logger.info("No follow-up response generated")
    return None


def clean_expired_sessions():
    """Dọn dẹp các session cũ với cải tiến"""
    now = datetime.now()
    expired_keys = []
    
    for key, value in session_contexts.items():
        last_activity = value.get("last_activity")
        if last_activity and now - last_activity > timedelta(hours=2):  # Tăng thời gian lưu trữ
            expired_keys.append(key)
    
    for key in expired_keys:
        del session_contexts[key]
        if key in conversation_history:
            del conversation_history[key]
        if key in request_counts:
            del request_counts[key]
    
    if expired_keys:
        logger.info(f"Cleaned {len(expired_keys)} expired sessions")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(session_contexts),
        "vector_db_status": "connected" if vectordb else "disconnected"
    }


@app.get("/stats")
async def get_stats():
    """Thống kê sử dụng"""
    return {
        "total_sessions": len(session_contexts),
        "total_conversations": sum(len(conv) for conv in conversation_history.values()),
        "rate_limit": MAX_REQUESTS_PER_MINUTE
    }


def remove_markdown(text: str) -> str:
    """Loại bỏ markdown như **bold**, *italic*, __bold__, _italic_, `code`"""
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    text = re.sub(r'`(.*?)`', r'\1', text)
    return text


@app.post("/chat-stream", response_model=ChatResponse)
async def chat_stream(req: ChatRequest, request: Request):
    """Chat endpoint với cải tiến toàn diện"""
    start_time = time.time()
    
    try:
        # Validate input
        question = validate_input(req.question)
        session_id = req.session_id or str(uuid.uuid4())
        
        # Log session information
        logger.info(f"Session ID: {session_id}")
        logger.info(f"Question: '{question}'")
        
        # Rate limiting
        if not check_rate_limit(session_id):
            raise HTTPException(status_code=429, detail="Quá nhiều yêu cầu. Vui lòng thử lại sau 1 phút.")
        
        # Clean expired sessions
        clean_expired_sessions()
        
        # Lấy ngữ cảnh hiện tại hoặc tạo mới
        context = session_contexts.get(session_id, {})
        context["last_activity"] = datetime.now()
        
        # Log current context
        logger.info(f"Current context for session {session_id}: {context}")
        
        # Log request
        logger.info(f"Session {session_id}: {question}")
        
        # Kiểm tra câu hỏi không liên quan
        if not is_relevant_question(question):
            async def fallback():
                yield "Xin lỗi, tôi chỉ hỗ trợ thông tin về đồng hồ. Bạn có thể hỏi về sản phẩm, giá cả, chính sách hoặc thông tin liên hệ."
            
            return StreamingResponse(fallback(), media_type="text/plain")

        # Xử lý câu hỏi tiếp theo dựa trên ngữ cảnh
        follow_up_response = handle_follow_up(question, context, session_id)
        if follow_up_response:
            logger.info(f"Follow-up response for session {session_id}: {follow_up_response}")
            async def respond():
                yield follow_up_response
            
            # Lưu vào lịch sử
            if session_id not in conversation_history:
                conversation_history[session_id] = []
            conversation_history[session_id].append({
                "question": question,
                "answer": follow_up_response,
                "timestamp": datetime.now()
            })
            
            return StreamingResponse(respond(), media_type="text/plain")

        # Truy vấn vector DB với cải tiến
        search_result = vectordb.similarity_search_with_score(question, k=3)
        relevant_docs = [doc for doc, score in search_result if score < 0.8]
        context_text = "\n".join([doc.page_content for doc in relevant_docs]) if relevant_docs else ""
        
        # Tăng cường context với lịch sử
        enhanced_context = enhance_context_with_history(session_id, context_text)
        
        # Log context để debug
        logger.info(f"Question: '{question}'")
        logger.info(f"Found {len(relevant_docs)} relevant docs")
        logger.info(f"Context: {context_text[:500]}...")
        logger.info(f"Enhanced context: {enhanced_context[:500]}...")

        # Cập nhật ngữ cảnh nếu tìm thấy sản phẩm
        if relevant_docs:
            product_info = extract_product_info(context_text)
            if product_info["product_name"]:
                context.update({
                    "current_product": product_info["product_name"],
                    "brand": product_info["brand"],
                    "price": product_info["price"],
                    "features": product_info["features"],
                    "warranty": product_info["warranty"],
                    "conversation_context": context_text
                })
                logger.info(f"Updated context with product info: {product_info}")

        # Lưu ngữ cảnh mới
        session_contexts[session_id] = context
        logger.info(f"Saved context for session {session_id}: {context}")

        # Xử lý khi không có thông tin phù hợp
        if not relevant_docs:
            if context.get("current_product"):
                async def no_info_current():
                    yield f"Bạn đang hỏi về {context['current_product']}. Tôi chưa có thông tin chi tiết. Bạn muốn biết gì thêm?"
                return StreamingResponse(no_info_current(), media_type="text/plain")
            elif any(pronoun in question.lower() for pronoun in ["nó", "cái này", "sản phẩm này", "đồng hồ này"]):
                async def no_info_pronoun():
                    yield "Bạn vui lòng nói rõ tên sản phẩm đồng hồ mà bạn muốn hỏi."
                return StreamingResponse(no_info_pronoun(), media_type="text/plain")
            else:
                brands_in_data = ["Casio", "Seiko", "Citizen", "Orient"]
                async def no_info_brand():
                    yield f"Hiện chúng tôi có đồng hồ các thương hiệu: {', '.join(brands_in_data)}. Bạn quan tâm dòng nào?"
                return StreamingResponse(no_info_brand(), media_type="text/plain")

        # Kiểm tra xem context có chứa thông tin phù hợp không
        if not context_text.strip():
            async def no_context():
                yield "Xin lỗi, tôi không tìm thấy thông tin liên quan. Vui lòng hỏi cụ thể hơn về sản phẩm, thương hiệu hoặc thông tin chung."
            return StreamingResponse(no_context(), media_type="text/plain")

        # Xử lý thông thường với LLM
        inputs = {"question": question, "context": enhanced_context}

        async def generate():
            try:
                response_chunks = []
                for chunk in llm.stream(prompt.format(**inputs)):
                    clean_chunk = remove_markdown(chunk)
                    response_chunks.append(clean_chunk)
                    yield clean_chunk
                # Lưu vào lịch sử
                full_response = "".join(response_chunks)
                if session_id not in conversation_history:
                    conversation_history[session_id] = []
                conversation_history[session_id].append({
                    "question": question,
                    "answer": full_response,
                    "timestamp": datetime.now()
                })
                # Log response time
                response_time = time.time() - start_time
                logger.info(f"Session {session_id}: Response time {response_time:.2f}s")
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                yield f"Đã xảy ra lỗi: {str(e)}"

        response = StreamingResponse(generate(), media_type="text/plain")
        response.set_cookie(
            key="session_id",
            value=session_id,
            max_age=7200,  # 2 giờ
            httponly=True,
            samesite="lax"
        )
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Lỗi server nội bộ")


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)