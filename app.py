from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
from rag_chain import create_qa_chain
from typing import Dict, Optional, List, Any
import re
import uuid
import logging
import time
from datetime import datetime, timedelta
from collections import defaultdict
import os
import asyncio
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
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:8001").split(","),
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
        'tư vấn', 'giới thiệu', 'so sánh', 'khuyến mãi', 'giảm giá',
        'giao hàng', 'ship', 'vận chuyển',
        'chống nước', 'size', 'màu', 'kích thước', 'mặt kính', 'dây', 'pin', 'cơ', 'automatic',
        'nó', 'cái này', 'cái đấy', 'em nó', 'bé này'
    ]
    
    brands = ['casio', 'seiko', 'citizen', 'orient', 'tissot', 'omega', 'rolex']
    
    # Nếu câu hỏi chỉ là tên thương hiệu
    if question in brands:
        return True
    
    # Kiểm tra từ khóa
    return any(keyword in question for keyword in keywords)


def detect_specific_model(question: str) -> Optional[str]:
    """Phát hiện xem câu hỏi có chứa model cụ thể không"""
    # Pattern: Brand + Code (e.g., Casio MTP-1374L)
    brands = ['casio', 'seiko', 'citizen', 'orient', 'tissot', 'omega', 'rolex', 'doxa', 'saga']
    for brand in brands:
        if brand in question.lower():
            # Tìm từ ngay sau brand
            match = re.search(fr"{brand}\s+([A-Za-z0-9\-]+)", question, re.IGNORECASE)
            if match:
                code = match.group(1)
                # Nếu code có số HOẶC có dấu gạch ngang, khả năng cao là model
                if any(c.isdigit() for c in code) or "-" in code:
                    return code
    return None


def extract_search_filters(question: str) -> Dict[str, Any]:
    """Trích xuất bộ lọc từ câu hỏi"""
    filters = {}
    question_lower = question.lower()
    
    # Lọc theo giới tính
    if "nam" in question_lower and "nữ" not in question_lower:
        filters["gender"] = "nam"
    elif "nữ" in question_lower and "nam" not in question_lower:
        filters["gender"] = "nu"
        
    # Lọc theo giá
    # Dưới X triệu
    under_match = re.search(r"dưới\s+(\d+)\s*(triệu|tr|m)", question_lower)
    if under_match:
        amount = int(under_match.group(1)) * 1000000
        filters["price"] = {"$lt": amount}
        
    # Trên X triệu
    over_match = re.search(r"(trên|hơn)\s+(\d+)\s*(triệu|tr|m)", question_lower)
    if over_match:
        amount = int(over_match.group(2)) * 1000000
        filters["price"] = {"$gt": amount}
        
    # Khoảng X-Y triệu (hỗ trợ -, đến, tới)
    range_match = re.search(r"từ\s+(\d+)\s*(?:-|đến|tới)\s*(\d+)\s*(triệu|tr|m)", question_lower)
    if range_match:
        min_amount = int(range_match.group(1)) * 1000000
        max_amount = int(range_match.group(2)) * 1000000
        # Chroma cần tách riêng các điều kiện
        if "$and" not in filters:
             filters["$and"] = []
        filters["$and"].append({"price": {"$gte": min_amount}})
        filters["$and"].append({"price": {"$lte": max_amount}})
        
    # Lọc theo chất liệu dây
    if "dây da" in question_lower:
        if "$and" not in filters: filters["$and"] = []
        filters["$and"].append({"strap_material": "day_da"})
    elif "dây kim loại" in question_lower or "thép" in question_lower:
        if "$and" not in filters: filters["$and"] = []
        filters["$and"].append({"strap_material": "day_kim_loai"})
    elif "dây vải" in question_lower:
        if "$and" not in filters: filters["$and"] = []
        filters["$and"].append({"strap_material": "day_vai"})
    elif "dây nhựa" in question_lower or "cao su" in question_lower:
        if "$and" not in filters: filters["$and"] = []
        filters["$and"].append({"strap_material": "day_nhua"})

    # Lọc theo Phong cách (Style)
    # ChromaDB không hỗ trợ $contains cho string metadata một cách đơn giản.
    # Chúng ta sẽ dựa vào vector search để tìm kiếm phong cách (thể thao, sang trọng...)
    # vì các từ khóa này đã có trong page_content.
    pass 

    # Nếu có nhiều hơn 1 điều kiện (không phải range đã xử lý), dùng $and
    final_filters = {}
    conditions = []
    
    # Gom các điều kiện đơn lẻ
    for k, v in filters.items():
        if k == "$and":
            conditions.extend(v)
        else:
            conditions.append({k: v})
            
    if len(conditions) > 1:
        return {"$and": conditions}
    elif len(conditions) == 1:
        return conditions[0]
    
    return {}


def handle_comparison(question: str, vectordb) -> Optional[str]:
    """Xử lý câu hỏi so sánh"""
    question_lower = question.lower()
    
    # Pattern 1: So sánh A với/và B
    compare_match = re.search(r"so sánh\s+(.+?)\s+(?:với|và)\s+(.+)", question_lower)
    
    # Pattern 2: Giữa A và B...
    if not compare_match:
        compare_match = re.search(r"giữa\s+(.+?)\s+(?:và|với)\s+(.+?)\s+(?:thì|mẫu nào|cái nào)", question_lower)

    if not compare_match:
        return None
        
    prod1 = compare_match.group(1).strip()
    prod2 = compare_match.group(2).strip()
    
    # Clean up product names (remove "đồng hồ", "mẫu")
    prod1 = re.sub(r"^(đồng hồ|mẫu)\s+", "", prod1).strip()
    prod2 = re.sub(r"^(đồng hồ|mẫu)\s+", "", prod2).strip()

    logger.info(f"Comparing {prod1} and {prod2}")
    
    # Tìm kiếm thông tin cho từng sản phẩm
    docs1 = vectordb.similarity_search(prod1, k=1)
    docs2 = vectordb.similarity_search(prod2, k=1)
    
    if not docs1 or not docs2:
        return None
        
    context = f"""
THÔNG TIN SẢN PHẨM 1 ({prod1}):
{docs1[0].page_content}

THÔNG TIN SẢN PHẨM 2 ({prod2}):
{docs2[0].page_content}
"""
    return context


def extract_product_info(context: str) -> dict[str, Optional[str]]:
    """Trích xuất thông tin sản phẩm từ context với cải tiến"""
    info: dict[str, Optional[str]] = {
        "product_name": None,
        "price": None,
        "features": None,
        "brand": None,
        "warranty": None,
        "stock": None,
        "contact": None
    }

    # Regex for CSV format (prioritized)
    
    # Tìm tên sản phẩm
    name_match = re.search(r"Tên sản phẩm:\s*(.+?)(?=\n|$)", context, re.IGNORECASE)
    if name_match:
        info["product_name"] = name_match.group(1).strip()
    
    # Tìm thương hiệu
    brand_match = re.search(r"Thương hiệu:\s*(.+?)(?=\n|$)", context, re.IGNORECASE)
    if brand_match:
        info["brand"] = brand_match.group(1).strip()
        
    # Fallback regex for Name/Brand
    if not info["product_name"]:
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

    # Tìm giá
    price_match_csv = re.search(r"Giá bán:\s*(.+?)(?=\n|$)", context, re.IGNORECASE)
    if price_match_csv:
        raw_price = price_match_csv.group(1).strip()
        # Remove (Đã bao gồm VAT) for cleaner number parsing if needed, or keep it
        # Let's keep it as is for display, but maybe clean it for calculation if we were doing that.
        # For display: "14780000 (Đã bao gồm VAT)" is fine.
        info["price"] = raw_price
    else:
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

    # Tìm số lượng (Stock)
    stock_match = re.search(r"Số lượng:\s*(\d+)", context, re.IGNORECASE)
    if stock_match:
        info["stock"] = stock_match.group(1)

    # Tìm đặc điểm / Thông số kỹ thuật
    specs_match = re.search(r"Thông số kỹ thuật:\s*(.+?)(?=\n|$)", context, re.IGNORECASE | re.DOTALL)
    if specs_match:
        info["features"] = specs_match.group(1).strip()
    else:
        desc_match = re.search(r"Mô tả:\s*(.+?)(?=\n|$)", context, re.IGNORECASE | re.DOTALL)
        if desc_match:
            info["features"] = desc_match.group(1).strip()
        else:
            features_match = re.search(r"đặc điểm[:\s]+(.+?)(?=\n|$)", context, re.IGNORECASE)
            if features_match:
                info["features"] = features_match.group(1).strip()

    # Tìm bảo hành
    warranty_match_csv = re.search(r"Bảo hành:\s*(.+?)(?=\n|$)", context, re.IGNORECASE)
    if warranty_match_csv:
        info["warranty"] = warranty_match_csv.group(1).strip()
    else:
        warranty_match = re.search(r"bảo hành[:\s]+(.+?)(?=\n|$)", context, re.IGNORECASE)
        if warranty_match:
            info["warranty"] = warranty_match.group(1).strip()

    # Tìm thông tin liên hệ - Cải thiện regex để bắt nhiều dòng
    contact_match = re.search(r"Thông tin liên hệ:[\s\n]*([^\n]+(?:\n[^\n]+)*)", context, re.IGNORECASE)
    if contact_match:
        # Clean up the captured text
        raw_contact = contact_match.group(1).strip()
        # Remove any trailing unrelated text if regex captured too much (unlikely with this pattern but good safety)
        info["contact"] = raw_contact
    else:
        # Fallback regex for email/phone if "Thông tin liên hệ:" header is missing or different
        email_match = re.search(r"Email:\s*([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", context)
        phone_match = re.search(r"Số điện thoại:\s*([\d\.]+)", context)
        contact_parts = []
        if email_match: contact_parts.append(f"Email: {email_match.group(1)}")
        if phone_match: contact_parts.append(f"Hotline: {phone_match.group(1)}")
        
        if contact_parts:
            info["contact"] = ", ".join(contact_parts)
        else:
            info["contact"] = None

    return info


def resolve_coreference(question: str, context: Dict) -> str:
    """Giải quyết tham chiếu (đại từ, số thứ tự) để làm rõ câu hỏi"""
    question_lower = question.lower()
    
    # 1. Xử lý số thứ tự (cái đầu tiên, mẫu thứ 2...)
    last_list = context.get("last_product_list", [])
    if last_list:
        ordinal_map = {
            "đầu tiên": 0, "thứ nhất": 0, "số 1": 0,
            "thứ hai": 1, "thứ 2": 1, "số 2": 1,
            "thứ ba": 2, "thứ 3": 2, "số 3": 2,
            "thứ tư": 3, "thứ 4": 3, "số 4": 3,
            "thứ năm": 4, "thứ 5": 4, "số 5": 4
        }
        for key, idx in ordinal_map.items():
            # Kiểm tra từ khóa
            if key in question_lower:
                # Yêu cầu có từ chỉ loại đi kèm hoặc là các từ đặc biệt
                triggers = ["cái", "mẫu", "chiếc", "con", "em", "sản phẩm"]
                is_valid = any(f"{t} {key}" in question_lower for t in triggers)
                
                if key in ["đầu tiên", "thứ nhất", "cuối cùng"]:
                    is_valid = True
                    
                if is_valid and idx < len(last_list):
                    prod_name = last_list[idx]
                    logger.info(f"Resolved ordinal '{key}' to '{prod_name}'")
                    # Thay thế từ khóa bằng tên sản phẩm
                    # Ví dụ: "Cái đầu tiên..." -> "Đồng hồ Casio... có..."
                    # Dùng replace 1 lần để tránh thay thế nhầm nếu lặp lại
                    return question_lower.replace(key, prod_name, 1)

    # 2. Xử lý đại từ (nó, cái này...)
    pronouns = ["nó", "cái này", "sản phẩm này", "đồng hồ này", "em nó", "bé này", "chiếc này", "giá này", "giá đó"]
    current_product = context.get("current_product")
    if current_product and any(p in question_lower for p in pronouns):
        logger.info(f"Resolved pronoun to '{current_product}'")
        return f"{current_product} {question}"
        
    return question


def get_conversation_history(session_id: str) -> str:
    """Lấy lịch sử hội thoại định dạng text"""
    if session_id not in conversation_history:
        return ""
    
    history = conversation_history[session_id]
    if not history:
        return ""
    
    # Lấy 5 câu hỏi gần nhất để tăng ngữ cảnh
    recent_history = history[-5:]
    history_text = "\n".join([
        f"User: {item['question']}\nAssistant: {item['answer'][:300]}" 
        for item in recent_history
    ])
    
    return history_text


def handle_follow_up(question: str, context: Dict, session_id: str) -> Optional[str]:
    """Xử lý câu hỏi tiếp theo với cải tiến ngữ cảnh"""
    question = question.lower()
    current_product = context.get("current_product")
    conversation_ctx = context.get("conversation_context", "")

    logger.info(f"Follow-up check - Question: '{question}', Current product: {current_product}")

    # Xử lý câu hỏi về liên hệ (Global check)
    if any(kw in question for kw in ["liên hệ", "địa chỉ", "số điện thoại", "email", "hotline"]):
        if context.get("contact"):
            return f"Thông tin liên hệ: {context['contact']}"
        # Fallback if not in context but maybe in general knowledge or footer
        return "Bạn có thể liên hệ với chúng tôi qua Hotline: 0905350808 hoặc Email: topwatch@gmail.com"

    if not current_product and not conversation_ctx:
        logger.info("No current product or conversation context found")
        return None

    # Xử lý tham chiếu thứ tự (mẫu đầu tiên, cái thứ 2...)
    # Đã được xử lý bởi resolve_coreference để rewrite query
    # Nên ở đây chúng ta bỏ qua để query đi xuống RAG
    pass

    # Xử lý đại từ (nó, cái này, sản phẩm này...)
    pronouns = ["nó", "cái này", "sản phẩm này", "đồng hồ này", "mẫu này", 
                "sản phẩm đó", "đồng hồ đó", "mẫu đó", "cái đó", "thứ đó",
                "cái đấy", "em nó", "bé này", "chiếc này", "chiếc đó",
                "giá này", "giá đó"] # Added implicit price references
    if any(pronoun in question for pronoun in pronouns):
        logger.info(f"Pronoun detected in question: {question}")
        if current_product:
            # Check VAT (Moved up to prioritize over general price check)
            if "vat" in question or "thuế" in question:
                return f"Giá bán của {current_product} đã bao gồm thuế VAT."

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
            
            # Check stock
            if "còn hàng" in question or "có sẵn" in question or "tồn kho" in question:
                stock = context.get("stock")
                if stock:
                    try:
                        stock_num = int(stock)
                        status = "còn hàng" if stock_num > 0 else "đã hết hàng"
                        return f"Sản phẩm {current_product} hiện {status} (Số lượng: {stock})."
                    except:
                        return f"Hiện tại {current_product} đang có sẵn tại cửa hàng."
                return f"Sản phẩm {current_product} hiện đang có sẵn."



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
        
        # Giải quyết tham chiếu (Coreference Resolution)
        # Điều này giúp biến "Nó giá bao nhiêu" thành "Casio ABC giá bao nhiêu"
        resolved_question = resolve_coreference(question, context)
        is_specific_query = False
        if resolved_question != question:
            logger.info(f"Rewrote question: '{question}' -> '{resolved_question}'")
            question = resolved_question
            is_specific_query = True
            
        original_question = question
        
        # Log current context
        logger.info(f"Current context for session {session_id}: {context}")
        
        # Log request
        logger.info(f"Session {session_id}: {question}")
        
        # Xử lý câu chào hỏi
        greeting_keywords = ["hi", "hello", "chào", "xin chào", "alo"]
        if any(question.lower().strip().startswith(kw) for kw in greeting_keywords) and len(question.split()) <= 4:
             async def greeting_response():
                yield "Chào bạn 👋 Tôi có thể hỗ trợ gì cho bạn hôm nay?"
             return StreamingResponse(greeting_response(), media_type="text/plain")

        # Kiểm tra câu hỏi không liên quan
        if not is_relevant_question(question):
            async def fallback():
                yield "Xin lỗi, tôi chỉ hỗ trợ thông tin về đồng hồ. Bạn có thể hỏi về sản phẩm, giá cả, chính sách hoặc thông tin liên hệ."
            
            return StreamingResponse(fallback(), media_type="text/plain")

        # Xử lý yêu cầu tư vấn
        consult_keywords = ["tư vấn", "giới thiệu", "các loại", "các hãng", "mua đồng hồ"]
        
        # Kiểm tra xem có tên thương hiệu nào trong câu hỏi không
        brands_map = {
            'casio': 'Casio',
            'seiko': 'Seiko',
            'citizen': 'Citizen',
            'orient': 'Orient',
            'doxa': 'Doxa',
            'saga': 'Saga'
        }
        has_brand = any(brand in question.lower() for brand in brands_map)
        
        # Chỉ hiển thị danh sách thương hiệu nếu câu hỏi ngắn và chung chung
        # Ví dụ: "Tư vấn", "Tư vấn đồng hồ", "Giới thiệu sản phẩm"
        is_generic_consult = any(kw in question.lower() for kw in consult_keywords)
        is_short_query = len(question.split()) <= 4
        
        if is_generic_consult and not has_brand and is_short_query:
            async def consult_response():
                yield "Chào bạn, hiện tại chúng tôi đang kinh doanh các thương hiệu đồng hồ sau:\n- Casio\n- Seiko\n- Citizen\n- Orient\n- Doxa\n- Saga\n\nBạn muốn tìm hiểu về thương hiệu nào?"
            return StreamingResponse(consult_response(), media_type="text/plain")

        # Xử lý khi người dùng chọn thương hiệu
        brands_map = {
            'casio': 'Casio',
            'seiko': 'Seiko',
            'citizen': 'Citizen',
            'orient': 'Orient',
            'doxa': 'Doxa',
            'saga': 'Saga'
        }
        
        # Initialize variables to prevent UnboundLocalError
        is_brand_query = False
        relevant_docs = []
        context_text = ""
        
        found_brand = None
        for key, value in brands_map.items():
            if key in question.lower():
                found_brand = value
                break
        
        is_brand_query = False
        # Nếu câu hỏi ngắn (chỉ tên thương hiệu) hoặc có ý định xem sản phẩm thương hiệu
        # Nếu câu hỏi ngắn (chỉ tên thương hiệu) hoặc có ý định xem sản phẩm thương hiệu
        if found_brand:
            # Kiểm tra xem có phải là model cụ thể không
            specific_code = detect_specific_model(question)
            
            general_keywords = ["thương hiệu", "sản phẩm", "các loại", "các mẫu", "tìm hiểu", "xem", "liệt kê", "danh sách", "tư vấn"]
            
            # Chỉ coi là brand query nếu KHÔNG phải là model cụ thể
            if not specific_code:
                if len(question.split()) <= 6 or any(kw in question.lower() for kw in general_keywords):
                    is_brand_query = True
                    # Điều chỉnh câu hỏi để RAG tìm kiếm tốt hơn
                    question = f"Liệt kê danh sách các mẫu đồng hồ {found_brand} nổi bật nhất kèm giá bán và đặc điểm."
                    logger.info(f"Optimized question for brand listing: {question}")

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

        # Xử lý đại từ để duy trì ngữ cảnh (Context Retention)
        pronouns = ["nó", "cái này", "sản phẩm này", "đồng hồ này", "mẫu này", 
                    "sản phẩm đó", "đồng hồ đó", "mẫu đó", "cái đó", "thứ đó",
                    "cái đấy", "em nó", "bé này", "chiếc này", "chiếc đó"]
        has_pronoun = any(p in question.lower() for p in pronouns)
        forced_context = False
        
        if has_pronoun and context.get("current_product") and context.get("conversation_context"):
            logger.info(f"Pronoun detected ({question}). Using stored context for {context['current_product']}")
            context_text = context["conversation_context"]
            relevant_docs = [True] # Dummy to indicate docs found
            forced_context = True
            
        # Xử lý so sánh (chỉ chạy nếu không phải forced context hoặc nếu câu hỏi có chứa "so sánh")
        comparison_context = None
        if not forced_context or "so sánh" in question.lower():
             comparison_context = handle_comparison(question, vectordb)

        if comparison_context:
            logger.info("Comparison context generated")
            context_text = comparison_context
            relevant_docs = [True] # Dummy to pass check
            
            # Cập nhật prompt cho so sánh
            question = f"So sánh chi tiết 2 sản phẩm dựa trên thông tin được cung cấp: {question}"
        elif not forced_context:
            # Truy vấn vector DB thông thường với bộ lọc
            filters = extract_search_filters(original_question)
            
            # Nếu là specific query (đã resolve coreference), bỏ qua filters để tìm kiếm chính xác sản phẩm
            if 'is_specific_query' in locals() and is_specific_query:
                logger.info("Specific query detected. Ignoring filters.")
                filters = {}
            
            # Nếu là brand query, thêm filter thương hiệu
            if 'is_brand_query' in locals() and is_brand_query and found_brand:
                brand_filter = {"brand": found_brand.lower()}
                if filters:
                    if "$and" in filters:
                        filters["$and"].append(brand_filter)
                    else:
                        # Wrap existing filter and brand filter in $and
                        filters = {"$and": [filters, brand_filter]}
                else:
                    filters = brand_filter
            
            logger.info(f"Search filters: {filters}")
            
            search_k = 6 if 'is_brand_query' in locals() and is_brand_query else 3
            
            if filters:
                # Nếu có filter, dùng filter
                search_result = vectordb.similarity_search_with_score(question, k=search_k, filter=filters)
            else:
                search_result = vectordb.similarity_search_with_score(question, k=search_k)
                
            # Log search results for debugging
            logger.info(f"Search results for '{question}':")
            for doc, score in search_result:
                logger.info(f"  - Score: {score:.4f}, Content: {doc.page_content[:50]}...")

            relevant_docs = [doc for doc, score in search_result if score < 1.8]
            
            # Nếu là specific query (đã resolve coreference), lọc relevant_docs để chỉ giữ lại đúng sản phẩm
            # Nếu là specific query (đã resolve coreference), lọc relevant_docs để chỉ giữ lại đúng sản phẩm
            if 'is_specific_query' in locals() and is_specific_query:
                specific_code = detect_specific_model(question)
                if specific_code:
                    # Strict filtering: Only keep docs that contain the specific code
                    filtered_docs = [doc for doc in relevant_docs if specific_code.lower() in doc.page_content.lower()]
                    
                    if filtered_docs:
                        logger.info(f"Filtered docs for specific model {specific_code}: {len(filtered_docs)} docs")
                        relevant_docs = filtered_docs
                    else:
                        # If no docs match the specific code, it's better to return nothing than irrelevant products
                        # However, sometimes the code format might differ slightly.
                        # Let's try to be smart: if we found relevant docs but none match the code, 
                        # it might be a retrieval error or the product isn't there.
                        # To be safe and avoid hallucination, we will limit to the top 1 doc if it has a high score
                        logger.warning(f"Specific model {specific_code} detected but not found in top docs. Limiting to top 1.")
                        if relevant_docs:
                             relevant_docs = [relevant_docs[0]]

            context_text = "\n".join([doc.page_content for doc in relevant_docs]) if relevant_docs else ""
            


        # TỐI ƯU HÓA: Trả về kết quả trực tiếp cho câu hỏi liệt kê (Bypass LLM)
        if is_brand_query and relevant_docs:
            logger.info("Bypassing LLM for brand listing query")
            
            response_lines = [f"Dưới đây là các mẫu đồng hồ {found_brand} nổi bật:\n"]
            
            for i, doc in enumerate(relevant_docs):
                # Extract info using metadata (more reliable)
                name = doc.metadata.get('name')
                price = doc.metadata.get('price')
                
                # Fallback to extraction if metadata missing
                if not name or not price:
                    extracted = extract_product_info(doc.page_content)
                    if not name: name = extracted['product_name']
                    if not price: price = extracted['price']
                
                # Format price
                try:
                    price_str = f"{int(price):,.0f}" if price else "Liên hệ"
                except:
                    price_str = str(price)
                
                # Get features from page_content
                features = "Thiết kế sang trọng, chính hãng." # Default
                
                # Try to find "Mô tả" or "Thông số kỹ thuật"
                desc_match = re.search(r"Mô tả:\s*(.+?)(?=\n|$)", doc.page_content, re.IGNORECASE)
                if desc_match:
                    features = desc_match.group(1).strip()
                
                # Truncate features if too long
                if len(features) > 120:
                    features = features[:117] + "..."

                response_lines.append(f"{i+1}. **{name}**")
                response_lines.append(f"   - 💰 Giá: {price_str} VND")
                response_lines.append(f"   - ✨ {features}\n")

            final_response = "\n".join(response_lines)
            
            # Save to history
            if session_id not in conversation_history:
                conversation_history[session_id] = []
            conversation_history[session_id].append({
                "question": question,
                "answer": final_response,
                "timestamp": datetime.now()
            })

             # Save product list to context for ordinal reference
            context['last_product_list'] = []
            context_content_list = []
            for doc in relevant_docs:
                 name = doc.metadata.get('name')
                 if not name:
                     extracted = extract_product_info(doc.page_content)
                     name = extracted['product_name']
                 if name:
                     context['last_product_list'].append(name)
                 context_content_list.append(doc.page_content)
            
            # Lưu nội dung vào conversation_context để dùng cho câu hỏi sau
            context['conversation_context'] = "\n".join(context_content_list)
            
            # Update session context
            session_contexts[session_id] = context

            async def direct_response():
                for line in response_lines:
                    yield line + "\n"
                    await asyncio.sleep(0.02)
            
            return StreamingResponse(direct_response(), media_type="text/plain")
        
        # Lấy lịch sử hội thoại
        history_text = get_conversation_history(session_id)
        
        # Log context để debug
        logger.info(f"Question: '{question}'")
        logger.info(f"Found {len(relevant_docs)} relevant docs")
        logger.info(f"Context: {context_text[:200]}...")
        logger.info(f"History: {history_text[:200]}...")

        # Cập nhật ngữ cảnh nếu tìm thấy sản phẩm
        if relevant_docs:
            product_info = extract_product_info(context_text)
            if product_info["product_name"]:
                # Logic to decide whether to update current_product
                should_update = True
                
                # Don't update if it's a purely general question without product identifiers
                general_intent_keywords = ["giao hàng", "ship", "vận chuyển", "thanh toán", "địa chỉ", "đổi trả", "liên hệ", "shop", "cửa hàng", "online"]
                is_general = any(kw in question.lower() for kw in general_intent_keywords)
                
                # Check if question has brand or specific model code
                brands_map_keys = ['casio', 'seiko', 'citizen', 'orient', 'doxa', 'saga', 'tissot', 'omega', 'rolex']
                has_brand_or_code = any(b in question.lower() for b in brands_map_keys) or detect_specific_model(question)
                
                if is_general and not has_brand_or_code:
                    should_update = False
                    logger.info("General question detected. Preserving previous product context.")

                if should_update:
                    context.update({
                        "current_product": product_info["product_name"],
                        "brand": product_info["brand"],
                        "price": product_info["price"],
                        "features": product_info["features"],
                        "warranty": product_info["warranty"],
                        "stock": product_info.get("stock"),
                        "contact": product_info.get("contact"),
                        "conversation_context": context_text
                    })
                    logger.info(f"Updated context with product info: {product_info}")

        # Lưu ngữ cảnh mới
        session_contexts[session_id] = context
        logger.info(f"Saved context for session {session_id}: {context}")

        # Xử lý khi không có thông tin phù hợp
        # Xử lý khi không có thông tin phù hợp từ search
        if not relevant_docs:
            # Kiểm tra xem có thể dùng ngữ cảnh cũ không
            pronouns = ["nó", "cái này", "sản phẩm này", "đồng hồ này", "mẫu này", "sản phẩm đó", "đồng hồ đó"]
            has_pronoun = any(p in question.lower() for p in pronouns)
            
            if has_pronoun and context.get("conversation_context"):
                logger.info("Using previous conversation context for follow-up")
                context_text = context["conversation_context"]
                # Proceed to LLM generation with this context
            else:
                # Fallback responses
                if context.get("current_product"):
                    async def no_info_current():
                        yield f"Bạn đang hỏi về {context['current_product']}. Tuy nhiên câu hỏi này nằm ngoài thông tin tôi có. Bạn có thể hỏi về giá, thông số hoặc bảo hành."
                    return StreamingResponse(no_info_current(), media_type="text/plain")
                elif has_pronoun:
                    async def no_info_pronoun():
                        yield "Bạn vui lòng nói rõ tên sản phẩm đồng hồ mà bạn muốn hỏi."
                    return StreamingResponse(no_info_pronoun(), media_type="text/plain")
                else:
                    brands_in_data = ["Casio", "Seiko", "Citizen", "Orient"]
                    async def no_info_brand():
                        yield f"Hiện chúng tôi có đồng hồ các thương hiệu: {', '.join(brands_in_data)}. Xin lỗi tôi không tìm thấy thông tin cho câu hỏi của bạn."
                    return StreamingResponse(no_info_brand(), media_type="text/plain")

        # Kiểm tra Model cụ thể có trong kết quả không (Chống hallucination)
        # Tạm thời bỏ qua check này vì nó quá strict và gây ra false negative
        # specific_code = detect_specific_model(question)
        # if specific_code and relevant_docs:
        #     # Kiểm tra xem code có trong context_text không
        #     if specific_code.lower() not in context_text.lower():
        #         logger.info(f"Specific model {specific_code} not found in retrieved docs")
        #         async def not_found_response():
        #             yield f"Xin lỗi, hiện tại shop chưa có sẵn mẫu đồng hồ {specific_code} hoặc thông tin chưa được cập nhật."
        #         return StreamingResponse(not_found_response(), media_type="text/plain")

        # Kiểm tra xem context có chứa thông tin phù hợp không
        if not context_text.strip():
            async def no_context():
                yield "Xin lỗi, tôi không tìm thấy thông tin liên quan. Vui lòng hỏi cụ thể hơn về sản phẩm, thương hiệu hoặc thông tin chung."
            return StreamingResponse(no_context(), media_type="text/plain")

        # Xử lý thông thường với LLM
        # Nếu là câu hỏi tìm kiếm (có filters) và không phải là follow-up cụ thể, 
        # chúng ta nên hạn chế history để tránh nhiễu (context pollution).
        # Tuy nhiên, nếu là câu hỏi so sánh, history có thể cần thiết.
        
        effective_history = history_text
        if 'filters' in locals() and filters and not comparison_context:
             logger.info("Search query detected with filters. Clearing history to focus on new search results.")
             effective_history = ""

        inputs = {
            "question": question, 
            "context": context_text,
            "history": effective_history
        }

        async def generate():
            try:
                response_chunks = []
                first_chunk = True
                buffer = ""
                in_think_block = False
                
                async for chunk in llm.astream(prompt.format(**inputs)):
                    content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                    
                    if content:
                        buffer += content
                        
                        # Check for <think> tags
                        if "<think>" in buffer:
                            in_think_block = True
                            
                        if in_think_block:
                            if "</think>" in buffer:
                                # Remove the think block and yield the rest
                                buffer = re.sub(r'<think>.*?</think>', '', buffer, flags=re.DOTALL)
                                in_think_block = False
                            else:
                                # Still in think block, wait for closing tag
                                continue
                        
                        # If not in think block, yield content
                        if not in_think_block and buffer:
                            # Skip leading whitespace/newlines to prevent empty bubbles
                            if first_chunk:
                                if not buffer.strip():
                                    buffer = "" # Keep buffering if only whitespace
                                    continue
                                first_chunk = False
                            
                            yield buffer
                            response_chunks.append(buffer)
                            buffer = ""
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