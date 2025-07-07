from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
import re
import logging
import os
from typing import Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

def create_qa_chain():
    """Tạo QA chain với cải tiến toàn diện"""
    try:
        # 1. Load dữ liệu từ file
        logger.info("Loading data from dongho_data.txt")
        loader = TextLoader("Data/dongho_data.txt", encoding='utf-8')
        raw_text = loader.load()[0].page_content
        logger.info(f"Loaded {len(raw_text)} characters of data")

        # 2. Tách từng sản phẩm thành 1 document, và 1 document cho thông tin chung
        product_docs = []
        # Tìm vị trí bắt đầu của từng sản phẩm
        product_splits = [m.start() for m in re.finditer(r"^\d+\. ", raw_text, re.MULTILINE)]
        product_splits.append(raw_text.find("THÔNG TIN CHUNG:"))
        product_splits = [i for i in product_splits if i != -1]
        product_splits = sorted(product_splits)
        
        for i in range(len(product_splits)-1):
            chunk = raw_text[product_splits[i]:product_splits[i+1]].strip()
            if chunk:
                product_docs.append(chunk)
        
        # Thêm thông tin chung
        chung_idx = raw_text.find("THÔNG TIN CHUNG:")
        if chung_idx != -1:
            chung_doc = raw_text[chung_idx:].strip()
            product_docs.append(chung_doc)
        
        logger.info(f"Split data into {len(product_docs)} documents")

        # 3. Tạo vector database
        logger.info("Initializing vector database")
        embedding = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Kiểm tra xem có cần tạo mới vector_db không
        persist_directory = "vector_db"
        if os.path.exists(persist_directory):
            logger.info("Loading existing vector database")
            try:
                vectordb = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=embedding
                )
                # Test connection
                test_result = vectordb.similarity_search("test", k=1)
                logger.info("Successfully loaded existing vector database")
            except Exception as e:
                logger.warning(f"Failed to load existing vector database: {e}")
                logger.info("Creating new vector database")
                vectordb = Chroma.from_texts(
                    product_docs,
                    embedding,
                    persist_directory=persist_directory
                )
        else:
            logger.info("Creating new vector database")
            vectordb = Chroma.from_texts(
                product_docs,
                embedding,
                persist_directory=persist_directory
            )

        # 4. Prompt template tối ưu với context memory
        template = '''[INST]
Bạn là trợ lý tư vấn đồng hồ. Trả lời ngắn gọn, đúng trọng tâm và chuyên nghiệp.

QUY TẮC BẮT BUỘC:
1. CHỈ sử dụng thông tin từ context được cung cấp
2. LUÔN trả lời bằng TIẾNG VIỆT 100%, TUYỆT ĐỐI KHÔNG sử dụng tiếng Anh
3. KHÔNG được xen kẽ tiếng Việt và tiếng Anh trong cùng một câu trả lời
4. KHÔNG sử dụng từ tiếng Anh như "model", "feature", "warranty", "price" - phải dùng từ tiếng Việt
5. Trả lời ngắn gọn, đúng trọng tâm, không lan man
6. KHÔNG sử dụng emoji hoặc biểu tượng
7. KHÔNG in đậm bất cứ câu nào
8. Nếu không có thông tin: "Xin lỗi, tôi chưa có thông tin về..."
9. Trả lời trực tiếp, không dùng câu hỏi cuối

TỪ VỰNG TIẾNG VIỆT BẮT BUỘC:
- "model" → "mẫu mã" hoặc "sản phẩm"
- "feature" → "tính năng" hoặc "đặc điểm"
- "warranty" → "bảo hành"
- "price" → "giá" hoặc "giá cả"
- "water resistant" → "chống nước"
- "stainless steel" → "thép không gỉ"
- "leather strap" → "dây da"
- "dial" → "mặt số"
- "brand" → "thương hiệu"
- "quality" → "chất lượng"
- "design" → "thiết kế"
- "style" → "kiểu dáng"
- "automatic" → "tự động"
- "quartz" → "pin"
- "movement" → "bộ máy"
- "case" → "vỏ"
- "crystal" → "kính"
- "chronograph" → "đồng hồ bấm giờ"
- "date display" → "hiển thị ngày"
- "luminous" → "phát sáng"
- "shock resistant" → "chống sốc"

VÍ DỤ TRẢ LỜI ĐÚNG:
- "Casio MTP-V002L-1B3UDF có giá 1.200.000 VND. Đặc điểm: chống nước 50m, dây da đen, mặt số 38mm. Bảo hành 2 năm."
- "Seiko 5 có giá 2.500.000 VND với tính năng chống nước 100m và dây thép không gỉ."

VÍ DỤ TRẢ LỜI SAI (KHÔNG ĐƯỢC LÀM):
- "Casio model MTP-V002L có price 1.200.000 VND với water resistant feature"
- "Seiko 5 với warranty 2 năm và leather strap"
- "Đồng hồ này có automatic movement và date display"
- "Brand này có good quality và nice design"

QUY TẮC BỔ SUNG:
- Nếu context có từ tiếng Anh, phải dịch sang tiếng Việt khi trả lời
- Không được giữ nguyên tên tiếng Anh của sản phẩm nếu có tên tiếng Việt
- Luôn dùng "đồng hồ" thay vì "watch"
- Luôn dùng "thương hiệu" thay vì "brand"
- Luôn dùng "sản phẩm" thay vì "product"
- Luôn dùng "thông tin" thay vì "information"
- Luôn dùng "liên hệ" thay vì "contact"
- Luôn dùng "chính sách" thay vì "policy"
- Luôn dùng "thanh toán" thay vì "payment"
- Luôn dùng "bảo hành" thay vì "warranty"
- Luôn dùng "đổi trả" thay vì "return"
- Luôn dùng "showroom" thay vì "store"
- Luôn dùng "hotline" thay vì "phone number"

LƯU Ý CUỐI CÙNG:
- TUYỆT ĐỐI KHÔNG sử dụng bất kỳ từ tiếng Anh nào trong câu trả lời
- Nếu không biết từ tiếng Việt, hãy dùng từ tiếng Việt tương tự hoặc mô tả bằng tiếng Việt
- Mục tiêu: 100% tiếng Việt, 0% tiếng Anh

Context: {context}
Câu hỏi: {question}
Trả lời: [/INST]'''

        prompt = PromptTemplate(
            input_variables=["question", "context"],
            template=template
        )

        # 5. Cấu hình LLM với tối ưu hóa
        logger.info("Initializing LLM")
        llm = OllamaLLM(
            model=os.getenv("OLLAMA_MODEL", "llama3"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            top_k=int(os.getenv("LLM_TOP_K", "40")),
            repeat_penalty=float(os.getenv("LLM_REPEAT_PENALTY", "1.2")),
            num_ctx=int(os.getenv("LLM_NUM_CTX", "4096"))
        )

        logger.info("QA chain initialized successfully")
        return {
            "llm": llm,
            "prompt": prompt,
            "vectordb": vectordb
        }
        
    except Exception as e:
        logger.error(f"Failed to create QA chain: {e}")
        raise