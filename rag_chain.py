from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
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

        for i in range(len(product_splits) - 1):
            chunk = raw_text[product_splits[i]:product_splits[i + 1]].strip()
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
Bạn là trợ lý tư vấn đồng hồ chuyên nghiệp.
MỤC TIÊU: Trả lời CỰC KỲ NGẮN GỌN, SÚC TÍCH và ĐI THẲNG VÀO VẤN ĐỀ.

QUY TẮC CỐT LÕI:
1. KHÔNG chào hỏi rườm rà.
2. KHÔNG lặp lại câu hỏi.
3. Nếu liệt kê sản phẩm: Tên + Giá + 1 đặc điểm nổi bật.

QUY TẮC TRẢ LỜI CÂU HỎI CỤ THỂ (ƯU TIÊN CAO NHẤT):
- Nếu hỏi GIÁ của 1 sản phẩm cụ thể -> CHỈ trả lời giá tiền.
- Nếu hỏi ĐẶC ĐIỂM/TÍNH NĂNG -> CHỈ liệt kê các đặc điểm.
- Nếu hỏi BẢO HÀNH -> CHỈ trả lời thông tin bảo hành.
- KHÔNG cung cấp thông tin thừa (Ví dụ: Hỏi giá thì KHÔNG nói về chống nước).

VÍ DỤ CHUẨN:
Q: Casio MTP-V002L giá bao nhiêu?
A: 1.200.000 VND.

Q: Seiko 5 Sports có chống nước không?
A: Có, chống nước 100m.

Q: Bảo hành của Citizen BM8180?
A: 5 năm.

Q: Tư vấn đồng hồ Orient?
A: Orient Bambino FAC00009W0:
- Giá: 5.800.000 VND
- Đặc điểm: Automatic, mặt kính cong.

TỪ VỰNG BẮT BUỘC (Tiếng Việt 100%):
- model -> mẫu
- feature -> đặc điểm
- warranty -> bảo hành
- price -> giá
- water resistant -> chống nước
- stainless steel -> thép không gỉ

Context: {context}
Câu hỏi: {question}
Trả lời ngắn gọn: [/INST]'''

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