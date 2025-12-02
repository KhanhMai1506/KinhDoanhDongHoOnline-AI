from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document
import csv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
import re
import logging
import os
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Configure logging
logger = logging.getLogger(__name__)
 

def create_qa_chain():
    """Tạo QA chain với cải tiến toàn diện"""
    try:
        # 1. Load dữ liệu từ file CSV và xử lý metadata
        logger.info("Loading data from Data_DongHo.csv with metadata extraction")
        product_docs = []
        
        with open("Data/Data_DongHo.csv", newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            for row in reader:
                # Xử lý giá tiền (14.780.000 -> 14780000)
                raw_price = row.get('Giá bán', '0')
                try:
                    price_value = int(raw_price.replace('.', '').replace(',', '').strip()) if raw_price and raw_price != 'NULL' else 0
                except ValueError:
                    price_value = 0
                
                # Xử lý giới tính
                category = row.get('Loại danh mục', '').lower()
                gender = 'unknown'
                if 'nam' in category:
                    gender = 'nam'
                elif 'nữ' in category:
                    gender = 'nu'
                
                # Xử lý số lượng tồn kho
                try:
                    stock = int(row.get('Số lượng', '0'))
                except ValueError:
                    stock = 0

                # Xử lý thông tin khác (Phong cách, Mục đích)
                other_info = row.get('Thông tin khác', '').lower()
                style = []
                if 'thể thao' in other_info: style.append('the_thao')
                if 'cổ điển' in other_info or 'classic' in other_info: style.append('co_dien')
                if 'sang trọng' in other_info: style.append('sang_trong')
                if 'thanh lịch' in other_info: style.append('thanh_lich')
                
                purpose = []
                if 'đi làm' in other_info or 'văn phòng' in other_info: purpose.append('di_lam')
                if 'đi date' in other_info: purpose.append('di_date')
                if 'đi chơi' in other_info: purpose.append('di_choi')
                if 'đi tiệc' in other_info: purpose.append('di_tiec')

                # Xử lý thông số kỹ thuật (Chất liệu dây, Size)
                specs = row.get('Thông số kỹ thuật', '').lower()
                strap_material = 'unknown'
                if 'dây da' in specs: strap_material = 'day_da'
                elif 'kim loại' in specs or 'thép' in specs: strap_material = 'day_kim_loai'
                elif 'vải' in specs: strap_material = 'day_vai'
                elif 'nhựa' in specs or 'cao su' in specs: strap_material = 'day_nhua'

                # Tạo content cho document
                content = "\n".join([f"{k}: {v}" for k, v in row.items() if v and v != 'NULL'])
                
                # Tạo metadata phong phú hơn
                metadata = {
                    "source": "Data/Data_DongHo.csv",
                    "row": row.get('STT', 0),
                    "price": price_value,
                    "gender": gender,
                    "brand": row.get('Thương hiệu', '').lower(),
                    "name": row.get('Tên sản phẩm', ''),
                    "stock": stock,
                    "style": " ".join(style), # Chroma không hỗ trợ list trong filter tốt, dùng string space-separated
                    "purpose": " ".join(purpose),
                    "strap_material": strap_material
                }
                
                product_docs.append(Document(page_content=content, metadata=metadata))
                
        logger.info(f"Loaded {len(product_docs)} documents with metadata")

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
                vectordb = Chroma.from_documents(
                    product_docs,
                    embedding,
                    persist_directory=persist_directory
                )
        else:
            logger.info("Creating new vector database")
            vectordb = Chroma.from_documents(
                product_docs,
                embedding,
                persist_directory=persist_directory
            )



        # 4. Prompt template tối ưu với context memory
        system_template = """Bạn là trợ lý tư vấn đồng hồ chuyên nghiệp.
NHIỆM VỤ: Trả lời câu hỏi dựa trên thông tin sản phẩm và lịch sử trò chuyện.

YÊU CẦU QUAN TRỌNG:
1. Trả lời đầy đủ nhưng ngắn gọn (tối đa 3-4 câu).
2. Đi thẳng vào vấn đề, không vòng vo.
3. Nếu không có thông tin, nói "Tôi chưa có thông tin này".
4. TUYỆT ĐỐI không bịa thông tin, chỉ trả lời câu được hỏi. Nếu không có dữ liệu trả lời, trả lời "Tôi chưa có thông tin này".
5. Trả lời theo tiếng Việt 100%.
6. Nếu câu hỏi là tìm kiếm hoặc tư vấn (ví dụ: "tư vấn", "tìm", "có mẫu nào"), hãy giới thiệu ngắn gọn các sản phẩm phù hợp có trong THÔNG TIN SẢN PHẨM.
7. CHỈ trả lời về sản phẩm được hỏi. TUYỆT ĐỐI KHÔNG tự ý liệt kê sản phẩm khác nếu không được yêu cầu (trừ khi là câu hỏi tư vấn/tìm kiếm).
8. TUYỆT ĐỐI KHÔNG ngắt quãng câu trả lời bằng dấu ba chấm (...). Phải trả lời trọn vẹn câu.
9. Nếu thông tin quá dài, hãy tóm tắt lại nhưng vẫn phải đủ ý và trọn vẹn câu.

LỊCH SỬ TRÒ CHUYỆN:
{history}

THÔNG TIN SẢN PHẨM:
{context}"""

        human_template = "{question}"

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])

        # 5. Cấu hình LLM với tối ưu hóa tốc độ
        logger.info("Initializing LLM")
        llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            top_k=int(os.getenv("LLM_TOP_K", "10")), 
            repeat_penalty=float(os.getenv("LLM_REPEAT_PENALTY", "1.1")),
            num_ctx=int(os.getenv("LLM_NUM_CTX", "2048")), 
            num_predict=1024, # Tăng giới hạn để không bị ngắt quãng
            streaming=True,
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