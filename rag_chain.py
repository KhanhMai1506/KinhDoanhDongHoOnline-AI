from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document
import csv
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
                if 'tối giản' in other_info: style.append('toi_gian')
                
                purpose = []
                if 'đi làm' in other_info or 'văn phòng' in other_info: purpose.append('di_lam')
                if 'đi học' in other_info: purpose.append('di_hoc')
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
        template = '''[INST]
Bạn là trợ lý tư vấn đồng hồ chuyên nghiệp.
MỤC TIÊU: Trả lời CỰC KỲ NGẮN GỌN, SÚC TÍCH và ĐI THẲNG VÀO VẤN ĐỀ.
NGÔN NGỮ: BẮT BUỘC TRẢ LỜI 100% BẰNG TIẾNG VIỆT. KHÔNG DÙNG TIẾNG ANH.

QUY TẮC CỐT LÕI:
1. KHÔNG chào hỏi rườm rà.
2. KHÔNG lặp lại câu hỏi.
3. KHÔNG DỊCH thông tin từ Context sang tiếng Anh. Giữ nguyên văn tiếng Việt trong Context.
4. Nếu liệt kê sản phẩm, BẮT BUỘC dùng định dạng sau cho từng dòng:
   - [Tên sản phẩm]: [Giá tiền] - [Đặc điểm nổi bật (Tiếng Việt)]

QUY TẮC TRẢ LỜI CÂU HỎI CỤ THỂ (ƯU TIÊN CAO NHẤT):
- Nếu hỏi GIÁ -> CHỈ trả lời giá tiền (Ví dụ: 1.500.000 VND). Giá đã bao gồm VAT.
- Nếu hỏi CÒN HÀNG KHÔNG -> Kiểm tra "Số lượng". Nếu > 0 trả lời "Còn hàng", ngược lại "Hết hàng".
- Nếu hỏi ĐẶC ĐIỂM/TÍNH NĂNG -> Sử dụng thông tin từ "Thông số kỹ thuật" hoặc "Mô tả".
- Nếu hỏi SIZE/KÍCH THƯỚC -> Tìm thông tin "Size mặt" trong "Thông số kỹ thuật".
- Nếu hỏi CHẤT LIỆU DÂY/KÍNH -> Tìm trong "Thông số kỹ thuật".
- Nếu hỏi BẢO HÀNH -> Sử dụng thông tin từ "Bảo hành".
- Nếu hỏi THANH TOÁN -> Sử dụng thông tin từ "Phương thức thanh toán".
- Nếu hỏi ĐỔI TRẢ -> Sử dụng thông tin từ "Điều kiện đổi trả".
- Nếu hỏi Giao Hàng -> Sử dụng thông tin từ "Giao hàng".
- KHÔNG cung cấp thông tin thừa.

VÍ DỤ CHUẨN:
Q: Casio MTP-V002L giá bao nhiêu?
A: 1.200.000 VND (Đã gồm VAT).

Q: Mẫu này còn hàng không?
A: Còn hàng (Số lượng: 45).

Q: Size mặt bao nhiêu?
A: 40mm.

Q: Liệt kê các mẫu đồng hồ Casio?
A: Dưới đây là các mẫu Casio nổi bật:
1. Casio MTP-1374L-1AVDF: 14.780.000 VND - Dây da đen, mặt số thể thao.
2. Casio AE-1200WHD-1AVDF: 1.506.000 VND - Pin 10 năm, giờ thế giới.
3. Casio MTP-VT01L-1BUDF: 1.182.000 VND - Thiết kế tối giản, mỏng nhẹ.

Q: Bảo hành của Citizen BM8180?
A: 5 năm chính hãng.

TỪ VỰNG BẮT BUỘC (DỊCH NẾU THẤY TIẾNG ANH):
- Price -> Giá
- Features -> Đặc điểm
- Warranty -> Bảo hành
- Water resistant -> Chống nước
- Here are -> Dưới đây là
- The top -> Các mẫu hàng đầu
- Stock -> Tồn kho
- VAT -> Thuế GTGT

Context: {context}
Câu hỏi: {question}
Trả lời (Bằng tiếng Việt): [/INST]'''

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