# Kịch bản Test Chatbot Đồng Hồ (Phiên bản đầy đủ)

Dưới đây là danh sách các câu hỏi để kiểm tra toàn diện khả năng của chatbot, bao gồm các tính năng mới nhất và các lỗi đã được khắc phục.

## 1. Hỏi về Thông tin sản phẩm chi tiết
*Mục tiêu: Kiểm tra khả năng trích xuất thông tin chính xác từ dữ liệu.*

*   **Thông số kỹ thuật:**
    *   "Size mặt của đồng hồ Casio MTP-1374L-1AVDF là bao nhiêu?"
    *   "Chất liệu dây của Seiko 5 Sports là gì?"
    *   "Đồng hồ Citizen BM8180 có chống nước không?"
*   **Giá & Tồn kho & VAT:**
    *   "Giá của Casio AE-1200WHD-1AVDF là bao nhiêu?"
    *   "Giá này đã bao gồm thuế VAT chưa?"
    *   "Mẫu Casio AE-1200WHD-1AVDF còn hàng không?"
    *   "Kiểm tra giúp tôi mẫu AE-1200WHD-1AVDF có sẵn không?"

## 2. Tư vấn & Lọc sản phẩm nâng cao
*Mục tiêu: Kiểm tra logic lọc theo Giá, Giới tính, Chất liệu dây và phân biệt câu hỏi chung/cụ thể.*

*   **Tư vấn chung (Nên ra danh sách thương hiệu):**
    *   "Tư vấn"
    *   "Tư vấn đồng hồ"
    *   "Giới thiệu sản phẩm"
*   **Tư vấn cụ thể (Nên ra danh sách sản phẩm - Đã fix lỗi):**
    *   "Tư vấn đồng hồ cho nam từ 2 đến 5 triệu"
    *   "Giới thiệu đồng hồ nữ dây da sang trọng"
    *   "Tìm đồng hồ nam dưới 2 triệu"
*   **Lọc theo chất liệu:**
    *   "Tôi muốn tìm đồng hồ dây da"
    *   "Có mẫu nào dây kim loại cho nam không?"
    *   "Tìm đồng hồ dây nhựa giá rẻ"

## 3. So sánh sản phẩm
*Mục tiêu: Kiểm tra tính năng so sánh 2 sản phẩm.*

*   "So sánh Casio AE-1200WHD-1AVDF với Seiko 5 Sports SSK045K1"
*   "So sánh Citizen BM8180 và Orient Bambino"
*   "Giữa Casio AE-1200WHD-1AVDF và Seiko 5 Sports SSK045K1 mẫu nào rẻ hơn?"

## 4. Hỏi về Chính sách & Dịch vụ
*Mục tiêu: Kiểm tra các câu trả lời mặc định về chính sách.*

*   **Bảo hành:** "Mẫu này bảo hành bao lâu?"
*   **Giao hàng:** "Shop có ship hàng toàn quốc không?" hoặc "Bao lâu thì nhận được hàng?"
*   **Đổi trả:** "Điều kiện đổi trả hàng là gì?"
*   **Thanh toán:** "Có hỗ trợ trả góp không?" hoặc "Tôi có thể thanh toán khi nhận hàng (COD) không?"

## 5. Liệt kê theo Thương hiệu
*Mục tiêu: Kiểm tra khả năng liệt kê danh sách sản phẩm.*

*   "Liệt kê các mẫu đồng hồ Casio"
*   "Thương hiệu Seiko có những mẫu nào?"
*   "Các sản phẩm của Orient"

## 6. Kịch bản Hội thoại liên tục (Context Memory)
*Mục tiêu: Kiểm tra khả năng nhớ ngữ cảnh và xử lý đại từ.*

*   **Kịch bản 1:**
    1.  User: "Giá của Casio MTP-1374L là bao nhiêu?"
    2.  Bot: (Trả lời giá)
    3.  User: "Nó có chống nước không?" (Bot phải hiểu "Nó" là Casio MTP-1374L)
    4.  User: "Còn hàng không?"
*   **Kịch bản 2:**
    1.  User: "Tư vấn đồng hồ Seiko"
    2.  Bot: (Liệt kê các mẫu Seiko)
    3.  User: "Mẫu đầu tiên giá bao nhiêu?" (Kiểm tra khả năng hiểu tham chiếu)

## 7. Xử lý lỗi & Sản phẩm không tồn tại
*Mục tiêu: Kiểm tra khả năng phát hiện và từ chối trả lời bịa đặt (hallucination).*

*   "Casio MTP-9999 có bán không?" (Mã không tồn tại -> Báo lỗi)
*   "Giá của đồng hồ Rolex Fake 123 là bao nhiêu?"
*   "Shop có bán Apple Watch không?"

Vẫn còn lỗi khi hỏi về sản phẩm không tồn tại và trả lời còn bị ngắt câu ở dạng liệt kê sản phẩm của thương hiệu. Ghi nhớ ngữ cảnh còn hạn chế