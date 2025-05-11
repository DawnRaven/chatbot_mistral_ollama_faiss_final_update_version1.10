# Chatbot Mistral + Ollama + FAISS

Đây là hệ thống chatbot sử dụng mô hình Ollama (ví dụ: Mistral) để trả lời câu hỏi dựa trên nội dung tài liệu `.docx` kết hợp với FAISS và LangChain.

## 🧱 Thành phần chính
- **FastAPI backend**
- **LangChain + FAISS** cho tìm kiếm thông tin
- **Ollama** làm LLM backend
- **Giao diện web** (`chat.html`, `settings.html`, `upload.html`)

---

## 🚀 Cài đặt và chạy

### 1. Cài Docker Desktop (Windows, macOS, Linux)
https://www.docker.com/products/docker-desktop/

> Yêu cầu Docker hỗ trợ GPU nếu bạn dùng GPU cho Ollama

### 2. Build và chạy dịch vụ

```bash
docker-compose up --build
```

### 3. Truy cập giao diện
- Trò chuyện: [http://localhost:8000/chat.html](http://localhost:8000/chat.html)
- Quản lý mô hình: [http://localhost:8000/settings.html](http://localhost:8000/settings.html)
- Tải tài liệu: [http://localhost:8000/upload.html](http://localhost:8000/upload.html)

---

## 🛠 Các chức năng chính

### ✅ Trò chuyện AI
- Sử dụng mô hình đang chọn để trả lời
- Trả lời dựa trên tài liệu `.docx` trong thư mục `data/`

### ✅ Giao diện chọn mô hình
- Liệt kê model từ Ollama (`/models/list`)
- Tải model mới (`/models/pull`)
- Chọn mô hình sử dụng (`/set-model`)
- Cập nhật lại FAISS index (`/reindex`)

---

## 📁 Cấu trúc thư mục

```
.
├── app.py                   # Backend FastAPI chính
├── Dockerfile               # Dockerfile cho backend
├── docker-compose.yml       # Cấu hình Docker Compose (gồm cả Ollama)
├── requirements.txt         # Thư viện Python
├── data/your_doc.docx       # Tài liệu để nhúng FAISS
├── index/                   # FAISS index đã lưu
├── chat.html                # Giao diện trò chuyện
├── settings.html            # Giao diện chọn mô hình
├── upload.html              # Giao diện upload tài liệu
└── current_model.txt        # Mô hình hiện đang được chọn
```

---

## 📌 Lưu ý
- Mặc định sử dụng mô hình `"mistral"` nếu chưa chọn mô hình
- Tài liệu hỗ trợ định dạng `.docx`, cần đặt ở thư mục `data/`
- Chạy `POST /reindex` sau khi thay đổi tài liệu để cập nhật index

---

## 📧 Hỗ trợ
Liên hệ người phát triển để được hỗ trợ hoặc nâng cấp thêm tính năng.