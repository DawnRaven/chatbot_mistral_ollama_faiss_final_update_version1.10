RAG_ONLY_MODE = True  # Bật chế độ chỉ trả lời theo tài liệu (RAG Only)
import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.llms import Ollama
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from pydantic import BaseModel
import torch
import socket
import json
import requests
import shutil

embedding = None
vectorstore = None
qa_chain = None


import re
from bs4 import BeautifulSoup

def sanitize_response(response: str) -> str:
    soup = BeautifulSoup(response, "html.parser")

    for tag in soup.find_all():
        if tag.name not in ['a']:
            if tag.name in ['think', 'meta']:
                tag.decompose()
            else:
                tag.unwrap()

    text = str(soup)
    url_pattern = re.compile(r"(?<!href=\")(https?://[^\s\"<>']+)")

    def wrap_url(match):
        url = match.group(1)
        return f'<a href="{url}" target="_blank">{url}</a>'

    result = url_pattern.sub(wrap_url, text)
    return result


# Định nghĩa lại biến toàn cục
INDEX_DIR = "./index"
os.makedirs(INDEX_DIR, exist_ok=True)

print("🚀 CUDA available:", torch.cuda.is_available())
print("🧠 Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

llm = None

def detect_base_url():
    if os.getenv("OLLAMA_HOST"):
        return os.getenv("OLLAMA_HOST")
    if os.path.exists("/.dockerenv"):
        return "http://ollama:11434"
    return "http://localhost:11434"

def resolve_host(hostname):
    try:
        socket.gethostbyname(hostname)
        return True
    except socket.error:
        return False

base_url = detect_base_url()
if resolve_host("ollama"):
    base_url = "http://ollama:11434"
else:
    base_url = "http://host.docker.internal:11434"

embeddings = HuggingFaceEmbeddings(
    model_name="distiluse-base-multilingual-cased-v2",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_instruction():
    try:
        with open("instruction.txt", "r", encoding="utf-8") as f:
            return f.read().strip()
    except:
        return ""

def get_current_model():
    try:
        with open("current_model.txt", "r") as f:
            return f.read().strip()
    except:
        return "mistral"

def create_llm():
    model_name = get_current_model()
    system_instruction = get_instruction()
    return Ollama(model=model_name, base_url=base_url, system=system_instruction)

llm = create_llm()

def create_faiss_index(doc_path):
    loader = Docx2txtLoader(doc_path)
    global embedding, vectorstore, qa_chain
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(INDEX_DIR)
    return vectorstore

if os.listdir(INDEX_DIR):
    try:
        vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    except AssertionError:
        print("⚠️ FAISS index không tương thích, sẽ tạo lại.")
        vectorstore = create_faiss_index("data/your_doc.docx")
else:
    vectorstore = create_faiss_index("data/your_doc.docx")

qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

class Question(BaseModel):
    question: str

# @app.post("/ask")
# async def ask_question(q: Question):
#     answer = qa_chain.invoke(q.question)
#     return {"answer": {"result": sanitize_response(answer)}}

@app.post("/ask")
async def ask_question(query: Question):
    try:
        global llm

        if RAG_ONLY_MODE:
            retriever = qa_chain.retriever
            documents = retriever.get_relevant_documents(query.question)

            if not documents:
                return {"answer": {"result": {"query": query.question, "result": "Xin lỗi, tôi không tìm thấy thông tin trong tài liệu để trả lời câu hỏi này."}}}

            document_content = "\n".join([doc.page_content for doc in documents])

            prompt = f"""
Chỉ sử dụng nội dung dưới đây để trả lời câu hỏi. Nếu không tìm thấy thông tin, hãy trả lời:
"Xin lỗi, tôi không tìm thấy thông tin trong tài liệu để trả lời câu hỏi này."

Nội dung tài liệu:
{document_content}

Câu hỏi:
{query.question}

Trả lời:
"""

            response = llm.invoke(prompt)
            response = sanitize_response(response)

            return {"answer": {"result": {"query": query.question, "result": response}}}

        else:
            # Nếu RAG Only tắt, trả lời như bình thường
            answer = qa_chain.invoke(query.question)
            answerQuery = answer.get("query", "")
            answerResult = answer.get("result", "")
            if not isinstance(answerQuery, str):
                answerQuery = str(answerQuery)
            if not isinstance(answerResult, str):
                answerResult = str(answerResult)

            answerOBJ = {
                "query": sanitize_response(answerQuery),
                "result": sanitize_response(answerResult)
            }
            return {"answer": {"result": answerOBJ}}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": "Đã xảy ra lỗi nội bộ", "details": str(e)})

@app.post("/upload")
async def upload_doc(file: UploadFile = File(...)):
    doc_path = f"data/{file.filename}"
    with open(doc_path, "wb") as buffer:
        buffer.write(await file.read())
    global vectorstore, qa_chain
    vectorstore = create_faiss_index(doc_path)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
    return {"message": "Tài liệu đã được cập nhật thành công!"}

@app.post("/retrain")
async def retrain_faiss():
    global vectorstore, qa_chain
    vectorstore = create_faiss_index("data/your_doc.docx")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
    return {"message": "Đã train lại FAISS index thành công!"}

@app.get("/system/gpu")
def get_gpu_info():
    if torch.cuda.is_available():
        return {"device": torch.cuda.get_device_name(0)}
    return {"device": "CPU"}

from fastapi import Body
from fastapi.responses import JSONResponse

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")

class ModelRequest(BaseModel):
    name: str

@app.get("/models/list")
async def list_models():
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        response.raise_for_status()
        return response.json().get("models", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Không lấy được danh sách model từ Ollama: {str(e)}")

@app.post("/models/pull")
async def pull_model(model: ModelRequest):
    try:
        response = requests.post(f"{OLLAMA_HOST}/api/pull", json={"name": model.name}, timeout=180)
        response.raise_for_status()

        # Lưu model vào file current_model.txt
        with open("current_model.txt", "w", encoding="utf-8") as f:
            f.write(model.name)

        # Tải lại LLM ngay
        global llm
        llm = Ollama(model=model.name)

        return {"message": f"Đã tải và cập nhật model: {model.name}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi tải model: {str(e)}")

@app.post("/reindex")
async def reindex():
    try:
        from langchain_community.document_loaders import UnstructuredWordDocumentLoader
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS

        doc_path = "data/your_doc.docx"
        loader = UnstructuredWordDocumentLoader(doc_path)
        global embedding, vectorstore, qa_chain

        documents = loader.load()

        # Đọc tên embedding từ file nếu có
        embedding_model = "distiluse-base-multilingual-cased-v2"
        if os.path.exists("current_embedding.txt"):
            with open("current_embedding.txt", "r", encoding="utf-8") as f:
                embedding_model = f.read().strip()

        print("✅ Using embedding:", embedding_model)
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        # Tạo FAISS index mới
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local("index")

        return {"status": "success", "message": "Đã cập nhật lại FAISS index"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi cập nhật FAISS index: {str(e)}")


@app.post("/set-model")
async def set_model(model: ModelRequest):
    try:
        with open("current_model.txt", "w") as f:
            f.write(model.name)
        return {"status": "success", "message": f"Đã chọn mô hình: {model.name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lưu mô hình: {str(e)}")



@app.get("/model/status")
async def model_status():
    try:
        # Đọc model đang được backend sử dụng
        with open("current_model.txt", "r") as f:
            backend_model = f.read().strip()
    except:
        backend_model = "Không xác định"

    try:
        # Gọi API /api/tags của Ollama để lấy danh sách model đang có
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        current_ollama_models = [m["name"] for m in models]
    except Exception as e:
        current_ollama_models = f"Lỗi khi kết nối Ollama: {str(e)}"

    return {
        "backend_model": backend_model,
        "ollama_models": current_ollama_models
    }

@app.post("/set-instruction")
async def set_instruction(data: dict = Body(...)):
    try:
        with open("instruction.txt", "w", encoding="utf-8") as f:
            f.write(data.get("instruction", ""))
        global llm, qa_chain
        llm = create_llm()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
        return {"status": "success", "message": "Đã cập nhật instruction!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lưu instruction: {str(e)}")

@app.get("/get-instruction")
async def get_instruction_api():
    try:
        with open("instruction.txt", "r", encoding="utf-8") as f:
            return {"instruction": f.read().strip()}
    except:
        return {"instruction": ""}
import subprocess

@app.get("/docker/containers")
def list_containers():
    if not shutil.which("docker"):
        # Nếu máy không có docker lệnh
        return fallback_list_models()

    try:
        result = subprocess.run(
            ["docker", "ps", "--format",
             "{{.ID}}|{{.Names}}|{{.Ports}}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5
        )

        if result.returncode != 0 or not result.stdout.strip():
            # Nếu docker ps lỗi hoặc không trả về gì ➔ fallback sang API
            return fallback_list_models()

        containers = []
        for line in result.stdout.strip().split("\n"):
            if line:
                parts = line.split("|")
                container_id, name, ports = parts
                ip_cmd = subprocess.run(
                    ["docker", "inspect", "-f",
                     "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", container_id],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                ip = ip_cmd.stdout.strip()
                containers.append({
                    "id": container_id,
                    "name": name,
                    "ip": ip,
                    "ports": ports
                })
        return {"containers": containers}
    except Exception as e:
        # Nếu subprocess lỗi ➔ fallback API
        return fallback_list_models()

def fallback_list_models():
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags")
        response.raise_for_status()
        models = response.json().get("models", [])
        # Định dạng models như list containers
        containers_fake = [
            {"id": model.get("name", ""), "name": model.get("name", ""), "ip": "127.0.0.1", "ports": ""}
            for model in models
        ]
        return {"containers": containers_fake, "message": "Using fallback Ollama API because docker is unavailable."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi fallback lấy models từ Ollama: {str(e)}")

from pydantic import BaseModel

class EmbeddingConfig(BaseModel):
    embedding: str

@app.post("/config/embedding")
async def set_embedding(config: EmbeddingConfig):
    with open("current_embedding.txt", "w", encoding="utf-8") as f:
        f.write(config.embedding)
    return {"message": "Updated embedding"}

@app.get("/config/embedding/current")
async def get_current_embedding():
    embedding_model = "distiluse-base-multilingual-cased-v2"
    if os.path.exists("current_embedding.txt"):
        with open("current_embedding.txt", "r", encoding="utf-8") as f:
            embedding_model = f.read().strip()
    return {"embedding": embedding_model}

class RagConfig(BaseModel):
    enable: bool

@app.post("/config/rag")
async def set_rag_mode(config: RagConfig):
    global RAG_ONLY_MODE
    RAG_ONLY_MODE = config.enable
    return {"message": f"Đã {'bật' if config.enable else 'tắt'} chế độ RAG Only"}

@app.get("/config/rag/current")
async def get_rag_mode():
    return {"rag_only_mode": RAG_ONLY_MODE}

@app.get("/model/status")
async def get_model_status():
    model_name = "mistral"  # Mặc định
    if os.path.exists("current_model.txt"):
        with open("current_model.txt", "r", encoding="utf-8") as f:
            model_name = f.read().strip()

    # Gọi API Ollama để lấy danh sách models đã tải
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        response.raise_for_status()
        models_data = response.json()
        ollama_models = [model.get("name", "") for model in models_data.get("models", [])]
    except Exception as e:
        ollama_models = []

    return {
        "backend_model": model_name,
        "ollama_models": ollama_models
    }

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/chat")
async def get_chat():
    return FileResponse("static/chat.html")