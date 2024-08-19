import os
import numpy as np
import sqlite3
import logging
import faiss
import torch
from cryptography.fernet import Fernet
import requests
from dotenv import load_dotenv
import gradio as gr
import time
import json

# 设置日志
logging.basicConfig(filename='rag_task.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# GPU 检查
gpu_available = torch.cuda.is_available()

# 全局变量来存储已经加载的索引和嵌入数据
loaded_faiss_index = None
loaded_embeddings = None
loaded_file_paths = None

# 加载 Faiss 索引
def load_faiss_index(file_path):
    global loaded_faiss_index  # 使用全局变量

    if loaded_faiss_index is not None:
        # 如果索引已经加载，直接返回已加载的索引
        print("Using pre-loaded Faiss index from GPU")
        return loaded_faiss_index

    try:
        start_time = time.time()
        index = faiss.read_index(file_path)
        if gpu_available:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                print("Successfully loaded index to GPU")
            except Exception as e:
                print(f"Error loading index to GPU: {e}")
                print("Using CPU index")
        loaded_faiss_index = index  # 将加载后的索引存储在全局变量中
        logging.info(f"Faiss index loaded from {file_path}")
        logging.info(f"Time taken to load Faiss index: {time.time() - start_time:.2f} seconds")
        return loaded_faiss_index
    except Exception as e:
        logging.error(f"Error loading Faiss index: {e}")
        raise

# 从数据库加载嵌入
def load_embeddings_from_db(conn):
    global loaded_embeddings, loaded_file_paths  # 使用全局变量

    if loaded_embeddings is not None and loaded_file_paths is not None:
        # 如果嵌入数据已经加载，直接返回已加载的数据
        print("Using pre-loaded embeddings from DB")
        return loaded_file_paths, loaded_embeddings

    try:
        start_time = time.time()
        c = conn.cursor()
        c.execute("SELECT folder_name, file_path, embedding FROM embeddings")
        results = c.fetchall()
        file_paths = [(row[0], row[1]) for row in results]
        embeddings = [np.frombuffer(row[2], dtype=np.float32) for row in results]
        loaded_file_paths = file_paths  # 将加载后的文件路径存储在全局变量中
        loaded_embeddings = np.array(embeddings)  # 将加载后的嵌入数据存储在全局变量中
        logging.info(f"Time taken to load embeddings from DB: {time.time() - start_time:.2f} seconds")
        return loaded_file_paths, loaded_embeddings
    except sqlite3.Error as e:
        logging.error(f"SQLite error in load_embeddings_from_db: {e}")
        raise

# 获取查询文本的嵌入
def get_embedding(query, api_key, embedding_url):
    start_time = time.time()
    payload = {
        "model": "BAAI/bge-large-zh-v1.5",
        "input": query
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}"
    }
    try:
        response = requests.post(embedding_url, json=payload, headers=headers)
        response.raise_for_status()
        response_json = response.json()
        embedding = response_json['data'][0]['embedding']
        logging.info(f"Time taken to get embedding: {time.time() - start_time:.2f} seconds")
        return np.array(embedding, dtype=np.float32)
    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP Request failed in get_embedding: {e}")
        raise ValueError(f"HTTP Request failed: {e}")

# 查询 Faiss 索引并返回相关内容
def query_faiss_index(index, file_paths, query_embedding, conn, top_k=1, similarity_threshold=0.1):
    start_time = time.time()
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    faiss.normalize_L2(query_embedding)
    D, I = index.search(query_embedding, top_k)
    
    if D[0][0] > similarity_threshold:
        folder_name, file_path = file_paths[I[0][0]]
        c = conn.cursor()
        c.execute("SELECT summary FROM embeddings WHERE file_path = ?", (file_path,))
        result = c.fetchone()
        logging.info(f"Time taken to query Faiss index: {time.time() - start_time:.2f} seconds")
        if result:
            summary = result[0]
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return content, file_path, folder_name, summary
            except FileNotFoundError:
                logging.error(f"File not found: {file_path}")
                return "File not found.", file_path, folder_name, summary
        else:
            logging.error(f"No summary found for file: {file_path}")
            return "No summary found.", file_path, folder_name, None
    else:
        return "No relevant content found.", None, None, None

def generate_answer(llm_url, api_key, user_question, extracted_content, chat_history):
    messages = [
        {"role": "system", "content": "你是一个智能助手，尽可能根据用户的问题和提供的相关内容生成回答。如果没有相关内容，请尽力回答用户的问题。"},
    ]
    
    for message in chat_history:
        messages.append({"role": "user", "content": message[0]})
        messages.append({"role": "assistant", "content": message[1]})
    
    if extracted_content and extracted_content != "No relevant content found.":
        messages.append({"role": "user", "content": f"根据以下用户问题和检索到的相关内容生成回答：\n用户问题：{user_question}\n相关内容：{extracted_content}\n请生成适合的回答。"})
    else:
        messages.append({"role": "user", "content": f"请回答以下问题，即使没有检索到信息：\n{user_question}"})

    llm_payload = {
        "model": "Qwen/Qwen2-72B-Instruct",
        "messages": messages,
        "stream": True
    }
    llm_headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    try:
        response = requests.post(llm_url, json=llm_payload, headers=llm_headers, stream=True)
        response.raise_for_status()

        collected_answer = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    decoded_line = decoded_line[len("data: "):]
                try:
                    json_data = json.loads(decoded_line)
                    if "choices" in json_data:
                        delta = json_data["choices"][0]["delta"]
                        if "content" in delta:
                            content = delta["content"]
                            collected_answer += content
                            yield collected_answer
                except json.JSONDecodeError:
                    continue
        
    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP Request failed in generate_answer: {e}")
        yield f"HTTP Request failed: {e}"
    except ValueError as e:
        logging.error(f"JSON Decode Error in generate_answer: {e}")
        yield f"JSON Decode Error: {e}"

def rag_system(user_input, chat_history):
    index_file_path = 'faiss_index.bin'
    db_path = 'embeddings.db'
    conn = None

    try:
        conn = sqlite3.connect(db_path)
        faiss_index = load_faiss_index(index_file_path)  # 现在只加载一次索引
        file_paths, embeddings = load_embeddings_from_db(conn)  # 现在只加载一次嵌入数据

        load_dotenv()

        encrypted_api_key = os.getenv("ENCRYPTED_API_KEY")
        encrypted_llm_url = os.getenv("ENCRYPTED_LLM_URL")
        encrypted_embedding_url = os.getenv("ENCRYPTED_EMBEDDING_URL")

        key_file_path = "secret_new.key"
        with open(key_file_path, "rb") as key_file:
            key = key_file.read()
        cipher_suite = Fernet(key)

        api_key = cipher_suite.decrypt(encrypted_api_key.encode()).decode()
        llm_url = cipher_suite.decrypt(encrypted_llm_url.encode()).decode()
        embedding_url = cipher_suite.decrypt(encrypted_embedding_url.encode()).decode()

        query_embedding = get_embedding(user_input, api_key, embedding_url)
        related_content, file_path, folder_name, summary = query_faiss_index(faiss_index, file_paths, query_embedding, conn, similarity_threshold=0.5)

        # 记录检索到内容的标题到日志中
        logging.info(f"Retrieved content for user query: {user_input}")
        logging.info(f"Related content summary: {summary}")

        for partial_answer in generate_answer(llm_url, api_key, user_input, related_content, chat_history):
            yield partial_answer

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        yield f"An error occurred: {e}"
    finally:
        if conn:
            conn.close()

# Gradio界面设计
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        user_message = history[-1][0]
        bot_message = ""
        for partial_message in rag_system(user_message, history[:-1]):
            bot_message = partial_message
            history[-1][1] = bot_message
            yield history
        
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()
