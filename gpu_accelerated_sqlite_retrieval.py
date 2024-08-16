import os
import requests
import numpy as np
from cryptography.fernet import Fernet
import json
from dotenv import load_dotenv
import sqlite3
import logging
from datetime import datetime
import torch

# 设置日志
logging.basicConfig(filename='embedding_system.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# GPU 检查
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

# 尝试导入GPU版本的Faiss，如果不可用则使用CPU版本
try:
    import faiss
    print("Faiss version:", faiss.__version__)
    gpu_available = torch.cuda.is_available() and hasattr(faiss, 'GpuIndexFlatIP')
    if gpu_available:
        print("Using GPU version of Faiss")
    else:
        print("Using CPU version of Faiss")
        if torch.cuda.is_available():
            print("GPU is available, but Faiss-GPU is not installed or not compatible")
        else:
            print("GPU is not available")
except ImportError:
    import faiss
    gpu_available = False
    print("Using CPU version of Faiss")

# 加载 .env 文件中的环境变量
load_dotenv()

# 从环境变量中获取加密密钥和 API 信息
encrypted_llm_url = os.getenv("ENCRYPTED_LLM_URL")
encrypted_api_key = os.getenv("ENCRYPTED_API_KEY")
encrypted_embedding_url = os.getenv("ENCRYPTED_EMBEDDING_URL")

# 读取加密密钥
key_file_path = "secret_new.key"
try:
    with open(key_file_path, "rb") as key_file:
        key = key_file.read()
    cipher_suite = Fernet(key)
except FileNotFoundError:
    logging.error(f"Key file not found: {key_file_path}")
    raise

# 解密API URL和API密钥
try:
    llm_url = cipher_suite.decrypt(encrypted_llm_url.encode()).decode()
    api_key = cipher_suite.decrypt(encrypted_api_key.encode()).decode()
    embedding_url = cipher_suite.decrypt(encrypted_embedding_url.encode()).decode()
except Exception as e:
    logging.error(f"Decryption failed: {e}")
    raise ValueError(f"Decryption failed: {e}")

def refine_prompt(content):
    llm_payload = {
        "model": "Qwen/Qwen2-72B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": f"你是一个严格遵循指令的大模型，请用200字以内中文总结以下内容：{content}"
            }
        ]
    }
    llm_headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    try:
        response = requests.post(llm_url, json=llm_payload, headers=llm_headers)
        response.raise_for_status()
        refined_prompt = response.json().get('choices')[0]['message']['content'] if response.json().get('choices') else "No refined prompt returned."
        return refined_prompt
    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP Request failed in refine_prompt: {e}")
        return f"HTTP Request failed: {e}"
    except ValueError as e:
        logging.error(f"JSON Decode Error in refine_prompt: {e}")
        return f"JSON Decode Error: {e}"

def get_embedding(text):
    url = embedding_url
    payload = {
        "model": "BAAI/bge-large-zh-v1.5",
        "input": text
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}"
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        response_json = response.json()
        if not isinstance(response_json, dict):
            raise ValueError("Response is not a valid JSON object")
        data = response_json.get('data')
        if not data or not isinstance(data, list) or 'embedding' not in data[0]:
            raise ValueError("No embedding returned from API")
        embedding = data[0]['embedding']
        return np.array(embedding, dtype=np.float32)
    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP Request failed in get_embedding: {e}")
        raise ValueError(f"HTTP Request failed: {e}")
    except json.JSONDecodeError as e:
        logging.error(f"JSON Decode Error in get_embedding: {e}")
        raise ValueError(f"JSON Decode Error: {e}")
    except ValueError as e:
        logging.error(f"Error parsing JSON response in get_embedding: {e}")
        raise

def create_database():
    try:
        conn = sqlite3.connect('embeddings.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS embeddings
                     (folder_name TEXT, file_path TEXT, summary TEXT, embedding BLOB)''')
        conn.commit()
        return conn
    except sqlite3.Error as e:
        logging.error(f"SQLite error in create_database: {e}")
        raise

def load_and_encode_txt_files(folder_path, conn, processed_files):
    c = conn.cursor()
    folder_name = os.path.basename(folder_path)
    files_processed = 0
    new_embeddings = []  # 用于存储新生成的嵌入向量
    unprocessed_files = []  # 用于存储未处理文件的列表

    # 获取文件夹中的所有文件
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    # 过滤出尚未处理的文件
    for file_name in all_files:
        full_file_path = os.path.join(folder_path, file_name)
        if full_file_path not in processed_files:
            unprocessed_files.append(file_name)

    if not unprocessed_files:
        logging.info(f"No new files to process in {folder_name}.")
        return new_embeddings  # 如果没有未处理的文件，直接返回空列表

    # 假设 embedding 模型的最大输入长度为 512（你需要根据实际模型设置）
    max_embedding_input_length = 512
    max_refine_attempts = 3  # 最大尝试次数

    # 处理未处理的文件
    for file_name in unprocessed_files:
        file_path = os.path.join(folder_path, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            summary = refine_prompt(content)
            attempts = 1

            # 检查生成的摘要是否超过模型输入限制
            while len(summary) > max_embedding_input_length and attempts < max_refine_attempts:
                logging.info(f"Summary too long for embedding model (length {len(summary)}), refining again. Attempt {attempts}")
                summary = refine_prompt(content)
                attempts += 1

            if len(summary) > max_embedding_input_length:
                logging.warning(f"Summary still too long after {max_refine_attempts} attempts, truncating to fit model input length.")
                summary = summary[:max_embedding_input_length]

            embedding = get_embedding(summary)
            c.execute("INSERT INTO embeddings VALUES (?, ?, ?, ?)",
                      (folder_name, file_path, summary, embedding.tobytes()))
            new_embeddings.append(embedding)  # 将新嵌入向量添加到列表中
            files_processed += 1
        except Exception as e:
            logging.error(f"Error processing file {file_name}: {e}")

    conn.commit()
    logging.info(f"Processed {files_processed} new files in {folder_path}")
    return new_embeddings  # 返回新生成的嵌入向量列表


def load_embeddings_from_db(conn):
    try:
        c = conn.cursor()
        c.execute("SELECT folder_name, file_path, embedding FROM embeddings")
        results = c.fetchall()
        file_paths = [(row[0], row[1]) for row in results]
        embeddings = [np.frombuffer(row[2], dtype=np.float32) for row in results]
        return file_paths, np.array(embeddings)
    except sqlite3.Error as e:
        logging.error(f"SQLite error in load_embeddings_from_db: {e}")
        raise

def create_or_update_faiss_index(faiss_index, embeddings, new_vectors=None):
    if new_vectors is not None:
        embeddings = np.vstack((embeddings, new_vectors))
    
    d = embeddings.shape[1]
    if gpu_available:
        try:
            res = faiss.StandardGpuResources()
            faiss_index = faiss.GpuIndexFlatIP(res, d)
            print("Successfully created GPU index")
        except Exception as e:
            print(f"Error creating GPU index: {e}")
            print("Falling back to CPU index")
            faiss_index = faiss.IndexFlatIP(d)
    else:
        faiss_index = faiss.IndexFlatIP(d)
    
    faiss.normalize_L2(embeddings)
    faiss_index.add(embeddings)
    return faiss_index

def save_faiss_index(index, file_path):
    try:
        if gpu_available and isinstance(index, faiss.GpuIndex):
            index = faiss.index_gpu_to_cpu(index)
        faiss.write_index(index, file_path)
        logging.info(f"Faiss index saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving Faiss index: {e}")
        raise

def load_faiss_index(file_path):
    try:
        index = faiss.read_index(file_path)
        if gpu_available:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                print("Successfully loaded index to GPU")
            except Exception as e:
                print(f"Error loading index to GPU: {e}")
                print("Using CPU index")
        logging.info(f"Faiss index loaded from {file_path}")
        return index
    except Exception as e:
        logging.error(f"Error loading Faiss index: {e}")
        raise

def query_faiss_index(index, file_paths, query, conn, top_k=1, similarity_threshold=0.1):
    query_embedding = get_embedding(query)
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    faiss.normalize_L2(query_embedding)
    D, I = index.search(query_embedding, top_k)
    
    if D[0][0] > similarity_threshold:
        folder_name, file_path = file_paths[I[0][0]]
        c = conn.cursor()
        c.execute("SELECT summary FROM embeddings WHERE file_path = ?", (file_path,))
        result = c.fetchone()
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

def delete_file_from_db(conn, file_path):
    try:
        c = conn.cursor()
        c.execute("DELETE FROM embeddings WHERE file_path = ?", (file_path,))
        conn.commit()
        logging.info(f"Deleted file {file_path} from database")
    except sqlite3.Error as e:
        logging.error(f"SQLite error in delete_file_from_db: {e}")
        raise

def delete_folder_from_db(conn, folder_name):
    try:
        c = conn.cursor()
        c.execute("DELETE FROM embeddings WHERE folder_name = ?", (folder_name,))
        conn.commit()
        logging.info(f"Deleted folder {folder_name} from database")
    except sqlite3.Error as e:
        logging.error(f"SQLite error in delete_folder_from_db: {e}")
        raise

# 在主函数中，当你处理完新文件并生成新向量时，直接添加到现有索引中
def main():
    folder_paths = ['c-api', 'extending', 'tutorial', 'library']
    index_file_path = 'faiss_index.bin'

    try:
        conn = create_database()

        # 加载现有的索引文件和数据库记录
        if os.path.exists(index_file_path):
            faiss_index = load_faiss_index(index_file_path)
            file_paths, embeddings = load_embeddings_from_db(conn)
            logging.info("Index loaded from file.")
        else:
            faiss_index = None
            file_paths, embeddings = [], []

        new_vectors = []

        # 检查并处理每个文件夹中的文件
        for folder_path in folder_paths:
            processed_files = [path for _, path in file_paths]
            new_vectors_for_folder = load_and_encode_txt_files(folder_path, conn, processed_files)
            new_vectors.extend(new_vectors_for_folder)  # 将新向量加入到新的列表中

        # 如果有新向量，直接添加到索引中或重建索引
        if new_vectors or faiss_index is None:
            faiss_index = create_or_update_faiss_index(faiss_index, embeddings, np.array(new_vectors))
            save_faiss_index(faiss_index, index_file_path)
            logging.info("Index updated with new vectors or rebuilt.")

        while True:
            print("\n1. 查询")
            print("2. 删除文件")
            print("3. 删除文件夹")
            print("4. 退出")
            choice = input("请选择操作 (1/2/3/4): ")

            if choice == '1':
                user_input = input("请输入您的查询: ")
                related_content, file_path, folder_name, summary = query_faiss_index(faiss_index, file_paths, user_input, conn, similarity_threshold=0.5)

                if file_path:
                    print(f"找到相关内容，来自 {folder_name} 文件夹中的文件路径 {file_path}:")
                    print(f"摘要: {summary}")
                    print("完整内容:")
                    print(related_content)
                else:
                    print(related_content)

            elif choice == '2':
                file_path = input("请输入要删除的文件路径: ")
                delete_file_from_db(conn, file_path)
                print(f"文件 {file_path} 已从数据库中删除")
                # 处理删除后需要更新索引的情况
                file_paths, embeddings = load_embeddings_from_db(conn)
                if len(embeddings) > 0:
                    faiss_index = create_or_update_faiss_index(faiss_index, embeddings)
                    save_faiss_index(faiss_index, index_file_path)
                    logging.info("Index updated after deletion.")
                else:
                    logging.warning("No embeddings found to update index.")

            elif choice == '3':
                folder_name = input("请输入要删除的文件夹名: ")
                delete_folder_from_db(conn, folder_name)
                print(f"文件夹 {folder_name} 已从数据库中删除")
                # 处理删除后需要更新索引的情况
                file_paths, embeddings = load_embeddings_from_db(conn)
                if len(embeddings) > 0:
                    faiss_index = create_or_update_faiss_index(faiss_index, embeddings)
                    save_faiss_index(faiss_index, index_file_path)
                    logging.info("Index updated after deletion.")
                else:
                    logging.warning("No embeddings found to update index.")

            elif choice == '4':
                print("退出程序")
                break

            else:
                print("无效的选择，请重新输入")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    main()
