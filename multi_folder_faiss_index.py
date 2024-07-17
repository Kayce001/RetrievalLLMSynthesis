import os
import requests
import faiss
import numpy as np
from cryptography.fernet import Fernet
import json
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# 从环境变量中获取加密密钥和 API 信息
encrypted_llm_url = os.getenv("ENCRYPTED_LLM_URL")
encrypted_api_key = os.getenv("ENCRYPTED_API_KEY")
encrypted_embedding_url = os.getenv("ENCRYPTED_EMBEDDING_URL")

# 读取加密密钥
key_file_path = "secret_new.key"  # 确保路径正确
with open(key_file_path, "rb") as key_file:
    key = key_file.read()
cipher_suite = Fernet(key)

# 解密API URL和API密钥
try:
    llm_url = cipher_suite.decrypt(encrypted_llm_url.encode()).decode()
    api_key = cipher_suite.decrypt(encrypted_api_key.encode()).decode()
    embedding_url = cipher_suite.decrypt(encrypted_embedding_url.encode()).decode()
except Exception as e:
    raise ValueError(f"Decryption failed: {e}")

# 使用大模型生成总结文本
def refine_prompt(content):
    llm_payload = {
        "model": "Qwen/Qwen2-72B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": f"你是一个严格遵循指令的大模型，请用100字以内中文总结以下内容：{content}"
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
        response.raise_for_status()  # 会抛出HTTP错误状态码的异常
        refined_prompt = response.json().get('choices')[0]['message']['content'] if response.json().get('choices') else "No refined prompt returned."
        return refined_prompt
    except requests.exceptions.RequestException as e:
        return f"HTTP Request failed: {e}"
    except ValueError as e:
        return f"JSON Decode Error: {e}"

# 调用 API获取嵌入
def get_embedding(text):
    url = embedding_url
    payload = {
        "model": "BAAI/bge-large-zh-v1.5",
        "input": text
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}"  # 请确保使用真实的API密钥
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        #print("Response content:", response.text)
        response.raise_for_status()
        try:
            response_json = response.json()
            if not isinstance(response_json, dict):
                raise ValueError("Response is not a valid JSON object")
            data = response_json.get('data')
            if not data or not isinstance(data, list) or 'embedding' not in data[0]:
                raise ValueError("No embedding returned from API")
            embedding = data[0]['embedding']
            return np.array(embedding, dtype=np.float32)
        except json.JSONDecodeError:
            print(f"Error decoding JSON response: {response.text}")
            raise ValueError("JSON Decode Error")
        except ValueError as e:
            print(f"Error parsing JSON response: {response.text}")
            raise e
    except requests.exceptions.RequestException as e:
        print(f"HTTP Request failed: {e}")
        raise ValueError(f"HTTP Request failed: {e}")

# 读取文件夹中的所有txt文件并进行向量化
def load_and_encode_txt_files(folder_path, processed_files):
    folder_name = os.path.basename(folder_path)  # 获取文件夹名称
    file_paths = []
    summarized_contents = []
    embeddings = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt') and file_name not in processed_files:  # 仅处理未处理过的文件
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            summary = refine_prompt(content)
            #print("summary: ",summary)
            try:
                embedding = get_embedding(summary)
                file_paths.append((folder_name, file_path))  # 保存文件夹名称和文件路径
                summarized_contents.append(summary)
                embeddings.append(embedding)
            except ValueError as e:
                print(f"Error processing file {file_name}: {e}")
    if len(embeddings) == 0:
        return file_paths, summarized_contents, np.array([])  # 返回空的嵌入数组以避免错误

    return file_paths, summarized_contents, np.array(embeddings, dtype=np.float32)

# 创建Faiss索引并添加向量
def create_faiss_index(embeddings):
    d = embeddings.shape[1]  # 向量的维度
    index = faiss.IndexFlatIP(d)  # 使用内积度量，余弦相似度
    faiss.normalize_L2(embeddings)  # 对向量进行L2归一化，以便使用余弦相似度
    index.add(embeddings)
    return index

# 将Faiss索引保存到磁盘
def save_faiss_index(index, file_path):
    faiss.write_index(index, file_path)

# 从磁盘加载Faiss索引
def load_faiss_index(file_path):
    return faiss.read_index(file_path)

# 保存文件路径和总结内容
def save_metadata(file_paths, summarized_contents, metadata_path):
    metadata = {
        "file_paths": file_paths,
        "summarized_contents": summarized_contents
    }
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False)

# 加载文件路径和总结内容
def load_metadata(metadata_path):
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return metadata['file_paths'], metadata['summarized_contents']

def query_faiss_index(index, file_paths, query, top_k=1, similarity_threshold=0.1):
    query_embedding = get_embedding(query)
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)  # 确保 query_embedding 是二维数组
    faiss.normalize_L2(query_embedding)  # 对查询向量进行L2归一化
    D, I = index.search(query_embedding, top_k)  # 查找最相似的top_k个向量
    
    # 检查相似度是否超过阈值
    if D[0][0] > similarity_threshold:
        folder_name, file_path = file_paths[I[0][0]]  # 解包文件夹名称和文件路径
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content, os.path.basename(file_path), folder_name  # 返回文件夹名称
    else:
        return "No relevant content found.", None, None


# 示例文件夹路径和索引文件路径
folder_paths = ['api']  # 可以包含多个文件夹路径的列表
index_file_path = 'faiss_index.bin'
metadata_file_path = 'metadata.json'

# 检查索引文件和元数据文件是否存在
if not os.path.exists(index_file_path) or not os.path.exists(metadata_file_path):
    # 如果索引文件或元数据文件不存在，则进行初始创建和保存
    file_paths, summarized_contents, embeddings = [], [], []
    for folder_path in folder_paths:
        fp, sc, em = load_and_encode_txt_files(folder_path, [])
        file_paths.extend(fp)
        summarized_contents.extend(sc)
        embeddings.extend(em)
    embeddings = np.array(embeddings, dtype=np.float32)
    if len(embeddings) > 0:
        faiss_index = create_faiss_index(embeddings)
        save_faiss_index(faiss_index, index_file_path)
        save_metadata(file_paths, summarized_contents, metadata_file_path)
        print("Index and metadata created and saved.")
else:
    # 如果索引文件和元数据文件存在，则加载索引和元数据
    faiss_index = load_faiss_index(index_file_path)
    file_paths, summarized_contents = load_metadata(metadata_file_path)
    print("Index and metadata loaded from file.")
    
    # 处理新的文件
    for folder_path in folder_paths:
        new_file_paths, new_summarized_contents, new_embeddings = load_and_encode_txt_files(folder_path, [os.path.basename(path) for _, path in file_paths])  # 传递已处理文件列表
        if len(new_embeddings) > 0:
            faiss.normalize_L2(new_embeddings)
            faiss_index.add(new_embeddings)
            file_paths.extend(new_file_paths)
            summarized_contents.extend(new_summarized_contents)
            save_faiss_index(faiss_index, index_file_path)
            save_metadata(file_paths, summarized_contents, metadata_file_path)
            print("New files processed and index updated.")



# 用户查询
user_input = "介绍抽象对象层"

# 获取相关内容
related_content, file_name, folder_name = query_faiss_index(faiss_index, file_paths, user_input, similarity_threshold=0.5)
if file_name:
    print(f"Content from {file_name} in {folder_name}:")
    print(related_content)
else:
    print(related_content)
