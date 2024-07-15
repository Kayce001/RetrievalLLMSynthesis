import os
import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from cryptography.fernet import Fernet
import json

# 加载预训练的SentenceTransformer模型
model = SentenceTransformer('moka-ai/m3e-base')

# 读取加密密钥
key_file_path = "secret.key"  # 确保路径正确
with open(key_file_path, "rb") as key_file:
    key = key_file.read()
cipher_suite = Fernet(key)

# 解密API URL和API密钥
encrypted_llm_url = "gAAAAABmi8sEdjn42gY2Jk2KpL-a3WLO2HtNho8GC_U3h6tTnDpOT8FHDXJeJFPaWagA-LnF2CuTgzXV-bvL1oy77mPMdRb3rfq2h2c6W_15zlE_dHkumpSOgjCQHqUhAJYX0N-5ZBxE"
encrypted_api_key = "gAAAAABmi8sE3tbc0rV7W7NwjHLB6lwyVs4GvPc2l6DmbywDQtri1zSQCN7PFOScxaqt3CW1azSbZxOsbfwu64jVNuDrx-xTo5Zg08tC4lba9gRP0TpX1s5-reD_2ppxw6KYtKSgbQco26wIYvfYWncu3oH-fptFEQ=="

try:
    llm_url = cipher_suite.decrypt(encrypted_llm_url.encode()).decode()
    api_key = cipher_suite.decrypt(encrypted_api_key.encode()).decode()
except Exception as e:
    raise ValueError(f"Decryption failed: {e}")

# 使用大模型生成总结文本
def refine_prompt(content):
    llm_payload = {
        "model": "Qwen/Qwen2-72B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": f"请用100字以内中文总结以下内容：{content}"
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

# 读取文件夹中的所有txt文件并进行向量化
def load_and_encode_txt_files(folder_path):
    file_paths = []
    summarized_contents = []
    embeddings = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            summary = refine_prompt(content)
            embedding = model.encode([summary])
            file_paths.append(file_path)
            summarized_contents.append(summary)
            embeddings.append(embedding[0])
    return file_paths, summarized_contents, np.array(embeddings)

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


# 根据用户查询提取相关内容
def query_faiss_index(index, file_paths, query, top_k=1, similarity_threshold=0.1):
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)  # 对查询向量进行L2归一化
    D, I = index.search(query_embedding, top_k)  # 查找最相似的top_k个向量
    
    # 检查相似度是否超过阈值
    if D[0][0] > similarity_threshold:
        file_path = file_paths[I[0][0]]
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content, os.path.basename(file_path)
    else:
        return "No relevant content found.", None


# 示例文件夹路径和索引文件路径
folder_path = '/content/c_api'
index_file_path = 'faiss_index.bin'
metadata_file_path = 'metadata.json'

# 检查索引文件和元数据文件是否存在
if not os.path.exists(index_file_path) or not os.path.exists(metadata_file_path):
    # 如果索引文件或元数据文件不存在，则进行初始创建和保存
    file_paths, summarized_contents, embeddings = load_and_encode_txt_files(folder_path)
    faiss_index = create_faiss_index(embeddings)
    save_faiss_index(faiss_index, index_file_path)
    save_metadata(file_paths, summarized_contents, metadata_file_path)
    print("Index and metadata created and saved.")
else:
    # 如果索引文件和元数据文件存在，则加载索引和元数据
    faiss_index = load_faiss_index(index_file_path)
    file_paths, summarized_contents = load_metadata(metadata_file_path)
    print("Index and metadata loaded from file.")

# 检查索引中向量的数量
print("Number of vectors in the index:", faiss_index.ntotal)

# 用户查询
user_input = "PyCode_GetFirstFree(PyCodeObject *co)"
#user_query = refine_prompt(user_input)
#print("user_query: ",user_query)

# 获取相关内容
related_content, file_name = query_faiss_index(faiss_index, file_paths, user_input, similarity_threshold=0.7)
if file_name:
    print(f"Content from {file_name}:")
    print(related_content)
else:
    print(related_content)
