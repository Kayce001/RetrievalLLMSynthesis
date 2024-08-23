#添加了删除历史对话功能
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
from datetime import datetime

# 设置日志
logging.basicConfig(filename='rag_task.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# GPU 检查
gpu_available = torch.cuda.is_available()

# 全局变量
loaded_faiss_index = None
loaded_embeddings = None
loaded_file_paths = None
chat_history_store = []
all_conversations = []
current_conversation_index = None
is_responding = False
conversation_list = None

# 加载 Faiss 索引
def load_faiss_index(file_path):
    global loaded_faiss_index
    if loaded_faiss_index is not None:
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
        loaded_faiss_index = index
        logging.info(f"Faiss index loaded from {file_path}")
        logging.info(f"Time taken to load Faiss index: {time.time() - start_time:.2f} seconds")
        return loaded_faiss_index
    except Exception as e:
        logging.error(f"Error loading Faiss index: {e}")
        raise

# 从数据库加载嵌入
def load_embeddings_from_db(conn):
    global loaded_embeddings, loaded_file_paths
    if loaded_embeddings is not None and loaded_file_paths is not None:
        print("Using pre-loaded embeddings from DB")
        return loaded_file_paths, loaded_embeddings
    try:
        start_time = time.time()
        c = conn.cursor()
        c.execute("SELECT folder_name, file_path, embedding FROM embeddings")
        results = c.fetchall()
        file_paths = [(row[0], row[1]) for row in results]
        embeddings = [np.frombuffer(row[2], dtype=np.float32) for row in results]
        loaded_file_paths = file_paths
        loaded_embeddings = np.array(embeddings)
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
        faiss_index = load_faiss_index(index_file_path)
        file_paths, embeddings = load_embeddings_from_db(conn)

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

def user(user_message, history):
    return "", history + [[user_message, None]]

def bot(history):
    global is_responding, conversation_list
    is_responding = True
    user_message = history[-1][0]
    bot_message = ""
    try:
        for partial_message in rag_system(user_message, history[:-1]):
            if not is_responding:
                break  # 如果被中断，则停止生成
            bot_message = partial_message
            history[-1][1] = bot_message
            yield history, gr.update(interactive=True, value="中断")
        
        # 回复完成后，禁用提交按钮
        yield history, gr.update(interactive=False, value="提交")
    except Exception as e:
        logging.error(f"生成对话时发生错误: {e}")
        yield history, gr.update(interactive=False, value="提交")
    finally:
        is_responding = False

    # 保存对话
    save_conversation(history)
    if conversation_list:
        conversation_list.choices = load_conversations()

def load_selected_conversation(selected):
    global current_conversation_index, all_conversations
    if selected:
        for i, conv in enumerate(all_conversations):
            if f"{conv['title']} ({conv['timestamp']})" == selected:
                current_conversation_index = i
                return conv["conversation"]
    current_conversation_index = None
    return []

def create_new_conversation():
    global current_conversation_index, is_responding
    current_conversation_index = None
    is_responding = False
    return [], gr.update(choices=load_conversations(), value=None), gr.update(interactive=False, value="提交"), ""

def save_conversation(conversation):
    global current_conversation_index, all_conversations
    if current_conversation_index is not None:
        all_conversations[current_conversation_index]["conversation"] = conversation
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        title = conversation[0][0][:20] if conversation else "新对话"
        all_conversations.insert(0, {
            "timestamp": timestamp,
            "title": title,
            "conversation": conversation
        })
        current_conversation_index = 0
    
    with open("conversations.json", "w", encoding="utf-8") as f:
        json.dump(all_conversations, f, ensure_ascii=False, indent=2)

def load_conversations():
    global all_conversations
    if os.path.exists("conversations.json"):
        with open("conversations.json", "r", encoding="utf-8") as f:
            all_conversations = json.load(f)
    return [f"{conv['title']} ({conv['timestamp']})" for conv in all_conversations]

def delete_conversation(selected):
    global all_conversations, current_conversation_index
    if selected:
        for i, conv in enumerate(all_conversations):
            if f"{conv['title']} ({conv['timestamp']})" == selected:
                del all_conversations[i]
                current_conversation_index = None
                with open("conversations.json", "w", encoding="utf-8") as f:
                    json.dump(all_conversations, f, ensure_ascii=False, indent=2)
                break
    return gr.update(choices=load_conversations(), value=None), []

def process_input(user_message, history):
    global is_responding
    if is_responding:
        is_responding = False
        return history, gr.update(interactive=False, value="提交"), ""  # 禁用提交按钮
    else:
        is_responding = True
        new_history = history + [[user_message, None]]
        return new_history, gr.update(interactive=False, value="中断"), ""
    
def check_input(input_text):
    # 直接根据输入框的原始内容更新按钮的状态
    return gr.update(interactive=bool(input_text.strip()))

# Gradio 界面设计
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            conversation_list = gr.Dropdown(choices=load_conversations(), label="历史对话")
            delete_btn = gr.Button("删除对话")
            new_conversation_btn = gr.Button("创建新对话")
        
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=600)  # 直接在初始化时设置高度
            with gr.Row():
                msg = gr.Textbox(label="输入消息", lines=3, max_lines=3)  # 固定输入框高度，并添加滚动条
                submit_btn = gr.Button("提交", size="sm", interactive=False)  # 初始化为不可点击

    # 动态控制提交按钮状态，每次输入内容变化时检查输入并更新按钮状态
    msg.change(check_input, inputs=msg, outputs=submit_btn)


    # 提交按钮点击事件
    submit_btn.click(
        process_input, [msg, chatbot], [chatbot, submit_btn, msg], queue=False
    ).then(
        bot, chatbot, [chatbot, submit_btn]
    )

    # 加载历史对话
    conversation_list.change(load_selected_conversation, conversation_list, chatbot)
    
    # 创建新对话
    new_conversation_btn.click(create_new_conversation, None, [chatbot, conversation_list, submit_btn, msg])

    # 删除对话按钮点击事件
    delete_btn.click(delete_conversation, conversation_list, [conversation_list, chatbot])

if __name__ == "__main__":
    demo.launch()
