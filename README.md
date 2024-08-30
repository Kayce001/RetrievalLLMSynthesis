# RetrievalLLMSynthesis

欢迎来到 RetrievalLLMSynthesis 项目，本项目结合了大型语言模型（LLM）与检索增强生成（RAG）技术，致力于提高文本生成的相关性和准确性。

## 项目概览

RetrievalLLMSynthesis 旨在通过整合先进的语言模型和检索技术，提高文本内容生成的质量和效率。
Complete_Chat_History_Manager.py这个文件实现了一个基于 Gradio 界面的聊天系统，结合了大型语言模型（LLM）与检索增强生成（RAG）技术，用于高效地生成基于查询的相关答案，并支持对话的创建、保存、加载和删除功能。
gpu_accelerated_sqlite_retrieval.py这个文件实现了一个基于嵌入向量和 FAISS 检索的系统，用于处理、存储、查询和管理文本数据的嵌入表示，并提供了查询和删除操作的命令行界面。

## 快速开始

### 1. 克隆仓库、创建并激活虚拟环境、安装依赖

```bash
git clone https://github.com/Kayce001/RetrievalLLMSynthesis.git
cd RetrievalLLMSynthesis

conda create -n rag_env python=3.8
conda activate rag_env

pip install -r requirements.txt
