import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.postprocessor import SentenceTransformerRerank
import json
import os
import pickle
from typing import List, Dict, Any, Optional

class RAGLegalSystem:
    def __init__(self, 
                 base_model_path="../model/lora_7b_total_no_comm_data_f16",
                 embedding_model_path="../model/Qwen3-Embedding-0.6B", 
                 rerank_model_path="../model/Qwen3-Reranker-0.6B",
                 document_path="../data/data_json_rag_acc"):
        
        self.base_model_path = base_model_path
        self.embedding_model_path = embedding_model_path
        self.rerank_model_path = rerank_model_path
        self.document_path = document_path
        
        # 初始化组件
        self.tokenizer = None
        self.model = None
        self.query_engine = None
        self.index = None
    
    def setup_base_model(self):
        """设置基础模型"""
        print("Loading base model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        
        # 设置padding token以避免批处理错误
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        
        # 为 LlamaIndex 创建 LLM 包装器
        llm = HuggingFaceLLM(
            model_name=self.base_model_path,
            tokenizer_name=self.base_model_path,
            context_window=8192,
            max_new_tokens=512,
            generate_kwargs={
                "temperature": 0.1,
                "do_sample": True,
            },
            model_kwargs={
                "torch_dtype": torch.float16,
            }
        )
        
        Settings.llm = llm
        return llm
    
    def setup_embedding_model(self):
        """设置 Embedding 模型"""
        print("Loading embedding model...")
        embed_model = HuggingFaceEmbedding(
            model_name=self.embedding_model_path,
            trust_remote_code=True,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        Settings.embed_model = embed_model
        return embed_model
    
    def _extract_text_from_json(self, json_obj) -> str:
        """从 JSON 对象中提取文本内容，拼接所有字段的值"""
        if isinstance(json_obj, dict):
            text_parts = []
            for value in json_obj.values():
                if isinstance(value, str) and value.strip():
                    text_parts.append(value.strip())
            return " ".join(text_parts)
        elif isinstance(json_obj, str):
            return json_obj
        else:
            return ""
    
    def load_legal_documents(self) -> List[Document]:
        """加载法律文档，支持 .json 和 .jsonl 文件"""
        print(f"Loading documents from {self.document_path}...")
        documents = []
        
        # 遍历 JSON 文件
        for filename in os.listdir(self.document_path):
            if not filename.endswith(('.json', '.jsonl')):
                continue

            file_path = os.path.join(self.document_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    if filename.endswith('.jsonl'):
                        # 处理 .jsonl 文件 (每行一个JSON对象)
                        for i, line in enumerate(f):
                            if line.strip():
                                data = json.loads(line)
                                text_content = self._extract_text_from_json(data)
                                if text_content:
                                    doc = Document(
                                        text=text_content,
                                        metadata={
                                            "source": filename,
                                            "line": i + 1,
                                            "file_path": file_path
                                        }
                                    )
                                    documents.append(doc)
                    else:  # .json file
                        # 处理 .json 文件 (单个JSON对象或JSON对象列表)
                        data = json.load(f)
                        
                        items_to_process = data if isinstance(data, list) else [data]

                        for i, item in enumerate(items_to_process):
                            text_content = self._extract_text_from_json(item)
                            if text_content:
                                metadata = {
                                    "source": filename,
                                    "file_path": file_path
                                }
                                if isinstance(data, list):
                                    metadata["index"] = i
                                doc = Document(
                                    text=text_content,
                                    metadata=metadata
                                )
                                documents.append(doc)
                            
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
        
        print(f"Loaded {len(documents)} documents")
        return documents
    
    def setup_rag_system(self, similarity_top_k=10, rerank_top_n=5, save_index=True, index_path="legal_index.pkl"):
        """设置 RAG 系统"""
        print("Setting up RAG system...")
        
        # 1. 设置模型
        self.setup_base_model()
        self.setup_embedding_model()
        
        # 2. 加载文档或从缓存加载索引
        if os.path.exists(index_path):
            print(f"Loading index from {index_path}...")
            try:
                with open(index_path, 'rb') as f:
                    self.index = pickle.load(f)
                print("Index loaded successfully!")
            except Exception as e:
                print(f"Error loading index: {e}. Building new index...")
                self._build_new_index(similarity_top_k, rerank_top_n, save_index, index_path)
        else:
            self._build_new_index(similarity_top_k, rerank_top_n, save_index, index_path)
        
        # 3. 设置检索器和查询引擎
        self._setup_retriever_and_engine(similarity_top_k, rerank_top_n)
        
        print("RAG system setup complete!")
    
    def _build_new_index(self, similarity_top_k, rerank_top_n, save_index, index_path):
        """构建新的索引"""
        # 加载文档
        documents = self.load_legal_documents()
        if not documents:
            raise ValueError("No documents loaded!")
        
        # 创建向量索引
        print("Building vector index...")
        self.index = VectorStoreIndex.from_documents(documents)
        
        # 保存索引
        if save_index:
            print(f"Saving index to {index_path}...")
            with open(index_path, 'wb') as f:
                pickle.dump(self.index, f)
            print("Index saved successfully!")
    
    def _setup_retriever_and_engine(self, similarity_top_k, rerank_top_n):
        """设置检索器和查询引擎"""
        # 设置检索器
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=similarity_top_k
        )
        
        # 设置 Rerank 模型
        try:
            rerank = SentenceTransformerRerank(
                model=self.rerank_model_path,
                top_n=rerank_top_n,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            post_processors = [
                SimilarityPostprocessor(similarity_cutoff=0.7),
                rerank
            ]
        except Exception as e:
            print(f"Rerank model loading failed, using similarity only: {e}")
            post_processors = [SimilarityPostprocessor(similarity_cutoff=0.7)]
        
        # 创建查询引擎
        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=post_processors
        )
    
    def get_relevant_context(self, question: str, max_context_length=4096) -> str:
        """获取相关上下文"""
        try:
            # 使用查询引擎检索相关信息
            response = self.query_engine.query(question)
            
            # 提取相关文档的文本
            context_parts = []
            total_length = 0
            
            for node in response.source_nodes:
                text = node.node.text.strip()
                if total_length + len(text) <= max_context_length:
                    context_parts.append(text)
                    total_length += len(text)
                else:
                    # 截取剩余长度
                    remaining = max_context_length - total_length
                    if remaining > 100:  # 只有在剩余长度足够时才添加
                        context_parts.append(text[:remaining] + "...")
                    break
            
            return "\n\n".join(context_parts)
        
        except Exception as e:
            print(f"Error retrieving context: {e}")
            # 尝试使用备用方法检索
            try:
                # 简单的备用检索方法
                retriever = VectorIndexRetriever(
                    index=self.index,
                    similarity_top_k=1  # 只检索最相似的一个文档
                )
                nodes = retriever.retrieve(question)
                if nodes:
                    return nodes[0].node.text
            except Exception as backup_error:
                print(f"Backup retrieval also failed: {backup_error}")
            return ""
    
    def generate_answer_with_rag(self, question: str, options: dict) -> str:
        """使用 RAG 生成答案"""
        # 1. 获取相关上下文
        context = self.get_relevant_context(question)
        
        # 2. 构建增强的提示
        if context:
            prompt = f"""
根据以下法律知识和背景信息，回答这个法律单项选择题。请仔细阅读相关法律条文，然后选择正确的选项。

相关法律知识：
{context}

问题：{question}
A：{options['A']}
B：{options['B']}
C：{options['C']}
D：{options['D']}

请根据上述法律知识选择正确答案，只需要回答正确选项即可，不需要进行分析,回答格式：B
"""
        else:
            # 如果没有检索到相关上下文，使用原始提示
            prompt = f"""
这是一个关于法律问题的单项选择题,请根据题目选择正确的选项，只需要回答正确选项即可，不需要进行分析,回答格式：D

问题：{question}
A：{options['A']}
B：{options['B']}
C：{options['C']}
D：{options['D']}
"""
        
        # 3. 使用原有的生成逻辑
        messages = [{"role": "user", "content": prompt}]
        print(messages)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=50,
            temperature=0.1,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        content = self.tokenizer.decode(
            generated_ids[0][model_inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        return content
    
    def save_config(self, file_path="rag_config.json"):
        """保存RAG系统配置，而不是整个系统"""
        config = {
            "base_model_path": self.base_model_path,
            "embedding_model_path": self.embedding_model_path,
            "rerank_model_path": self.rerank_model_path,
            "document_path": self.document_path,
            "has_index": self.index is not None
        }
        
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"RAG system configuration saved to {file_path}")
        return config

# 修改后的保存和加载函数
def save_rag_system(rag_system, config_path="rag_config.json", index_path="legal_index.pkl"):
    """保存RAG系统配置和索引（不保存整个系统）"""
    try:
        # 1. 保存配置
        config = rag_system.save_config(config_path)
        
        # 2. 确保索引已保存
        if rag_system.index and not os.path.exists(index_path):
            with open(index_path, 'wb') as f:
                pickle.dump(rag_system.index, f)
            print(f"Index saved to {index_path}")
            
        return True
    except Exception as e:
        print(f"Error saving RAG system: {e}")
        return False

def load_rag_system(config_path="rag_config.json", index_path="legal_index.pkl"):
    """从配置加载RAG系统"""
    try:
        # 1. 加载配置
        if not os.path.exists(config_path):
            print(f"Configuration file {config_path} not found.")
            return None
            
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 2. 创建RAG系统
        rag_system = RAGLegalSystem(
            base_model_path=config.get("base_model_path", "../model/lora_7b_total_no_comm_data_f16"),
            embedding_model_path=config.get("embedding_model_path", "../model/Qwen3-Embedding-0.6B"),
            rerank_model_path=config.get("rerank_model_path", "../model/Qwen3-Reranker-0.6B"),
            document_path=config.get("document_path", "../data/data_json_rag_acc")
        )
        
        # 3. 设置RAG系统
        rag_system.setup_rag_system(
            save_index=False,  # 不需要再次保存索引
            index_path=index_path
        )
        
        print(f"RAG system loaded from configuration {config_path}")
        return rag_system
        
    except Exception as e:
        print(f"Error loading RAG system: {e}")
        return None

# 初始化RAG系统的示例代码
if __name__ == "__main__":
    # 创建评测器
    rag_system = RAGLegalSystem(
        base_model_path="../model/lora_7b_total_no_comm_data_f16",
        embedding_model_path="../model/Qwen3-Embedding-0.6B",
        rerank_model_path="../model/Qwen3-Reranker-0.6B",
        document_path="../data/data_json_rag_acc"
    )
    
    # 设置 RAG 系统
    rag_system.setup_rag_system(
        similarity_top_k=10,  # 第一阶段检索文档数
        rerank_top_n=5,       # 最终使用的文档数
        save_index=True,      # 保存索引以便重用
        index_path="legal_index.pkl"
    )
    
    # 保存RAG系统配置
    save_rag_system(rag_system, "rag_config.json", "legal_index.pkl")
    
    print("RAG system initialized and configuration saved successfully!") 