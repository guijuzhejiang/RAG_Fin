# RAG API - 技术架构文档

## 目录
1. [系统概述](#系统概述)
2. [架构组件](#架构组件)
3. [数据流与算法](#数据流与算法)
4. [数据库设计](#数据库设计)
5. [API端点](#api端点)
6. [核心算法](#核心算法)
7. [RAG方法论](#rag方法论)
8. [当前限制](#当前限制)
9. [改进机会](#改进机会)

---

## 系统概述

RAG API 是一个基于FastAPI和Supabase的检索增强生成系统，使用Python/FastAPI技术栈实现。该系统提供了完整的RAG功能，包括知识库管理、文档处理、向量搜索、智能对话等核心功能。

### 主要特性
- **知识库管理**: 创建和管理多个知识库，支持自定义分块参数
- **文档处理**: 多格式文档上传、解析、分块和向量化
- **向量搜索**: 基于语义相似度的智能检索
- **智能对话**: RAG增强的对话系统，支持上下文感知
- **文件管理**: 完整的文件上传、存储和管理功能
- **实时处理**: 异步文档处理和状态跟踪

---

## 架构组件

### 1. 核心服务层

#### `KnowledgeBaseService` (`services/knowledge_base_service.py`)
**功能**: 知识库业务逻辑管理器

**核心方法**:
```python
async def create_knowledge_base(kb_data: KnowledgeBaseCreate) -> KnowledgeBaseResponse
async def get_knowledge_base(kb_id: str) -> Optional[KnowledgeBaseResponse]
async def list_knowledge_bases(page, page_size, search) -> tuple[List, int]
async def update_knowledge_base(kb_id: str, kb_update: KnowledgeBaseUpdate) -> Optional[KnowledgeBaseResponse]
async def delete_knowledge_base(kb_id: str) -> bool
async def get_knowledge_base_stats(kb_id: str) -> Optional[KnowledgeBaseStats]
```

#### `DocumentService` (`services/document_service.py`)
**功能**: 文档生命周期管理

**核心方法**:
```python
async def create_document(doc_data: DocumentCreate) -> DocumentResponse
async def get_document(doc_id: str, include_content: bool = True) -> Optional[DocumentResponse]
async def list_documents(kb_id, page, page_size, search, status) -> tuple[List, int]
async def update_document_status(doc_id, status, progress, error_message, chunk_count) -> bool
async def process_document(doc_id: str) -> bool  # 文档分块处理
```

**数据库表**:
- `documents`: 文档元数据和内容存储

#### `ChunkService` (`services/chunk_service.py`)
**功能**: 文档块管理和向量搜索

**核心方法**:
```python
async def create_chunk(chunk_data: ChunkCreate) -> ChunkResponse
async def vector_search(search_request: VectorSearchRequest) -> List[VectorSearchResult]
async def update_chunk_embedding(chunk_id: str, embedding: List[float]) -> bool
async def get_chunks_by_document(document_id: str) -> List[ChunkResponse]
```

**数据库表**:
- `chunks`: 文档块内容和向量存储

#### `ConversationService` (`services/conversation_service.py`)
**功能**: 对话管理和RAG集成

**核心方法**:
```python
async def create_conversation(conv_data: ConversationCreate) -> ConversationResponse
async def chat(chat_request: ChatRequest) -> ChatResponse
async def add_message(conv_id: str, message_data: MessageCreate) -> MessageResponse
async def get_messages(conv_id: str, page: int, page_size: int) -> tuple[List, int]
```

**数据库表**:
- `conversations`: 对话配置和元数据
- `conversation_messages`: 对话消息历史

#### `FileService` (`services/file_service.py`)
**功能**: 文件存储和管理

**核心方法**:
```python
async def upload_file(file: UploadFile) -> FileResponse
async def get_file(file_id: str) -> Optional[FileResponse]
async def download_file(file_id: str) -> StreamingResponse
async def delete_file(file_id: str) -> bool
```

**数据库表**:
- `files`: 文件元数据和存储路径

### 2. 数据访问层

#### `SupabaseConnection` (`database/connection.py`)
**功能**: Supabase数据库连接管理

**核心方法**:
```python
def connect() -> None
def disconnect() -> None
async def health_check() -> bool
def get_db() -> Client  # 依赖注入
```

**连接配置**:
- **Supabase URL**: 项目连接地址
- **Supabase Key**: 公共匿名密钥
- **连接池**: 自动管理连接生命周期

### 3. API路由层

#### `KnowledgeBaseRouter` (`api/knowledge_base.py`)
**功能**: 知识库HTTP接口

**端点**:
- `POST /api/v1/knowledge-bases/`: 创建知识库
- `GET /api/v1/knowledge-bases/`: 获取知识库列表
- `GET /api/v1/knowledge-bases/{kb_id}`: 获取知识库详情
- `PUT /api/v1/knowledge-bases/{kb_id}`: 更新知识库
- `DELETE /api/v1/knowledge-bases/{kb_id}`: 删除知识库
- `GET /api/v1/knowledge-bases/{kb_id}/stats`: 获取统计信息

#### 其他路由模块
- `DocumentRouter` (`api/document.py`): 文档管理接口
- `ChunkRouter` (`api/chunk.py`): 块管理和搜索接口
- `ConversationRouter` (`api/conversation.py`): 对话系统接口
- `FileRouter` (`api/file.py`): 文件管理接口

---

## 数据流与算法

### 1. RAG系统架构概览

![RAG系统架构](images/flow_chart.png)

上图展示了完整的RAG系统工作流程：

- **索引阶段**：文档被加载、分割成块、转换为向量嵌入，然后存储到向量数据库中
- **检索与生成阶段**：处理用户查询，基于相似度搜索检索相关文档，LLM使用检索到的上下文生成回答

### 2. 文档处理流程

```
文件上传 → 内容提取 → 文本分块 → 向量生成 → 存储到数据库
                                    ↓
状态更新 ← 进度跟踪 ← 批量处理 ← 队列管理
```

### 3. 文档分块算法

```python
# 文档分块处理
def chunk_document(content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    将文档内容分割成固定大小的块
    
    Args:
        content: 文档内容
        chunk_size: 块大小（字符数）
        chunk_overlap: 重叠大小
    
    Returns:
        分块后的文本列表
    """
    chunks = []
    start = 0
    
    while start < len(content):
        end = start + chunk_size
        chunk = content[start:end]
        chunks.append(chunk)
        
        # 计算下一个块的起始位置（考虑重叠）
        start = end - chunk_overlap if end < len(content) else end
    
    return chunks
```

### 4. 向量搜索算法

```python
# 简化的向量搜索实现（当前使用文本匹配）
async def vector_search(query: str, kb_ids: List[str], top_k: int) -> List[VectorSearchResult]:
    """
    向量相似度搜索
    
    注意：当前实现使用文本匹配作为临时方案
    生产环境需要集成真实的向量搜索引擎
    """
    results = []
    
    for kb_id in kb_ids:
        # 文本搜索（临时实现）
        text_results = db.table("chunks").select(
            "id,content,metadata,document_id"
        ).eq("kb_id", kb_id).ilike("content", f"%{query}%").limit(top_k).execute()
        
        for item in text_results.data:
            # 计算简单的相似度分数
            similarity_score = calculate_text_similarity(query, item["content"])
            
            if similarity_score >= similarity_threshold:
                results.append(VectorSearchResult(
                    chunk_id=item["id"],
                    content=item["content"],
                    similarity_score=similarity_score,
                    metadata=item["metadata"]
                ))
    
    # 按相似度排序
    results.sort(key=lambda x: x.similarity_score, reverse=True)
    return results[:top_k]
```

### 5. RAG对话流程

```
用户输入 → 向量检索 → 上下文构建 → LLM生成 → 响应返回
                ↓            ↓          ↓
        相关文档块 → 提示词模板 → 带上下文的回答
```

```python
async def chat_with_rag(request: ChatRequest) -> ChatResponse:
    """RAG增强的对话处理"""
    
    # 1. 检索相关文档块
    if request.use_rag:
        search_request = VectorSearchRequest(
            query=request.message,
            kb_ids=conversation.kb_ids,
            top_k=request.top_k
        )
        context_chunks = await chunk_service.vector_search(search_request)
    
    # 2. 构建上下文
    context_text = "\n".join([chunk.content for chunk in context_chunks])
    
    # 3. 构建提示词
    prompt = f"""
    基于以下上下文回答问题：
    
    上下文：
    {context_text}
    
    问题：{request.message}
    
    请基于上下文提供准确的回答。如果上下文中没有相关信息，请说明。
    """
    
    # 4. 调用LLM生成回答（当前是简化实现）
    response_content = await llm_generate(prompt, conversation.temperature)
    
    return ChatResponse(
        content=response_content,
        context_chunks=context_chunks
    )
```

---

## 数据库设计

### Supabase数据库 (`PostgreSQL + pgvector`)

#### `knowledge_bases` 表
```sql
CREATE TABLE knowledge_bases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    embedding_model VARCHAR(100) DEFAULT 'text-embedding-3-small',
    chunk_size INTEGER DEFAULT 1000,
    chunk_overlap INTEGER DEFAULT 200,
    similarity_threshold FLOAT DEFAULT 0.7,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### `documents` 表
```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    kb_id UUID REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    file_type VARCHAR(50),
    file_size BIGINT,
    content TEXT,
    metadata JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'pending',
    progress FLOAT DEFAULT 0.0,
    error_message TEXT,
    chunk_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### `chunks` 表
```sql
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    kb_id UUID REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding VECTOR(1536),  -- pgvector扩展
    position_in_doc INTEGER,
    chunk_size INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### `conversations` 表
```sql
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) DEFAULT 'New Conversation',
    kb_ids UUID[] DEFAULT '{}',
    system_prompt TEXT,
    temperature FLOAT DEFAULT 0.7,
    max_tokens INTEGER DEFAULT 2000,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### `conversation_messages` 表
```sql
CREATE TABLE conversation_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,  -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    context_chunks UUID[] DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### `files` 表
```sql
CREATE TABLE files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    original_name VARCHAR(255) NOT NULL,
    file_type VARCHAR(50),
    file_size BIGINT,
    storage_path VARCHAR(500),
    mime_type VARCHAR(100),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

---

## API端点

### 同步端点

#### `POST /api/v1/knowledge-bases/`
**功能**: 创建知识库
**请求体**:
```typescript
{
  name: string;
  description?: string;
  embedding_model?: string;
  chunk_size?: number;
  chunk_overlap?: number;
  similarity_threshold?: number;
}
```

#### `GET /api/v1/knowledge-bases/`
**功能**: 获取知识库列表
**查询参数**:
- `page`: 页码（默认1）
- `page_size`: 每页数量（默认20）
- `search`: 搜索关键词

#### `POST /api/v1/documents/upload`
**功能**: 上传文档文件
**请求体**: `multipart/form-data`
- `kb_id`: 知识库ID
- `file`: 文档文件

#### `POST /api/v1/chunks/search`
**功能**: 向量搜索
**请求体**:
```typescript
{
  query: string;
  kb_ids: string[];
  top_k?: number;
  similarity_threshold?: number;
}
```

### 对话系统端点

#### `POST /api/v1/conversations/`
**功能**: 创建对话
**请求体**:
```typescript
{
  title?: string;
  kb_ids: string[];
  system_prompt?: string;
  temperature?: number;
  max_tokens?: number;
}
```

#### `POST /api/v1/conversations/{conv_id}/chat`
**功能**: 发送聊天消息
**请求体**:
```typescript
{
  conversation_id: string;
  message: string;
  use_rag?: boolean;
  top_k?: number;
}
```

### 文件管理端点

#### `POST /api/v1/files/upload`
**功能**: 上传文件

#### `GET /api/v1/files/{file_id}/download`
**功能**: 下载文件

---

## 核心算法

### 1. 文本分块算法

**固定大小分块**:
```python
class FixedSizeChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """将文本分割成固定大小的块"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap
        
        return chunks
```

**智能分块（按句子边界）**:
```python
class SentenceChunker:
    def chunk_text(self, text: str, max_size: int) -> List[str]:
        """按句子边界分块，避免截断"""
        sentences = text.split('。')  # 简化的句子分割
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) <= max_size:
                current_chunk += sentence + '。'
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + '。'
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
```

### 2. 相似度计算算法

**文本相似度（临时实现）**:
```python
def calculate_text_similarity(query: str, content: str) -> float:
    """
    计算文本相似度
    注意：这是简化实现，生产环境应使用向量相似度
    """
    query_words = set(query.lower().split())
    content_words = set(content.lower().split())
    
    intersection = query_words & content_words
    union = query_words | content_words
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)
```

**向量相似度（待实现）**:
```python
def calculate_vector_similarity(vec1: List[float], vec2: List[float]) -> float:
    """计算向量余弦相似度"""
    import numpy as np
    
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    
    cosine_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(cosine_sim)
```

---

## RAG方法论

### 1. 检索增强生成流程

**检索阶段**:
```python
async def retrieve_context(query: str, kb_ids: List[str], top_k: int) -> List[str]:
    """检索相关上下文"""
    # 1. 查询向量化
    query_embedding = await embedding_service.embed_text(query)
    
    # 2. 向量搜索
    search_results = await vector_search(query_embedding, kb_ids, top_k)
    
    # 3. 重排序（可选）
    reranked_results = await rerank_results(query, search_results)
    
    return [result.content for result in reranked_results]
```

**生成阶段**:
```python
async def generate_response(query: str, context: List[str], conversation_history: List[str]) -> str:
    """基于上下文生成回答"""
    # 1. 构建提示词
    prompt = build_rag_prompt(query, context, conversation_history)
    
    # 2. 调用LLM
    response = await llm_service.generate(prompt)
    
    # 3. 后处理
    return postprocess_response(response)
```

### 2. 提示词模板

**基础RAG提示词**:
```python
RAG_PROMPT_TEMPLATE = """
你是一个有帮助的AI助手。请基于提供的上下文回答用户的问题。

上下文信息：
{context}

对话历史：
{history}

用户问题：{query}

请基于上下文信息回答问题。如果上下文中没有相关信息，请明确说明。

回答：
"""
```

**多轮对话RAG提示词**:
```python
MULTI_TURN_RAG_PROMPT = """
你是一个专业的AI助手，正在与用户进行多轮对话。

知识库上下文：
{context}

对话历史：
{history}

当前用户问题：{query}

请注意：
1. 优先使用知识库中的信息
2. 考虑对话历史的上下文
3. 如果信息不足，请明确说明
4. 保持回答的连贯性和一致性

回答：
"""
```

### 3. 上下文管理策略

**上下文窗口管理**:
```python
class ContextManager:
    def __init__(self, max_context_length: int = 4000):
        self.max_context_length = max_context_length
    
    def manage_context(self, query: str, retrieved_chunks: List[str], history: List[str]) -> str:
        """智能管理上下文长度"""
        # 1. 计算可用空间
        query_length = len(query)
        history_length = sum(len(msg) for msg in history[-3:])  # 最近3轮对话
        available_length = self.max_context_length - query_length - history_length
        
        # 2. 截断或选择最相关的块
        selected_chunks = self.select_relevant_chunks(retrieved_chunks, available_length)
        
        return "\n\n".join(selected_chunks)
    
    def select_relevant_chunks(self, chunks: List[str], max_length: int) -> List[str]:
        """选择最相关的文档块"""
        selected = []
        current_length = 0
        
        for chunk in chunks:
            if current_length + len(chunk) <= max_length:
                selected.append(chunk)
                current_length += len(chunk)
            else:
                break
        
        return selected
```

---

## 当前限制

### 1. 向量搜索限制

**缺失真实向量搜索**:
- 当前使用文本匹配作为临时实现
- 缺少embedding生成和向量存储
- 无法进行语义相似度搜索
- 搜索准确性有限

**建议改进**:
```python
# 当前实现（临时）
similarity_score = len(query_words & content_words) / len(query_words | content_words)

# 期望实现
query_embedding = await embedding_service.embed_text(query)
chunk_embeddings = await get_chunk_embeddings(chunk_ids)
similarity_scores = cosine_similarity(query_embedding, chunk_embeddings)
```

### 2. LLM集成限制

**简化的回答生成**:
- 当前使用模板拼接生成回答
- 没有集成实际的LLM服务
- 无法进行智能对话生成

**当前实现**:
```python
# 简化的回答生成
assistant_content = f"基于您的问题：{question}\n\n"
if context_text:
    assistant_content += f"我从知识库中找到了相关信息:\n{context_text}\n\n"
    assistant_content += "根据这些信息，我的回答是：[这里应该调用LLM生成实际回复]"
```

### 3. 文档处理限制

**基础文件解析**:
- 只支持简单的文本文件
- 缺少复杂格式（PDF、Word、Excel等）的解析
- 没有表格、图像等结构化内容的处理

### 4. 性能限制

**数据库查询优化**:
- 缺少复杂查询的索引优化
- 大量数据时的分页性能问题
- 没有查询结果缓存

**并发处理**:
- 文档处理不是真正的异步
- 缺少任务队列和后台处理
- 大文件处理可能阻塞主线程

---

## 改进机会

### 1. 向量搜索增强

**集成Embedding服务**:
```python
class EmbeddingService:
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        self.client = OpenAI()  # 或其他embedding服务
    
    async def embed_text(self, text: str) -> List[float]:
        """生成文本向量"""
        response = await self.client.embeddings.create(
            model=self.model_name,
            input=text
        )
        return response.data[0].embedding
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成向量"""
        response = await self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        return [item.embedding for item in response.data]
```

**向量数据库集成**:
```python
# 方案1：使用pgvector扩展
CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops);

# 方案2：集成专用向量数据库
class VectorStore:
    def __init__(self, store_type: str = "pinecone"):  # pinecone, weaviate, qdrant
        self.store = self._init_store(store_type)
    
    async def insert_vectors(self, vectors: List[dict]):
        """插入向量"""
        await self.store.upsert(vectors)
    
    async def search_similar(self, query_vector: List[float], top_k: int) -> List[dict]:
        """向量相似度搜索"""
        return await self.store.query(vector=query_vector, top_k=top_k)
```

### 2. LLM服务集成

**多LLM提供商支持**:
```python
class LLMService:
    def __init__(self, provider: str = "openai", model: str = "gpt-3.5-turbo"):
        self.provider = provider
        self.model = model
        self.client = self._init_client(provider)
    
    async def generate_response(
        self, 
        messages: List[dict], 
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """生成回答"""
        if self.provider == "openai":
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        # 其他提供商的实现...
```

**流式响应支持**:
```python
async def stream_chat_response(request: ChatRequest) -> AsyncGenerator[str, None]:
    """流式生成对话回答"""
    # 1. 检索上下文
    context = await retrieve_context(request.message, request.kb_ids)
    
    # 2. 构建提示词
    messages = build_chat_messages(request.message, context)
    
    # 3. 流式生成
    async for chunk in llm_service.stream_generate(messages):
        yield chunk
```

### 3. 高级RAG技术

**混合检索策略**:
```python
class HybridRetriever:
    def __init__(self):
        self.dense_retriever = VectorRetriever()  # 向量检索
        self.sparse_retriever = BM25Retriever()   # 关键词检索
        self.reranker = CrossEncoderReranker()    # 重排序
    
    async def retrieve(self, query: str, top_k: int) -> List[Chunk]:
        # 1. 密集检索
        dense_results = await self.dense_retriever.search(query, top_k * 2)
        
        # 2. 稀疏检索
        sparse_results = await self.sparse_retriever.search(query, top_k * 2)
        
        # 3. 结果融合
        combined_results = self.combine_results(dense_results, sparse_results)
        
        # 4. 重排序
        reranked = await self.reranker.rerank(query, combined_results)
        
        return reranked[:top_k]
```

**查询扩展和改写**:
```python
class QueryProcessor:
    def __init__(self):
        self.query_expander = QueryExpander()
        self.query_rewriter = QueryRewriter()
    
    async def process_query(self, query: str, conversation_history: List[str]) -> str:
        # 1. 查询改写（考虑对话上下文）
        rewritten_query = await self.query_rewriter.rewrite(query, conversation_history)
        
        # 2. 查询扩展（添加同义词、相关词）
        expanded_query = await self.query_expander.expand(rewritten_query)
        
        return expanded_query
```

### 4. 文档处理增强

**多格式文档解析**:
```python
class DocumentProcessor:
    def __init__(self):
        self.parsers = {
            'pdf': PDFParser(),
            'docx': WordParser(),
            'xlsx': ExcelParser(),
            'pptx': PowerPointParser(),
            'md': MarkdownParser(),
            'html': HTMLParser()
        }
    
    async def process_document(self, file_path: str, file_type: str) -> ProcessedDocument:
        """处理不同格式的文档"""
        parser = self.parsers.get(file_type)
        if not parser:
            raise ValueError(f"不支持的文件类型: {file_type}")
        
        # 1. 解析文档
        content = await parser.parse(file_path)
        
        # 2. 结构化提取
        structured_content = await parser.extract_structure(content)
        
        # 3. 智能分块
        chunks = await self.smart_chunk(structured_content)
        
        return ProcessedDocument(content=content, chunks=chunks)
```

**智能分块策略**:
```python
class SmartChunker:
    def __init__(self):
        self.sentence_splitter = SentenceSplitter()
        self.semantic_chunker = SemanticChunker()
    
    async def chunk_document(self, content: str, chunk_size: int) -> List[Chunk]:
        # 1. 基于语义的分块
        semantic_chunks = await self.semantic_chunker.chunk(content)
        
        # 2. 大小调整
        sized_chunks = self.adjust_chunk_size(semantic_chunks, chunk_size)
        
        # 3. 重叠处理
        overlapped_chunks = self.add_overlap(sized_chunks)
        
        return overlapped_chunks
```

### 5. 性能优化

**缓存策略**:
```python
class CacheManager:
    def __init__(self):
        self.redis_client = Redis()
        self.embedding_cache = EmbeddingCache()
        self.search_cache = SearchCache()
    
    async def get_or_compute_embedding(self, text: str) -> List[float]:
        """缓存的向量计算"""
        cache_key = f"embedding:{hash(text)}"
        cached = await self.redis_client.get(cache_key)
        
        if cached:
            return json.loads(cached)
        
        embedding = await embedding_service.embed_text(text)
        await self.redis_client.setex(cache_key, 3600, json.dumps(embedding))
        
        return embedding
```

**异步任务队列**:
```python
from celery import Celery

app = Celery('rag_api')

@app.task
async def process_document_async(document_id: str):
    """异步处理文档"""
    # 1. 获取文档
    document = await document_service.get_document(document_id)
    
    # 2. 分块处理
    chunks = await chunk_processor.process(document.content)
    
    # 3. 生成向量
    for chunk in chunks:
        embedding = await embedding_service.embed_text(chunk.content)
        await chunk_service.update_chunk_embedding(chunk.id, embedding)
    
    # 4. 更新状态
    await document_service.update_document_status(
        document_id, 
        DocumentStatus.COMPLETED,
        progress=1.0,
        chunk_count=len(chunks)
    )
```

---

## 结论

当前RAG API系统提供了一个完整的检索增强生成框架，包含知识库管理、文档处理、对话系统等核心功能。系统采用现代化的技术栈，具有良好的扩展性和维护性。

### 技术亮点
1. **模块化架构**: 清晰的分层设计，各组件职责明确
2. **异步支持**: 全面的异步编程，提高并发性能
3. **类型安全**: 完整的Pydantic模型定义，确保数据一致性
4. **数据库集成**: 与Supabase深度集成，支持pgvector向量扩展
5. **RESTful API**: 标准化的HTTP接口，易于集成和使用

### 关键改进方向
1. **真实向量搜索**: 集成embedding服务和向量数据库
2. **LLM服务集成**: 连接OpenAI、Azure OpenAI等LLM提供商
3. **高级RAG技术**: 实现混合检索、查询扩展、重排序等
4. **文档处理增强**: 支持更多文件格式和智能分块
5. **性能优化**: 实现缓存、异步任务、查询优化等

这些改进将使系统从原型向生产级应用演进，提供更准确、更智能、更高效的RAG服务。

