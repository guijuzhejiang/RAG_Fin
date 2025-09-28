# RAGFlow API 文档（中文版）

## 概述

RAGFlow 是一个基于深度文档理解的 RAG（Retrieval-Augmented Generation）引擎，提供完整的用户管理、知识库管理、文档解析、对话等功能。本文档详细介绍了 RAGFlow 的 API 接口设计和使用方法。

## 架构说明

RAGFlow 采用模块化设计，包含两个主要的 API 系统：

### 1. 用户端 API（/api/apps/）
位于 `api/apps/` 目录下，提供面向最终用户的功能接口：
- 用户认证与管理
- 知识库操作
- 文档管理
- 对话系统
- API Token 管理

### 2. 管理端 API（/management/server/routes/）
位于 `management/server/routes/` 目录下，提供管理员级别的功能接口：
- 用户管理
- 系统配置
- 批量操作
- 监控统计

## 认证机制

### 用户认证
- **Session 认证**：基于 Flask-Login 的会话认证
- **Token 认证**：API Token 方式，适用于外部系统集成
- **OAuth 认证**：支持 GitHub 和飞书 OAuth 登录

### API Token 验证流程
```python
# Token 验证示例
Authorization: Bearer <your-api-token>
```

## 核心 API 模块

## 1. 用户管理 API

### 1.1 用户认证（user_app.py）

#### 用户登录
- **接口**: `POST /v1/user/login`
- **功能**: 用户登录验证
- **请求参数**:
  ```json
  {
    "email": "user@example.com",
    "password": "encrypted_password"
  }
  ```
- **响应**:
  ```json
  {
    "code": 0,
    "data": {
      "id": "user_id",
      "email": "user@example.com",
      "nickname": "用户昵称"
    },
    "message": "Welcome back!"
  }
  ```

#### 用户注册
- **接口**: `POST /v1/user/register`
- **功能**: 新用户注册
- **请求参数**:
  ```json
  {
    "nickname": "用户昵称",
    "email": "user@example.com",
    "password": "encrypted_password"
  }
  ```

#### 获取用户信息
- **接口**: `GET /v1/user/info`
- **功能**: 获取当前登录用户信息
- **认证**: 需要登录
- **响应**:
  ```json
  {
    "code": 0,
    "data": {
      "id": "user_id",
      "nickname": "用户昵称",
      "email": "user@example.com"
    }
  }
  ```

#### 用户设置更新
- **接口**: `POST /v1/user/setting`
- **功能**: 更新用户设置
- **认证**: 需要登录
- **请求参数**:
  ```json
  {
    "nickname": "新昵称",
    "password": "当前密码",
    "new_password": "新密码"
  }
  ```

### 1.2 OAuth 认证

#### GitHub OAuth 回调
- **接口**: `GET /v1/user/github_callback`
- **功能**: 处理 GitHub OAuth 登录回调

#### 飞书 OAuth 回调
- **接口**: `GET /v1/user/feishu_callback`
- **功能**: 处理飞书 OAuth 登录回调

## 2. 知识库管理 API

### 2.1 知识库基本操作（kb_app.py）

#### 创建知识库
- **接口**: `POST /v1/dataset/create`
- **功能**: 创建新的知识库
- **认证**: 需要登录
- **请求参数**:
  ```json
  {
    "name": "知识库名称",
    "description": "知识库描述",
    "permission": "me",
    "parser_id": "naive",
    "parser_config": {}
  }
  ```
- **响应**:
  ```json
  {
    "code": 0,
    "data": {
      "kb_id": "knowledge_base_id"
    }
  }
  ```

#### 更新知识库
- **接口**: `POST /v1/dataset/update`
- **功能**: 更新知识库信息
- **认证**: 需要登录
- **请求参数**:
  ```json
  {
    "kb_id": "knowledge_base_id",
    "name": "新名称",
    "description": "新描述",
    "permission": "me",
    "parser_id": "naive"
  }
  ```

#### 获取知识库列表
- **接口**: `GET /v1/dataset/list`
- **功能**: 获取用户的知识库列表
- **认证**: 需要登录
- **查询参数**:
  - `keywords`: 搜索关键词
  - `page`: 页码（默认 1）
  - `page_size`: 每页数量（默认 150）
  - `parser_id`: 解析器过滤
  - `orderby`: 排序字段（默认 create_time）
  - `desc`: 是否降序

#### 获取知识库详情
- **接口**: `GET /v1/dataset/detail?kb_id=<kb_id>`
- **功能**: 获取指定知识库详细信息
- **认证**: 需要登录

#### 删除知识库
- **接口**: `POST /v1/dataset/rm`
- **功能**: 删除知识库
- **认证**: 需要登录
- **请求参数**:
  ```json
  {
    "kb_id": "knowledge_base_id"
  }
  ```

### 2.2 知识库高级功能

#### 获取知识库标签
- **接口**: `GET /v1/dataset/<kb_id>/tags`
- **功能**: 获取知识库中的所有标签

#### 删除标签
- **接口**: `POST /v1/dataset/<kb_id>/rm_tags`
- **功能**: 删除指定标签
- **请求参数**:
  ```json
  {
    "tags": ["标签1", "标签2"]
  }
  ```

#### 重命名标签
- **接口**: `POST /v1/dataset/<kb_id>/rename_tag`
- **功能**: 重命名标签
- **请求参数**:
  ```json
  {
    "from_tag": "旧标签名",
    "to_tag": "新标签名"
  }
  ```

#### 获取知识图谱
- **接口**: `GET /v1/dataset/<kb_id>/knowledge_graph`
- **功能**: 获取知识库的知识图谱数据
- **响应**:
  ```json
  {
    "code": 0,
    "data": {
      "graph": {
        "nodes": [],
        "edges": []
      },
      "mind_map": {}
    }
  }
  ```

#### 获取知识库图片
- **接口**: `GET /v1/dataset/images`
- **功能**: 获取知识库中的图片列表
- **查询参数**:
  - `kb_id`: 知识库 ID
  - `page`: 页码
  - `page_size`: 每页数量
  - `search`: 搜索关键词

## 3. 文档管理 API

### 3.1 文档基本操作（document_app.py）

#### 上传文档
- **接口**: `POST /v1/document/upload`
- **功能**: 上传文档到知识库
- **认证**: 需要登录
- **请求类型**: multipart/form-data
- **请求参数**:
  - `kb_id`: 知识库 ID
  - `file`: 文件对象
  - `parser_id`: 解析器 ID（可选）
  - `run`: 是否立即解析（可选）

#### 网页抓取
- **接口**: `POST /v1/document/web_crawl`
- **功能**: 从 URL 抓取内容并创建文档
- **认证**: 需要登录
- **请求参数**:
  - `kb_id`: 知识库 ID
  - `name`: 文档名称
  - `url`: 网页 URL

#### 创建虚拟文档
- **接口**: `POST /v1/document/create`
- **功能**: 创建空的虚拟文档
- **认证**: 需要登录
- **请求参数**:
  ```json
  {
    "name": "文档名称",
    "kb_id": "knowledge_base_id"
  }
  ```

#### 获取文档列表
- **接口**: `GET /v1/document/list`
- **功能**: 获取知识库中的文档列表
- **认证**: 需要登录
- **查询参数**:
  - `kb_id`: 知识库 ID
  - `keywords`: 搜索关键词
  - `page`: 页码
  - `page_size`: 每页数量
  - `orderby`: 排序字段
  - `desc`: 是否降序

#### 获取文档信息
- **接口**: `POST /v1/document/infos`
- **功能**: 获取多个文档的详细信息
- **认证**: 需要登录
- **请求参数**:
  ```json
  {
    "doc_ids": ["doc_id1", "doc_id2"]
  }
  ```

#### 删除文档
- **接口**: `POST /v1/document/rm`
- **功能**: 删除文档
- **认证**: 需要登录
- **请求参数**:
  ```json
  {
    "doc_id": ["doc_id1", "doc_id2"]
  }
  ```

### 3.2 文档处理操作

#### 运行文档解析
- **接口**: `POST /v1/document/run`
- **功能**: 启动或停止文档解析
- **认证**: 需要登录
- **请求参数**:
  ```json
  {
    "doc_ids": ["doc_id1", "doc_id2"],
    "run": "1",  // 1:运行, 0:停止
    "delete": true  // 是否删除旧数据
  }
  ```

#### 重命名文档
- **接口**: `POST /v1/document/rename`
- **功能**: 重命名文档
- **认证**: 需要登录
- **请求参数**:
  ```json
  {
    "doc_id": "document_id",
    "name": "新文档名称"
  }
  ```

#### 更改文档状态
- **接口**: `POST /v1/document/change_status`
- **功能**: 启用或禁用文档
- **认证**: 需要登录
- **请求参数**:
  ```json
  {
    "doc_id": "document_id",
    "status": 1  // 1:启用, 0:禁用
  }
  ```

#### 更改解析器
- **接口**: `POST /v1/document/change_parser`
- **功能**: 更改文档的解析器
- **认证**: 需要登录
- **请求参数**:
  ```json
  {
    "doc_id": "document_id",
    "parser_id": "new_parser_id",
    "parser_config": {}
  }
  ```

#### 获取文档内容
- **接口**: `GET /v1/document/get/<doc_id>`
- **功能**: 下载文档原始内容

#### 获取文档图片
- **接口**: `GET /v1/document/image/<image_id>`
- **功能**: 获取文档中的图片

#### 设置文档元数据
- **接口**: `POST /v1/document/set_meta`
- **功能**: 设置文档元数据
- **认证**: 需要登录
- **请求参数**:
  ```json
  {
    "doc_id": "document_id",
    "meta": "{\"key\": \"value\"}"
  }
  ```

### 3.3 文档解析和处理

#### 上传并解析
- **接口**: `POST /v1/document/upload_and_parse`
- **功能**: 上传文档并立即解析
- **认证**: 需要登录
- **请求类型**: multipart/form-data
- **请求参数**:
  - `conversation_id`: 对话 ID
  - `file`: 文件对象

#### 解析文档内容
- **接口**: `POST /v1/document/parse`
- **功能**: 解析文档或 URL 内容
- **认证**: 需要登录
- **请求参数**:
  ```json
  {
    "url": "http://example.com"  // 可选，解析网页
  }
  ```
- **或上传文件**: multipart/form-data

## 4. 块（Chunk）管理 API

### 4.1 块操作（chunk_app.py）

#### 获取文档块列表
- **接口**: `POST /v1/chunk/list`
- **功能**: 获取文档的分块列表
- **认证**: 需要登录
- **请求参数**:
  ```json
  {
    "doc_id": "document_id",
    "page": 1,
    "size": 30,
    "keywords": "搜索关键词",
    "available_int": 1  // 可选，过滤可用状态
  }
  ```

#### 获取块详情
- **接口**: `GET /v1/chunk/get?chunk_id=<chunk_id>`
- **功能**: 获取指定块的详细信息
- **认证**: 需要登录

#### 更新块内容
- **接口**: `POST /v1/chunk/set`
- **功能**: 更新块的内容和属性
- **认证**: 需要登录
- **请求参数**:
  ```json
  {
    "doc_id": "document_id",
    "chunk_id": "chunk_id",
    "content_with_weight": "更新后的内容",
    "important_kwd": ["重要", "关键词"],
    "question_kwd": ["问题", "关键词"],
    "tag_kwd": ["标签"],
    "available_int": 1,
    "img_id": "image_id"
  }
  ```

#### 删除块
- **接口**: `POST /v1/chunk/rm`
- **功能**: 删除指定的块
- **认证**: 需要登录
- **请求参数**:
  ```json
  {
    "chunk_ids": ["chunk_id1", "chunk_id2"]
  }
  ```

#### 创建块
- **接口**: `POST /v1/chunk/create`
- **功能**: 创建新的块
- **认证**: 需要登录
- **请求参数**:
  ```json
  {
    "doc_id": "document_id",
    "content_with_weight": "块内容"
  }
  ```

## 5. 对话系统 API

### 5.1 对话管理（dialog_app.py）

#### 创建/更新对话配置
- **接口**: `POST /v1/dialog/set`
- **功能**: 创建或更新对话机器人配置
- **认证**: 需要登录
- **请求参数**:
  ```json
  {
    "dialog_id": "dialog_id",  // 可选，不提供则创建新对话
    "name": "对话名称",
    "description": "对话描述",
    "kb_ids": ["kb_id1", "kb_id2"],
    "llm_id": "llm_model_id",
    "llm_setting": {
      "temperature": 0.7,
      "max_tokens": 512
    },
    "prompt_config": {
      "system": "系统提示词",
      "prologue": "开场白",
      "parameters": [{"key": "knowledge", "optional": false}],
      "empty_response": "未找到相关内容的回复"
    },
    "top_n": 6,
    "top_k": 1024,
    "similarity_threshold": 0.1,
    "vector_similarity_weight": 0.3,
    "rerank_id": "rerank_model_id",
    "icon": "icon_url"
  }
  ```

#### 获取对话配置
- **接口**: `GET /v1/dialog/get?dialog_id=<dialog_id>`
- **功能**: 获取对话配置详情
- **认证**: 需要登录

#### 获取对话列表
- **接口**: `GET /v1/dialog/list`
- **功能**: 获取用户的对话列表
- **认证**: 需要登录

#### 删除对话
- **接口**: `POST /v1/dialog/rm`
- **功能**: 删除对话配置
- **认证**: 需要登录
- **请求参数**:
  ```json
  {
    "dialog_ids": ["dialog_id1", "dialog_id2"]
  }
  ```

### 5.2 对话交互（conversation_app.py）

#### 发送消息
- **接口**: `POST /v1/conversation/completion`
- **功能**: 发送消息并获取AI回复
- **认证**: 需要登录
- **请求参数**:
  ```json
  {
    "conversation_id": "conversation_id",
    "messages": [
      {
        "role": "user",
        "content": "用户消息",
        "id": "message_id"
      }
    ],
    "stream": true,  // 是否流式返回
    "quote": false,  // 是否显示引用
    "doc_ids": ["doc_id1"]  // 可选，指定搜索文档
  }
  ```

#### 创建对话会话
- **接口**: `POST /v1/conversation/create`
- **功能**: 创建新的对话会话
- **认证**: 需要登录
- **请求参数**:
  ```json
  {
    "dialog_id": "dialog_id",
    "name": "对话会话名称"
  }
  ```

#### 获取对话历史
- **接口**: `GET /v1/conversation/get?conversation_id=<conversation_id>`
- **功能**: 获取对话历史记录
- **认证**: 需要登录

#### 获取对话列表
- **接口**: `GET /v1/conversation/list`
- **功能**: 获取用户的对话会话列表
- **认证**: 需要登录
- **查询参数**:
  - `dialog_id`: 对话配置 ID
  - `page`: 页码
  - `page_size`: 每页数量

#### 删除对话会话
- **接口**: `POST /v1/conversation/rm`
- **功能**: 删除对话会话
- **认证**: 需要登录
- **请求参数**:
  ```json
  {
    "conversation_ids": ["conv_id1", "conv_id2"]
  }
  ```

## 6. API Token 管理（api_app.py）

### 6.1 Token 操作

#### 创建 API Token
- **接口**: `POST /api/v1/new_token`
- **功能**: 创建新的 API Token
- **认证**: 需要登录
- **请求参数**:
  ```json
  {
    "dialog_id": "dialog_id",  // 或 canvas_id
    "canvas_id": "canvas_id"
  }
  ```

#### 获取 Token 列表
- **接口**: `GET /api/v1/token_list?dialog_id=<dialog_id>`
- **功能**: 获取指定对话的 Token 列表
- **认证**: 需要登录

#### 删除 Token
- **接口**: `POST /api/v1/rm`
- **功能**: 删除指定的 Token
- **认证**: 需要登录
- **请求参数**:
  ```json
  {
    "tokens": ["token1", "token2"],
    "tenant_id": "tenant_id"
  }
  ```

#### 获取使用统计
- **接口**: `GET /api/v1/stats`
- **功能**: 获取 API 使用统计
- **认证**: 需要登录
- **查询参数**:
  - `from_date`: 开始日期
  - `to_date`: 结束日期

### 6.2 外部 API 接口

#### 创建对话会话（外部）
- **接口**: `GET /api/v1/new_conversation?user_id=<user_id>`
- **功能**: 通过 API Token 创建对话会话
- **认证**: Bearer Token
- **请求头**: `Authorization: Bearer <api_token>`

#### 发送消息（外部）
- **接口**: `POST /api/v1/completion`
- **功能**: 通过 API Token 发送消息
- **认证**: Bearer Token
- **请求参数**:
  ```json
  {
    "conversation_id": "conversation_id",
    "messages": [
      {
        "role": "user",
        "content": "用户消息"
      }
    ],
    "stream": true,
    "quote": false
  }
  ```

#### 获取对话历史（外部）
- **接口**: `GET /api/v1/conversation/<conversation_id>`
- **功能**: 通过 API Token 获取对话历史
- **认证**: Bearer Token

#### 上传文档（外部）
- **接口**: `POST /api/v1/document/upload`
- **功能**: 通过 API Token 上传文档
- **认证**: Bearer Token
- **请求类型**: multipart/form-data
- **请求参数**:
  - `kb_name`: 知识库名称
  - `file`: 文件对象
  - `parser_id`: 解析器 ID（可选）
  - `run`: 是否立即解析（可选）

#### 检索接口
- **接口**: `POST /api/v1/retrieval`
- **功能**: 知识库检索
- **认证**: Bearer Token
- **请求参数**:
  ```json
  {
    "kb_id": ["kb_id1"],
    "question": "检索问题",
    "doc_ids": ["doc_id1"],  // 可选
    "page": 1,
    "size": 30,
    "similarity_threshold": 0.2,
    "vector_similarity_weight": 0.3,
    "top_k": 1024,
    "rerank_id": "rerank_model_id",  // 可选
    "keyword": false  // 是否启用关键词增强
  }
  ```

#### 获取文档块列表（外部）
- **接口**: `POST /api/v1/list_chunks`
- **功能**: 获取文档的分块列表
- **认证**: Bearer Token
- **请求参数**:
  ```json
  {
    "doc_name": "文档名称",  // 或使用 doc_id
    "doc_id": "document_id"
  }
  ```

#### 获取知识库文档列表（外部）
- **接口**: `POST /api/v1/list_kb_docs`
- **功能**: 获取知识库的文档列表
- **认证**: Bearer Token
- **请求参数**:
  ```json
  {
    "kb_name": "知识库名称",
    "page": 1,
    "page_size": 15,
    "orderby": "create_time",
    "desc": true,
    "keywords": "搜索关键词"
  }
  ```

#### 删除文档（外部）
- **接口**: `DELETE /api/v1/document`
- **功能**: 删除文档
- **认证**: Bearer Token
- **请求参数**:
  ```json
  {
    "doc_names": ["文档名称1"],  // 或使用 doc_ids
    "doc_ids": ["doc_id1"]
  }
  ```

## 7. 文件管理 API（file_app.py）

### 7.1 文件操作

#### 上传文件
- **接口**: `POST /v1/file/upload`
- **功能**: 上传文件到文件系统
- **认证**: 需要登录
- **请求类型**: multipart/form-data
- **请求参数**:
  - `parent_id`: 父目录 ID（可选）
  - `file`: 文件对象

#### 创建文件夹
- **接口**: `POST /v1/file/create`
- **功能**: 创建文件夹或虚拟文件
- **认证**: 需要登录
- **请求参数**:
  ```json
  {
    "name": "文件夹名称",
    "parent_id": "parent_folder_id",  // 可选
    "type": "folder"  // 或 "virtual"
  }
  ```

#### 获取文件列表
- **接口**: `GET /v1/file/list`
- **功能**: 获取文件和文件夹列表
- **认证**: 需要登录
- **查询参数**:
  - `parent_id`: 父目录 ID
  - `page`: 页码
  - `page_size`: 每页数量
  - `orderby`: 排序字段
  - `desc`: 是否降序

#### 删除文件
- **接口**: `POST /v1/file/rm`
- **功能**: 删除文件或文件夹
- **认证**: 需要登录
- **请求参数**:
  ```json
  {
    "file_ids": ["file_id1", "file_id2"]
  }
  ```

#### 重命名文件
- **接口**: `POST /v1/file/rename`
- **功能**: 重命名文件或文件夹
- **认证**: 需要登录
- **请求参数**:
  ```json
  {
    "file_id": "file_id",
    "name": "新名称"
  }
  ```

#### 移动文件
- **接口**: `POST /v1/file/move`
- **功能**: 移动文件到指定目录
- **认证**: 需要登录
- **请求参数**:
  ```json
  {
    "file_ids": ["file_id1"],
    "to_parent_id": "target_parent_id"
  }
  ```

#### 下载文件
- **接口**: `GET /v1/file/get/<file_id>`
- **功能**: 下载文件内容

#### 解析文件
- **接口**: `POST /v1/file/parse`
- **功能**: 解析文件内容并返回文本
- **认证**: 需要登录
- **请求类型**: multipart/form-data 或 JSON（带 URL）

## 8. LLM 管理 API（llm_app.py）

### 8.1 LLM 配置

#### 获取 LLM 工厂列表
- **接口**: `GET /v1/llm/factories`
- **功能**: 获取支持的 LLM 提供商列表
- **认证**: 需要登录

#### 设置 API Key
- **接口**: `POST /v1/llm/set_api_key`
- **功能**: 设置 LLM 提供商的 API Key
- **认证**: 需要登录
- **请求参数**:
  ```json
  {
    "llm_factory": "OpenAI",
    "api_key": "your_api_key",
    "base_url": "https://api.openai.com/v1"  // 可选
  }
  ```

#### 添加自定义 LLM
- **接口**: `POST /v1/llm/add_llm`
- **功能**: 添加自定义 LLM 模型
- **认证**: 需要登录
- **请求参数**:
  ```json
  {
    "llm_factory": "Custom",
    "llm_name": "custom-model",
    "model_type": "chat",
    "api_key": "api_key",
    "base_url": "model_endpoint"
  }
  ```

#### 获取我的 LLM 列表
- **接口**: `GET /v1/llm/my_llms`
- **功能**: 获取用户配置的 LLM 列表
- **认证**: 需要登录

#### 删除 LLM 配置
- **接口**: `POST /v1/llm/rm`
- **功能**: 删除 LLM 配置
- **认证**: 需要登录
- **请求参数**:
  ```json
  {
    "llm_factory": "provider_name",
    "llm_name": "model_name"
  }
  ```

## 9. 管理端 API

### 9.1 用户管理（management/server/routes/users/）

#### 获取用户列表
- **接口**: `GET /api/v1/users`
- **功能**: 获取用户列表（管理员）
- **查询参数**:
  - `currentPage`: 当前页码
  - `size`: 每页数量
  - `username`: 用户名过滤
  - `email`: 邮箱过滤
  - `sort_by`: 排序字段
  - `sort_order`: 排序方向

#### 创建用户
- **接口**: `POST /api/v1/users`
- **功能**: 创建新用户（管理员）
- **请求参数**:
  ```json
  {
    "username": "新用户名",
    "email": "user@example.com",
    "password": "用户密码"
  }
  ```

#### 更新用户
- **接口**: `PUT /api/v1/users/<user_id>`
- **功能**: 更新用户信息（管理员）
- **请求参数**:
  ```json
  {
    "id": "user_id",
    "username": "更新后的用户名",
    "email": "new@example.com"
  }
  ```

#### 删除用户
- **接口**: `DELETE /api/v1/users/<user_id>`
- **功能**: 删除用户（管理员）

#### 重置用户密码
- **接口**: `PUT /api/v1/users/<user_id>/reset-password`
- **功能**: 重置用户密码（管理员）
- **请求参数**:
  ```json
  {
    "password": "新密码"
  }
  ```

#### 获取当前用户信息
- **接口**: `GET /api/v1/users/me`
- **功能**: 获取当前登录的管理员信息

### 9.2 知识库管理（management/server/routes/knowledgebases/）

#### 获取知识库列表
- **接口**: `GET /api/v1/knowledgebases`
- **功能**: 获取知识库列表（管理员）
- **查询参数**:
  - `currentPage`: 当前页码
  - `size`: 每页数量
  - `name`: 知识库名称过滤
  - `sort_by`: 排序字段
  - `sort_order`: 排序方向

#### 获取知识库详情
- **接口**: `GET /api/v1/knowledgebases/<kb_id>`
- **功能**: 获取知识库详情（管理员）

#### 创建知识库
- **接口**: `POST /api/v1/knowledgebases`
- **功能**: 创建知识库（管理员）
- **请求参数**:
  ```json
  {
    "name": "知识库名称",
    "description": "知识库描述"
  }
  ```

#### 更新知识库
- **接口**: `PUT /api/v1/knowledgebases/<kb_id>`
- **功能**: 更新知识库（管理员）

#### 删除知识库
- **接口**: `DELETE /api/v1/knowledgebases/<kb_id>`
- **功能**: 删除知识库（管理员）

#### 批量删除知识库
- **接口**: `DELETE /api/v1/knowledgebases/batch`
- **功能**: 批量删除知识库（管理员）
- **请求参数**:
  ```json
  {
    "ids": ["kb_id1", "kb_id2"]
  }
  ```

#### 获取知识库文档
- **接口**: `GET /api/v1/knowledgebases/<kb_id>/documents`
- **功能**: 获取知识库下的文档列表（管理员）

#### 添加文档到知识库
- **接口**: `POST /api/v1/knowledgebases/<kb_id>/documents`
- **功能**: 添加文档到知识库（管理员）
- **请求参数**:
  ```json
  {
    "file_ids": ["file_id1", "file_id2"]
  }
  ```

#### 删除文档
- **接口**: `DELETE /api/v1/knowledgebases/documents/<doc_id>`
- **功能**: 从知识库删除文档（管理员）

#### 获取文档解析进度
- **接口**: `GET /api/v1/knowledgebases/documents/<doc_id>/parse/progress`
- **功能**: 获取文档解析进度（管理员）

#### 解析文档
- **接口**: `POST /api/v1/knowledgebases/documents/<doc_id>/parse`
- **功能**: 开始解析文档（管理员）

#### 批量解析
- **接口**: `POST /api/v1/knowledgebases/<kb_id>/batch_parse_sequential/start`
- **功能**: 启动顺序批量解析任务（管理员）

#### 获取批量解析进度
- **接口**: `GET /api/v1/knowledgebases/<kb_id>/batch_parse_sequential/progress`
- **功能**: 获取批量解析进度（管理员）

#### Embedding 配置

##### 获取系统 Embedding 配置
- **接口**: `GET /api/v1/knowledgebases/system_embedding_config`
- **功能**: 获取系统级 Embedding 配置（管理员）

##### 设置系统 Embedding 配置
- **接口**: `POST /api/v1/knowledgebases/system_embedding_config`
- **功能**: 设置系统级 Embedding 配置（管理员）
- **请求参数**:
  ```json
  {
    "llm_name": "text-embedding-3-small",
    "api_base": "https://api.openai.com/v1",
    "api_key": "your_api_key"
  }
  ```

##### 获取租户 Embedding 模型
- **接口**: `GET /api/v1/knowledgebases/embedding_models/<kb_id>`
- **功能**: 获取租户的嵌入模型配置（管理员）

##### 获取知识库 Embedding 配置
- **接口**: `GET /api/v1/knowledgebases/embedding_config?kb_id=<kb_id>`
- **功能**: 获取知识库的嵌入模型配置（管理员）

## 错误处理

### 标准错误响应格式
```json
{
  "code": 错误代码,
  "message": "错误描述",
  "data": false
}
```

### 常见错误代码
- `0`: 成功
- `400`: 参数错误
- `401`: 认证失败
- `403`: 权限不足
- `404`: 资源不存在
- `500`: 服务器内部错误

### 认证错误
- `AUTHENTICATION_ERROR`: 认证失败
- `OPERATING_ERROR`: 操作权限错误

## 使用示例

### 1. 用户登录并创建知识库
```python
import requests

# 1. 用户登录
login_data = {
    "email": "user@example.com", 
    "password": "encrypted_password"
}
response = requests.post("/v1/user/login", json=login_data)
auth_token = response.json()["data"]["access_token"]

# 2. 创建知识库
headers = {"Authorization": f"Bearer {auth_token}"}
kb_data = {"name": "我的知识库", "description": "测试知识库"}
response = requests.post("/v1/dataset/create", json=kb_data, headers=headers)
kb_id = response.json()["data"]["kb_id"]
```

### 2. 上传文档并解析
```python
# 上传文档
files = {"file": open("document.pdf", "rb")}
form_data = {"kb_id": kb_id, "run": "1"}
response = requests.post("/v1/document/upload", 
                        files=files, data=form_data, headers=headers)

# 获取文档列表
params = {"kb_id": kb_id}
response = requests.get("/v1/document/list", params=params, headers=headers)
documents = response.json()["data"]["docs"]
```

### 3. 创建对话并发送消息
```python
# 创建对话配置
dialog_data = {
    "name": "智能助手",
    "kb_ids": [kb_id],
    "llm_id": "gpt-3.5-turbo"
}
response = requests.post("/v1/dialog/set", json=dialog_data, headers=headers)
dialog_id = response.json()["data"]["id"]

# 创建对话会话
conv_data = {"dialog_id": dialog_id, "name": "新对话"}
response = requests.post("/v1/conversation/create", json=conv_data, headers=headers)
conversation_id = response.json()["data"]["id"]

# 发送消息
message_data = {
    "conversation_id": conversation_id,
    "messages": [{"role": "user", "content": "你好，请介绍一下上传的文档"}],
    "stream": False
}
response = requests.post("/v1/conversation/completion", 
                        json=message_data, headers=headers)
```

### 4. 使用外部 API Token
```python
# 创建 API Token
token_data = {"dialog_id": dialog_id}
response = requests.post("/api/v1/new_token", json=token_data, headers=headers)
api_token = response.json()["data"]["token"]

# 使用 API Token 创建会话
headers = {"Authorization": f"Bearer {api_token}"}
response = requests.get("/api/v1/new_conversation?user_id=test_user", headers=headers)
conv_id = response.json()["data"]["id"]

# 发送消息
message_data = {
    "conversation_id": conv_id,
    "messages": [{"role": "user", "content": "Hello"}]
}
response = requests.post("/api/v1/completion", json=message_data, headers=headers)
```

## 技术特性

### 文档解析支持
- **PDF**: 支持文本和图像提取
- **Word**: 支持 .docx 和 .doc 格式
- **PowerPoint**: 支持 .pptx 和 .ppt 格式
- **Excel**: 支持 .xlsx 和 .xls 格式
- **图片**: 支持 OCR 文字识别
- **音频**: 支持语音转文字
- **网页**: 支持 URL 内容抓取

### RAG 功能特性
- **智能分块**: 基于文档结构的智能分割
- **向量检索**: 支持语义相似度搜索
- **混合检索**: 向量检索 + 关键词检索
- **重排序**: 支持重排序模型优化检索结果
- **知识图谱**: 自动构建实体关系图
- **标签系统**: 支持文档和分块标签管理

### 流式响应
对话接口支持流式响应，实时返回生成内容：
```javascript
// 前端接收流式响应示例
const response = await fetch('/v1/conversation/completion', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({...data, stream: true})
});

const reader = response.body.getReader();
while (true) {
  const {done, value} = await reader.read();
  if (done) break;
  
  const chunk = new TextDecoder().decode(value);
  const lines = chunk.split('\n');
  for (const line of lines) {
    if (line.startsWith('data:')) {
      const data = JSON.parse(line.slice(5));
      console.log(data);
    }
  }
}
```

## 部署说明

### Docker 部署
```bash
# 克隆代码
git clone https://github.com/infiniflow/ragflow.git
cd ragflow

# 启动服务
docker-compose up -d
```

### 环境变量配置
```bash
# 必要的环境变量
SECRET_KEY=your_secret_key
DATABASE_URL=postgresql://user:pass@localhost/ragflow
REDIS_URL=redis://localhost:6379

# LLM 配置
LLM_FACTORY=OpenAI
API_KEY=your_openai_api_key
LLM_BASE_URL=https://api.openai.com/v1

# 文件存储配置
STORAGE_TYPE=minio  # 或 filesystem
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
```

### 生产环境建议
1. **负载均衡**: 使用 Nginx 进行负载均衡
2. **数据库**: 使用 PostgreSQL 作为主数据库
3. **缓存**: 使用 Redis 进行缓存和会话存储
4. **文件存储**: 使用 MinIO 或云存储服务
5. **日志**: 配置日志聚合和监控
6. **安全**: 配置 HTTPS 和防火墙规则

本文档涵盖了 RAGFlow 的主要 API 接口。如需更详细信息，请参考源代码或联系开发团队。
