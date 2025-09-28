# RAGFlow 数据库设计文档（中文版）

## 概述

RAGFlow 采用关系型数据库设计，支持 MySQL 和 PostgreSQL。数据库设计围绕 RAG（检索增强生成）的核心业务流程，包括用户管理、租户管理、知识库管理、文档处理、对话系统等模块。本文档详细介绍了每个表的结构、用途和使用方式。

## 数据库架构特点

### 技术特点
- **ORM框架**: 使用 Peewee ORM 进行数据库操作
- **多数据库支持**: 支持 MySQL 和 PostgreSQL
- **连接池**: 使用 PooledDatabase 进行连接池管理
- **事务支持**: 支持数据库事务和锁机制
- **自动时间戳**: 自动维护创建时间和更新时间

### 设计模式
- **多租户架构**: 通过 tenant_id 实现数据隔离
- **软删除**: 通过 status 字段实现逻辑删除
- **关联表设计**: 使用中间表处理多对多关系
- **JSON字段**: 使用 JSON 字段存储复杂配置和元数据

## 核心枚举和常量

### 状态枚举 (StatusEnum)
```python
class StatusEnum(Enum):
    VALID = "1"      # 有效
    INVALID = "0"    # 无效
```

### 用户角色 (UserTenantRole)
```python
class UserTenantRole(StrEnum):
    OWNER = 'owner'     # 拥有者
    ADMIN = 'admin'     # 管理员
    NORMAL = 'normal'   # 普通用户
    INVITE = 'invite'   # 邀请状态
```

### 文件类型 (FileType)
```python
class FileType(StrEnum):
    PDF = 'pdf'           # PDF文档
    DOC = 'doc'           # Word文档
    VISUAL = 'visual'     # 图片文件
    AURAL = 'aural'       # 音频文件
    VIRTUAL = 'virtual'   # 虚拟文件
    FOLDER = 'folder'     # 文件夹
    OTHER = "other"       # 其他类型
```

### LLM类型 (LLMType)
```python
class LLMType(StrEnum):
    CHAT = 'chat'                    # 对话模型
    EMBEDDING = 'embedding'          # 向量模型
    SPEECH2TEXT = 'speech2text'      # 语音转文字
    IMAGE2TEXT = 'image2text'        # 图像转文字
    RERANK = 'rerank'               # 重排序模型
    TTS = 'tts'                     # 文字转语音
```

### 任务状态 (TaskStatus)
```python
class TaskStatus(StrEnum):
    UNSTART = "0"    # 未开始
    RUNNING = "1"    # 运行中
    CANCEL = "2"     # 已取消
    DONE = "3"       # 已完成
    FAIL = "4"       # 失败
```

### 解析器类型 (ParserType)
```python
class ParserType(StrEnum):
    PRESENTATION = "presentation"    # 演示文稿
    LAWS = "laws"                   # 法律文档
    MANUAL = "manual"               # 手册
    PAPER = "paper"                 # 论文
    RESUME = "resume"               # 简历
    BOOK = "book"                   # 书籍
    QA = "qa"                       # 问答
    TABLE = "table"                 # 表格
    NAIVE = "naive"                 # 简单解析
    PICTURE = "picture"             # 图片
    ONE = "one"                     # 单一
    AUDIO = "audio"                 # 音频
    EMAIL = "email"                 # 邮件
    KG = "knowledge_graph"          # 知识图谱
    TAG = "tag"                     # 标签
```

## 数据表详细设计

## 1. 用户管理表

### 1.1 用户表 (user)
**用途**: 存储系统用户的基本信息和认证信息

| 字段名 | 类型 | 长度 | 是否空 | 默认值 | 索引 | 说明 |
|--------|------|------|--------|--------|------|------|
| id | VARCHAR | 32 | 否 | - | 主键 | 用户唯一标识 |
| access_token | VARCHAR | 255 | 是 | - | 是 | 访问令牌 |
| nickname | VARCHAR | 100 | 否 | - | 是 | 用户昵称 |
| password | VARCHAR | 255 | 是 | - | 是 | 密码哈希 |
| email | VARCHAR | 255 | 否 | - | 是 | 邮箱地址 |
| avatar | TEXT | - | 是 | - | 否 | 头像base64字符串 |
| language | VARCHAR | 32 | 是 | Chinese/English | 是 | 语言偏好 |
| color_schema | VARCHAR | 32 | 是 | Bright | 是 | 颜色主题 |
| timezone | VARCHAR | 64 | 是 | UTC+8 | 是 | 时区设置 |
| last_login_time | DATETIME | - | 是 | - | 是 | 最后登录时间 |
| is_authenticated | VARCHAR | 1 | 否 | 1 | 是 | 是否已认证 |
| is_active | VARCHAR | 1 | 否 | 1 | 是 | 是否激活 |
| is_anonymous | VARCHAR | 1 | 否 | 0 | 是 | 是否匿名 |
| login_channel | VARCHAR | - | 是 | - | 是 | 登录渠道 |
| status | VARCHAR | 1 | 是 | 1 | 是 | 用户状态 |
| is_superuser | BOOLEAN | - | 是 | FALSE | 是 | 是否超级用户 |
| create_time | BIGINT | - | 是 | - | 是 | 创建时间戳 |
| create_date | DATETIME | - | 是 | - | 是 | 创建日期 |
| update_time | BIGINT | - | 是 | - | 是 | 更新时间戳 |
| update_date | DATETIME | - | 是 | - | 是 | 更新日期 |

**使用场景**:
- 用户注册、登录、认证
- 用户信息管理和偏好设置
- OAuth 登录（GitHub、飞书）
- 权限控制和超级用户管理

**关联关系**:
- 通过 UserTenant 表与 Tenant 表建立多对多关系
- 通过 created_by 字段关联各种资源

### 1.2 租户表 (tenant)
**用途**: 多租户架构的核心表，每个用户或组织对应一个租户

| 字段名 | 类型 | 长度 | 是否空 | 默认值 | 索引 | 说明 |
|--------|------|------|--------|--------|------|------|
| id | VARCHAR | 32 | 否 | - | 主键 | 租户唯一标识 |
| name | VARCHAR | 100 | 是 | - | 是 | 租户名称 |
| public_key | VARCHAR | 255 | 是 | - | 是 | 公钥 |
| llm_id | VARCHAR | 128 | 否 | - | 是 | 默认LLM模型ID |
| embd_id | VARCHAR | 128 | 否 | - | 是 | 默认向量模型ID |
| asr_id | VARCHAR | 128 | 否 | - | 是 | 默认ASR模型ID |
| img2txt_id | VARCHAR | 128 | 否 | - | 是 | 默认图像转文字模型ID |
| rerank_id | VARCHAR | 128 | 否 | - | 是 | 默认重排序模型ID |
| tts_id | VARCHAR | 256 | 是 | - | 是 | 默认TTS模型ID |
| parser_ids | VARCHAR | 256 | 否 | - | 是 | 文档解析器列表 |
| credit | INTEGER | - | 否 | 512 | 是 | 积分余额 |
| status | VARCHAR | 1 | 是 | 1 | 是 | 租户状态 |
| create_time | BIGINT | - | 是 | - | 是 | 创建时间戳 |
| create_date | DATETIME | - | 是 | - | 是 | 创建日期 |
| update_time | BIGINT | - | 是 | - | 是 | 更新时间戳 |
| update_date | DATETIME | - | 是 | - | 是 | 更新日期 |

**使用场景**:
- 多租户数据隔离
- 租户级别的AI模型配置
- 资源使用量统计和计费
- 租户级别的系统配置

### 1.3 用户租户关系表 (user_tenant)
**用途**: 用户与租户的多对多关系表，支持用户在多个租户间切换

| 字段名 | 类型 | 长度 | 是否空 | 默认值 | 索引 | 说明 |
|--------|------|------|--------|--------|------|------|
| id | VARCHAR | 32 | 否 | - | 主键 | 关系唯一标识 |
| user_id | VARCHAR | 32 | 否 | - | 是 | 用户ID |
| tenant_id | VARCHAR | 32 | 否 | - | 是 | 租户ID |
| role | VARCHAR | 32 | 否 | - | 是 | 用户角色 |
| invited_by | VARCHAR | 32 | 否 | - | 是 | 邀请人ID |
| status | VARCHAR | 1 | 是 | 1 | 是 | 关系状态 |
| create_time | BIGINT | - | 是 | - | 是 | 创建时间戳 |
| create_date | DATETIME | - | 是 | - | 是 | 创建日期 |
| update_time | BIGINT | - | 是 | - | 是 | 更新时间戳 |
| update_date | DATETIME | - | 是 | - | 是 | 更新日期 |

**使用场景**:
- 租户成员管理
- 角色权限控制
- 邀请和协作功能
- 跨租户用户管理

### 1.4 邀请码表 (invitation_code)
**用途**: 管理租户邀请码，用于用户邀请和注册

| 字段名 | 类型 | 长度 | 是否空 | 默认值 | 索引 | 说明 |
|--------|------|------|--------|--------|------|------|
| id | VARCHAR | 32 | 否 | - | 主键 | 邀请码唯一标识 |
| code | VARCHAR | 32 | 否 | - | 是 | 邀请码 |
| visit_time | DATETIME | - | 是 | - | 是 | 访问时间 |
| user_id | VARCHAR | 32 | 是 | - | 是 | 使用者用户ID |
| tenant_id | VARCHAR | 32 | 是 | - | 是 | 所属租户ID |
| status | VARCHAR | 1 | 是 | 1 | 是 | 邀请码状态 |
| create_time | BIGINT | - | 是 | - | 是 | 创建时间戳 |
| create_date | DATETIME | - | 是 | - | 是 | 创建日期 |
| update_time | BIGINT | - | 是 | - | 是 | 更新时间戳 |
| update_date | DATETIME | - | 是 | - | 是 | 更新日期 |

## 2. LLM模型管理表

### 2.1 LLM工厂表 (llm_factories)
**用途**: 存储LLM提供商信息，如OpenAI、Anthropic等

| 字段名 | 类型 | 长度 | 是否空 | 默认值 | 索引 | 说明 |
|--------|------|------|--------|--------|------|------|
| name | VARCHAR | 128 | 否 | - | 主键 | LLM工厂名称 |
| logo | TEXT | - | 是 | - | 否 | logo base64字符串 |
| tags | VARCHAR | 255 | 否 | - | 是 | 支持的模型类型标签 |
| status | VARCHAR | 1 | 是 | 1 | 是 | 工厂状态 |
| create_time | BIGINT | - | 是 | - | 是 | 创建时间戳 |
| create_date | DATETIME | - | 是 | - | 是 | 创建日期 |
| update_time | BIGINT | - | 是 | - | 是 | 更新时间戳 |
| update_date | DATETIME | - | 是 | - | 是 | 更新日期 |

### 2.2 LLM模型表 (llm)
**用途**: 存储所有可用的LLM模型信息

| 字段名 | 类型 | 长度 | 是否空 | 默认值 | 索引 | 说明 |
|--------|------|------|--------|--------|------|------|
| llm_name | VARCHAR | 128 | 否 | - | 复合主键 | LLM模型名称 |
| model_type | VARCHAR | 128 | 否 | - | 是 | 模型类型 |
| fid | VARCHAR | 128 | 否 | - | 复合主键 | LLM工厂ID |
| max_tokens | INTEGER | - | 否 | 0 | 否 | 最大Token数 |
| tags | VARCHAR | 255 | 否 | - | 是 | 模型标签 |
| status | VARCHAR | 1 | 是 | 1 | 是 | 模型状态 |
| create_time | BIGINT | - | 是 | - | 是 | 创建时间戳 |
| create_date | DATETIME | - | 是 | - | 是 | 创建日期 |
| update_time | BIGINT | - | 是 | - | 是 | 更新时间戳 |
| update_date | DATETIME | - | 是 | - | 是 | 更新日期 |

**复合主键**: (fid, llm_name)

### 2.3 租户LLM配置表 (tenant_llm)
**用途**: 存储每个租户的LLM配置信息，包括API密钥等

| 字段名 | 类型 | 长度 | 是否空 | 默认值 | 索引 | 说明 |
|--------|------|------|--------|--------|------|------|
| tenant_id | VARCHAR | 32 | 否 | - | 复合主键 | 租户ID |
| llm_factory | VARCHAR | 128 | 否 | - | 复合主键 | LLM工厂名称 |
| llm_name | VARCHAR | 128 | 是 | - | 复合主键 | LLM模型名称 |
| model_type | VARCHAR | 128 | 是 | - | 是 | 模型类型 |
| api_key | VARCHAR | 1024 | 是 | - | 是 | API密钥 |
| api_base | VARCHAR | 255 | 是 | - | 否 | API基础URL |
| max_tokens | INTEGER | - | 否 | 8192 | 是 | 最大Token数 |
| used_tokens | INTEGER | - | 否 | 0 | 是 | 已使用Token数 |
| create_time | BIGINT | - | 是 | - | 是 | 创建时间戳 |
| create_date | DATETIME | - | 是 | - | 是 | 创建日期 |
| update_time | BIGINT | - | 是 | - | 是 | 更新时间戳 |
| update_date | DATETIME | - | 是 | - | 是 | 更新日期 |

**复合主键**: (tenant_id, llm_factory, llm_name)

**使用场景**:
- 租户级别的LLM配置管理
- API密钥安全存储
- Token使用量统计
- 多模型切换和管理

## 3. 知识库管理表

### 3.1 知识库表 (knowledgebase)
**用途**: 存储知识库的基本信息和配置

| 字段名 | 类型 | 长度 | 是否空 | 默认值 | 索引 | 说明 |
|--------|------|------|--------|--------|------|------|
| id | VARCHAR | 32 | 否 | - | 主键 | 知识库唯一标识 |
| avatar | TEXT | - | 是 | - | 否 | 头像base64字符串 |
| tenant_id | VARCHAR | 32 | 否 | - | 是 | 所属租户ID |
| name | VARCHAR | 128 | 否 | - | 是 | 知识库名称 |
| language | VARCHAR | 32 | 是 | Chinese/English | 是 | 语言设置 |
| description | TEXT | - | 是 | - | 否 | 知识库描述 |
| embd_id | VARCHAR | 128 | 否 | - | 是 | 向量模型ID |
| permission | VARCHAR | 16 | 否 | me | 是 | 权限设置 |
| created_by | VARCHAR | 32 | 否 | - | 是 | 创建者ID |
| doc_num | INTEGER | - | 否 | 0 | 是 | 文档数量 |
| token_num | INTEGER | - | 否 | 0 | 是 | Token总数 |
| chunk_num | INTEGER | - | 否 | 0 | 是 | 分块总数 |
| similarity_threshold | FLOAT | - | 否 | 0.2 | 是 | 相似度阈值 |
| vector_similarity_weight | FLOAT | - | 否 | 0.3 | 是 | 向量相似度权重 |
| parser_id | VARCHAR | 32 | 否 | naive | 是 | 默认解析器ID |
| parser_config | JSON | - | 否 | {...} | 否 | 解析器配置 |
| pagerank | INTEGER | - | 否 | 0 | 否 | PageRank值 |
| status | VARCHAR | 1 | 是 | 1 | 是 | 知识库状态 |
| create_time | BIGINT | - | 是 | - | 是 | 创建时间戳 |
| create_date | DATETIME | - | 是 | - | 是 | 创建日期 |
| update_time | BIGINT | - | 是 | - | 是 | 更新时间戳 |
| update_date | DATETIME | - | 是 | - | 是 | 更新日期 |

**使用场景**:
- 知识库创建和管理
- 文档检索配置
- 权限控制和共享
- 统计信息展示

## 4. 文档和文件管理表

### 4.1 文档表 (document)
**用途**: 存储知识库中文档的元数据和处理状态

| 字段名 | 类型 | 长度 | 是否空 | 默认值 | 索引 | 说明 |
|--------|------|------|--------|--------|------|------|
| id | VARCHAR | 32 | 否 | - | 主键 | 文档唯一标识 |
| thumbnail | TEXT | - | 是 | - | 否 | 缩略图base64 |
| kb_id | VARCHAR | 256 | 否 | - | 是 | 所属知识库ID |
| parser_id | VARCHAR | 32 | 否 | - | 是 | 解析器ID |
| parser_config | JSON | - | 否 | {...} | 否 | 解析器配置 |
| source_type | VARCHAR | 128 | 否 | local | 是 | 来源类型 |
| type | VARCHAR | 32 | 否 | - | 是 | 文件类型 |
| created_by | VARCHAR | 32 | 否 | - | 是 | 创建者ID |
| name | VARCHAR | 255 | 是 | - | 是 | 文档名称 |
| location | VARCHAR | 255 | 是 | - | 是 | 存储位置 |
| size | INTEGER | - | 否 | 0 | 是 | 文件大小 |
| token_num | INTEGER | - | 否 | 0 | 是 | Token数量 |
| chunk_num | INTEGER | - | 否 | 0 | 是 | 分块数量 |
| progress | FLOAT | - | 否 | 0 | 是 | 处理进度 |
| progress_msg | TEXT | - | 是 | - | 否 | 处理消息 |
| process_begin_at | DATETIME | - | 是 | - | 是 | 处理开始时间 |
| process_duation | FLOAT | - | 否 | 0 | 否 | 处理耗时 |
| meta_fields | JSON | - | 是 | {} | 否 | 元数据字段 |
| run | VARCHAR | 1 | 是 | 0 | 是 | 运行状态 |
| status | VARCHAR | 1 | 是 | 1 | 是 | 文档状态 |
| create_time | BIGINT | - | 是 | - | 是 | 创建时间戳 |
| create_date | DATETIME | - | 是 | - | 是 | 创建日期 |
| update_time | BIGINT | - | 是 | - | 是 | 更新时间戳 |
| update_date | DATETIME | - | 是 | - | 是 | 更新日期 |

**使用场景**:
- 文档上传和管理
- 文档解析状态跟踪
- 元数据存储和搜索
- 处理进度监控

### 4.2 文件表 (file)
**用途**: 通用文件系统，支持文件夹层次结构

| 字段名 | 类型 | 长度 | 是否空 | 默认值 | 索引 | 说明 |
|--------|------|------|--------|--------|------|------|
| id | VARCHAR | 32 | 否 | - | 主键 | 文件唯一标识 |
| parent_id | VARCHAR | 32 | 否 | - | 是 | 父文件夹ID |
| tenant_id | VARCHAR | 32 | 否 | - | 是 | 所属租户ID |
| created_by | VARCHAR | 32 | 否 | - | 是 | 创建者ID |
| name | VARCHAR | 255 | 否 | - | 是 | 文件/文件夹名称 |
| location | VARCHAR | 255 | 是 | - | 是 | 存储位置 |
| size | INTEGER | - | 否 | 0 | 是 | 文件大小 |
| type | VARCHAR | 32 | 否 | - | 是 | 文件类型 |
| source_type | VARCHAR | 128 | 否 | - | 是 | 来源类型 |
| create_time | BIGINT | - | 是 | - | 是 | 创建时间戳 |
| create_date | DATETIME | - | 是 | - | 是 | 创建日期 |
| update_time | BIGINT | - | 是 | - | 是 | 更新时间戳 |
| update_date | DATETIME | - | 是 | - | 是 | 更新日期 |

**使用场景**:
- 文件夹层次结构管理
- 通用文件上传和存储
- 文件组织和分类
- 个人文件空间

### 4.3 文件文档关联表 (file2document)
**用途**: 关联文件系统和知识库文档，一个文件可以对应多个文档

| 字段名 | 类型 | 长度 | 是否空 | 默认值 | 索引 | 说明 |
|--------|------|------|--------|--------|------|------|
| id | VARCHAR | 32 | 否 | - | 主键 | 关联唯一标识 |
| file_id | VARCHAR | 32 | 是 | - | 是 | 文件ID |
| document_id | VARCHAR | 32 | 是 | - | 是 | 文档ID |
| create_time | BIGINT | - | 是 | - | 是 | 创建时间戳 |
| create_date | DATETIME | - | 是 | - | 是 | 创建日期 |
| update_time | BIGINT | - | 是 | - | 是 | 更新时间戳 |
| update_date | DATETIME | - | 是 | - | 是 | 更新日期 |

## 5. 任务管理表

### 5.1 任务表 (task)
**用途**: 管理文档处理任务的执行状态

| 字段名 | 类型 | 长度 | 是否空 | 默认值 | 索引 | 说明 |
|--------|------|------|--------|--------|------|------|
| id | VARCHAR | 32 | 否 | - | 主键 | 任务唯一标识 |
| doc_id | VARCHAR | 32 | 否 | - | 是 | 关联文档ID |
| from_page | INTEGER | - | 否 | 0 | 否 | 起始页码 |
| to_page | INTEGER | - | 否 | 100000000 | 否 | 结束页码 |
| task_type | VARCHAR | 32 | 否 | - | 否 | 任务类型 |
| begin_at | DATETIME | - | 是 | - | 是 | 开始时间 |
| process_duation | FLOAT | - | 否 | 0 | 否 | 处理耗时 |
| progress | FLOAT | - | 否 | 0 | 是 | 任务进度 |
| progress_msg | TEXT | - | 是 | - | 否 | 进度消息 |
| retry_count | INTEGER | - | 否 | 0 | 否 | 重试次数 |
| digest | TEXT | - | 是 | - | 否 | 任务摘要 |
| chunk_ids | LONGTEXT | - | 是 | - | 否 | 分块ID列表 |
| create_time | BIGINT | - | 是 | - | 是 | 创建时间戳 |
| create_date | DATETIME | - | 是 | - | 是 | 创建日期 |
| update_time | BIGINT | - | 是 | - | 是 | 更新时间戳 |
| update_date | DATETIME | - | 是 | - | 是 | 更新日期 |

**使用场景**:
- 异步任务管理
- 进度跟踪和监控
- 错误重试机制
- 任务状态查询

## 6. 对话系统表

### 6.1 对话配置表 (dialog)
**用途**: 存储对话机器人的配置信息

| 字段名 | 类型 | 长度 | 是否空 | 默认值 | 索引 | 说明 |
|--------|------|------|--------|--------|------|------|
| id | VARCHAR | 32 | 否 | - | 主键 | 对话配置唯一标识 |
| tenant_id | VARCHAR | 32 | 否 | - | 是 | 所属租户ID |
| name | VARCHAR | 255 | 是 | - | 是 | 对话应用名称 |
| description | TEXT | - | 是 | - | 否 | 对话描述 |
| icon | TEXT | - | 是 | - | 否 | 图标base64 |
| language | VARCHAR | 32 | 是 | Chinese/English | 是 | 语言设置 |
| llm_id | VARCHAR | 128 | 否 | - | 否 | LLM模型ID |
| llm_setting | JSON | - | 否 | {...} | 否 | LLM配置参数 |
| prompt_type | VARCHAR | 16 | 否 | simple | 是 | 提示词类型 |
| prompt_config | JSON | - | 否 | {...} | 否 | 提示词配置 |
| similarity_threshold | FLOAT | - | 否 | 0.2 | 否 | 相似度阈值 |
| vector_similarity_weight | FLOAT | - | 否 | 0.3 | 否 | 向量相似度权重 |
| top_n | INTEGER | - | 否 | 6 | 否 | 检索数量 |
| top_k | INTEGER | - | 否 | 1024 | 否 | 候选数量 |
| do_refer | VARCHAR | 1 | 否 | 1 | 否 | 是否插入引用 |
| rerank_id | VARCHAR | 128 | 否 | - | 否 | 重排序模型ID |
| kb_ids | JSON | - | 否 | [] | 否 | 关联知识库ID列表 |
| status | VARCHAR | 1 | 是 | 1 | 是 | 对话状态 |
| create_time | BIGINT | - | 是 | - | 是 | 创建时间戳 |
| create_date | DATETIME | - | 是 | - | 是 | 创建日期 |
| update_time | BIGINT | - | 是 | - | 是 | 更新时间戳 |
| update_date | DATETIME | - | 是 | - | 是 | 更新日期 |

**使用场景**:
- 对话机器人配置
- RAG参数调优
- 多知识库联合检索
- 提示词工程

### 6.2 对话会话表 (conversation)
**用途**: 存储用户与对话机器人的会话记录

| 字段名 | 类型 | 长度 | 是否空 | 默认值 | 索引 | 说明 |
|--------|------|------|--------|--------|------|------|
| id | VARCHAR | 32 | 否 | - | 主键 | 会话唯一标识 |
| dialog_id | VARCHAR | 32 | 否 | - | 是 | 对话配置ID |
| name | VARCHAR | 255 | 是 | - | 是 | 会话名称 |
| message | JSON | - | 是 | - | 否 | 消息历史 |
| reference | JSON | - | 是 | [] | 否 | 引用信息 |
| user_id | VARCHAR | 255 | 是 | - | 是 | 用户ID |
| create_time | BIGINT | - | 是 | - | 是 | 创建时间戳 |
| create_date | DATETIME | - | 是 | - | 是 | 创建日期 |
| update_time | BIGINT | - | 是 | - | 是 | 更新时间戳 |
| update_date | DATETIME | - | 是 | - | 是 | 更新日期 |

**使用场景**:
- 对话历史记录
- 上下文维护
- 会话管理
- 用户行为分析

## 7. API管理表

### 7.1 API令牌表 (api_token)
**用途**: 管理外部API访问令牌

| 字段名 | 类型 | 长度 | 是否空 | 默认值 | 索引 | 说明 |
|--------|------|------|--------|--------|------|------|
| tenant_id | VARCHAR | 32 | 否 | - | 复合主键 | 租户ID |
| token | VARCHAR | 255 | 否 | - | 复合主键 | API令牌 |
| dialog_id | VARCHAR | 32 | 是 | - | 是 | 关联对话ID |
| source | VARCHAR | 16 | 是 | - | 是 | 来源类型 |
| beta | VARCHAR | 255 | 是 | - | 是 | Beta标识 |
| create_time | BIGINT | - | 是 | - | 是 | 创建时间戳 |
| create_date | DATETIME | - | 是 | - | 是 | 创建日期 |
| update_time | BIGINT | - | 是 | - | 是 | 更新时间戳 |
| update_date | DATETIME | - | 是 | - | 是 | 更新日期 |

**复合主键**: (tenant_id, token)

### 7.2 API对话表 (api_4_conversation)
**用途**: 存储通过API进行的对话记录和统计信息

| 字段名 | 类型 | 长度 | 是否空 | 默认值 | 索引 | 说明 |
|--------|------|------|--------|--------|------|------|
| id | VARCHAR | 32 | 否 | - | 主键 | 对话唯一标识 |
| dialog_id | VARCHAR | 32 | 否 | - | 是 | 对话配置ID |
| user_id | VARCHAR | 255 | 否 | - | 是 | 用户ID |
| message | JSON | - | 是 | - | 否 | 消息内容 |
| reference | JSON | - | 是 | [] | 否 | 引用信息 |
| tokens | INTEGER | - | 否 | 0 | 否 | 消耗Token数 |
| source | VARCHAR | 16 | 是 | - | 是 | 来源类型 |
| dsl | JSON | - | 是 | {} | 否 | DSL配置 |
| duration | FLOAT | - | 否 | 0 | 是 | 响应耗时 |
| round | INTEGER | - | 否 | 0 | 是 | 对话轮次 |
| thumb_up | INTEGER | - | 否 | 0 | 是 | 点赞数 |
| create_time | BIGINT | - | 是 | - | 是 | 创建时间戳 |
| create_date | DATETIME | - | 是 | - | 是 | 创建日期 |
| update_time | BIGINT | - | 是 | - | 是 | 更新时间戳 |
| update_date | DATETIME | - | 是 | - | 是 | 更新日期 |

**使用场景**:
- API使用统计
- 性能监控
- 用户行为分析
- 计费和限额管理

## 8. 画布功能表

### 8.1 用户画布表 (user_canvas)
**用途**: 存储用户创建的画布配置

| 字段名 | 类型 | 长度 | 是否空 | 默认值 | 索引 | 说明 |
|--------|------|------|--------|--------|------|------|
| id | VARCHAR | 32 | 否 | - | 主键 | 画布唯一标识 |
| avatar | TEXT | - | 是 | - | 否 | 头像base64 |
| user_id | VARCHAR | 255 | 否 | - | 是 | 用户ID |
| title | VARCHAR | 255 | 是 | - | 否 | 画布标题 |
| description | TEXT | - | 是 | - | 否 | 画布描述 |
| canvas_type | VARCHAR | 32 | 是 | - | 是 | 画布类型 |
| dsl | JSON | - | 是 | {} | 否 | DSL配置 |
| create_time | BIGINT | - | 是 | - | 是 | 创建时间戳 |
| create_date | DATETIME | - | 是 | - | 是 | 创建日期 |
| update_time | BIGINT | - | 是 | - | 是 | 更新时间戳 |
| update_date | DATETIME | - | 是 | - | 是 | 更新日期 |

### 8.2 画布模板表 (canvas_template)
**用途**: 存储系统预置的画布模板

| 字段名 | 类型 | 长度 | 是否空 | 默认值 | 索引 | 说明 |
|--------|------|------|--------|--------|------|------|
| id | VARCHAR | 32 | 否 | - | 主键 | 模板唯一标识 |
| avatar | TEXT | - | 是 | - | 否 | 头像base64 |
| title | VARCHAR | 255 | 是 | - | 否 | 模板标题 |
| description | TEXT | - | 是 | - | 否 | 模板描述 |
| canvas_type | VARCHAR | 32 | 是 | - | 是 | 画布类型 |
| dsl | JSON | - | 是 | {} | 否 | DSL配置 |
| create_time | BIGINT | - | 是 | - | 是 | 创建时间戳 |
| create_date | DATETIME | - | 是 | - | 是 | 创建日期 |
| update_time | BIGINT | - | 是 | - | 是 | 更新时间戳 |
| update_date | DATETIME | - | 是 | - | 是 | 更新日期 |

## 数据库关系图

```
用户模块:
User ←→ UserTenant ←→ Tenant
   ↓
InvitationCode

LLM模块:
LLMFactories → LLM ← TenantLLM → Tenant

知识库模块:
Tenant → Knowledgebase → Document
                    ↑        ↓
                 Dialog  File2Document
                    ↓        ↑
             Conversation   File

任务模块:
Document → Task

API模块:
Tenant → APIToken → Dialog
Dialog → API4Conversation

画布模块:
User → UserCanvas
CanvasTemplate (独立)
```

## 服务层设计

### 基础服务类 (CommonService)
所有服务类都继承自 `CommonService`，提供标准的CRUD操作：

```python
class CommonService:
    model = None  # 对应的数据模型
    
    # 基础CRUD操作
    @classmethod
    def query(cls, **kwargs)        # 查询
    @classmethod  
    def get_by_id(cls, pid)         # 根据ID获取
    @classmethod
    def save(cls, **kwargs)         # 保存
    @classmethod
    def insert(cls, **kwargs)       # 插入
    @classmethod
    def update_by_id(cls, pid, data) # 更新
    @classmethod
    def delete_by_id(cls, pid)      # 删除
```

### 具体服务类
- **UserService**: 用户管理服务
- **TenantService**: 租户管理服务
- **KnowledgebaseService**: 知识库管理服务
- **DocumentService**: 文档管理服务
- **FileService**: 文件管理服务
- **DialogService**: 对话管理服务
- **ConversationService**: 会话管理服务
- **TaskService**: 任务管理服务
- **LLMService**: LLM管理服务

## 数据库设计原则

### 1. 多租户隔离
- 所有核心表都包含 `tenant_id` 字段
- 通过 `tenant_id` 实现数据隔离
- 支持跨租户的用户和资源管理

### 2. 软删除机制
- 使用 `status` 字段实现软删除
- `status = "1"` 表示有效，`status = "0"` 表示删除
- 保留数据用于审计和恢复

### 3. 时间戳管理
- 所有表都包含 `create_time/create_date` 和 `update_time/update_date`
- 支持时间戳和日期两种格式
- 自动维护创建和更新时间

### 4. JSON字段应用
- 配置信息使用JSON字段存储
- 灵活的元数据存储
- 支持复杂的结构化数据

### 5. 索引设计
- 主键和外键字段建立索引
- 查询频繁的字段建立索引
- 时间字段建立索引支持范围查询

### 6. 权限控制
- 基于租户的数据访问控制
- 用户角色权限管理
- 资源级别的权限控制

## 性能优化建议

### 1. 分页查询
```python
# 使用LIMIT和OFFSET进行分页
docs, total = DocumentService.get_by_kb_id(
    kb_id, page, page_size, orderby, desc, keywords
)
```

### 2. 连接池配置
```python
# 使用连接池管理数据库连接
database_connection = PooledMySQLDatabase(
    db_name, 
    max_connections=20,
    stale_timeout=300
)
```

### 3. 事务管理
```python
# 使用事务确保数据一致性
with DB.atomic():
    # 批量操作
    for data in data_list:
        Model.create(**data)
```

### 4. 查询优化
- 避免 SELECT *，只查询需要的字段
- 使用索引字段进行WHERE条件
- 合理使用JOIN减少查询次数

## 迁移和维护

### 数据库迁移
系统提供自动迁移机制，在 `migrate_db()` 函数中定义：

```python
def migrate_db():
    migrator = DatabaseMigrator[settings.DATABASE_TYPE.upper()].value(DB)
    # 添加新字段
    migrate(migrator.add_column('table', 'field', FieldType()))
    # 修改字段类型
    migrate(migrator.alter_column_type('table', 'field', NewFieldType()))
```

### 备份策略
- 定期备份数据库
- 重要操作前创建快照
- 保留历史版本用于回滚

### 监控指标
- 数据库连接数
- 查询响应时间
- 存储空间使用
- 索引效率分析

## 使用示例

### 1. 创建知识库
```python
from api.db.services.knowledgebase_service import KnowledgebaseService

# 创建知识库
kb_data = {
    "name": "技术文档库",
    "tenant_id": "tenant_123",
    "description": "存储技术文档",
    "created_by": "user_456"
}
kb = KnowledgebaseService.insert(**kb_data)
```

### 2. 上传和解析文档
```python
from api.db.services.document_service import DocumentService
from api.db.services.task_service import TaskService

# 创建文档记录
doc_data = {
    "kb_id": "kb_123",
    "name": "API文档.pdf",
    "type": "pdf",
    "size": 1024000,
    "created_by": "user_456"
}
doc = DocumentService.insert(**doc_data)

# 创建解析任务
task_data = {
    "doc_id": doc.id,
    "task_type": "parse",
    "from_page": 1,
    "to_page": 100
}
task = TaskService.insert(**task_data)
```

### 3. 创建对话配置
```python
from api.db.services.dialog_service import DialogService

# 创建对话配置
dialog_data = {
    "name": "技术助手",
    "tenant_id": "tenant_123",
    "llm_id": "gpt-4",
    "kb_ids": ["kb_123", "kb_456"],
    "prompt_config": {
        "system": "你是一个技术助手",
        "prologue": "有什么技术问题可以帮助您？"
    }
}
dialog = DialogService.insert(**dialog_data)
```

这个数据库设计为RAGFlow提供了完整的数据支撑，涵盖了用户管理、多租户架构、知识库管理、文档处理、对话系统等所有核心功能。设计充分考虑了可扩展性、性能和数据安全性，为构建企业级RAG应用提供了坚实的基础。
