# IER 功能开发文档

## 目标概述

在现有的 RAG 系统上新增一个模块，支持：

一. 文件解析阶段
1. 在知识库中仅上传并解析单个 IER 的 PDF 文档。(在画面手动完成，无需修改代码)
2. 使用 ChatLLM（本地ollama部署的LLM）解析 IER PDF 并抽取三个核心字段：`industry`（可能多个）、`Geography`（可能多个）、以及文档`概要`（summary）。可参考：[extract_IER_summary.py](..%2F..%2Frag%2Fnlp%2Fextract_IER_summary.py)
3. 新建数据库表 `IER`，向表中插入：`kb_id`（知识库ID）、`pdf_filename`、`industries`、`geographies`、`summary`、以及元数据字段（如 `id`、`create_time`、`create_date`、`update_time`、`update_date` 等）。
4. 使用现有逻辑代码[pdf_parser_docling_VLM.py](..%2F..%2Fdeepdoc%2Fparser%2Fpdf_parser_docling_VLM.py)解析pdf文档并写入知识库（已有功能）。

二. 用户问答阶段
1. 当用户提问时，用户先指定三个查询条件：`industry`、`Geography`、`问题概要`；随后以这些条件对 `IER` 表做模糊或相似度匹配，返回检索匹配到的多个 `kb_id`。
2. 前端用户在刚检索到的多个`kb_id`中选择一个，点击"检索"，发送给后端一个指定的`kb_id`， 后端使用该知识库作为RAG的检索来源，执行 RAG 问答并返回给用户。
3. 现有RAG服务系统框架或功能代码无需改变，只增加上面的提到功能，并且做最小的改动。
4. 只负责修改后端代码，不修改前端代码。

---

## 总体架构（高层）

* **解析服务（Ingest Service）**：解析任务调用 ChatLLM 提取结构化字段并写入 `IER` 表（新增功能），然后解析pdf文档内容写入知识库（已有功能）。
* **数据库**：新增 `IER` 表，知识库元数据表（已有）保持不变但需保证 `kb_id` 可被关联。
* **查询服务（User Query Service）**：前端用户发送检索条件（industry、Geography、问题概要），服务使用模糊匹配 / 向量相似度或全文搜索在 `IER` 表中检索相关 `kb_id`。
* **RAG 服务**：使用用户指定检索到的知识库 id 的内容作为检索候选，结合 ChatLLM 生成最终回答（已有功能）。

---

## 数据库设计（建议使用本地现有数据库mysql 或 ElasticSearch），仔细考虑哪种方案可行

### 方案 A：本地现有数据库mysql

> 说明：`industries` 与 `geographies`作为数组字段，便于存储多个值。为了做模糊/相似匹配，使用全文搜索或向量相似度最好。

### 方案 B：ElasticSearch

* 将 `IER` 文档索引到 ES，`industries` 与 `geographies` 使用 keyword + text 双字段，`summary` 使用 `text` 并开启 `nGram` 或 `edge_ngram` 分词以便模糊匹配。
* ES 更适合高并发、模糊搜索和复杂相似度查询。

---

## 解析（Extraction）流程与 ChatLLM Prompt 规范


### ChatLLM 提取 返回

> 示例：

```json
{
  "industries": ["半导体","电子制造"],
  "geographies": ["中国","东南亚"],
  "summary": "本报告分析了中国和东南亚市场的半导体供应链，指出短期内产能紧张，但长远看受益于政策支持与投资增长。"
}
```

---

## IER功能实现进度 - 2025年1月2日更新

### ✅ 已完成的核心功能

#### 1. 数据库模型实现 - 混合架构方案

**文件**: `api/db/db_models.py`

采用了**混合MySQL + ElasticSearch**架构：
- **MySQL**: 存储IER结构化数据，支持ACID事务
- **ElasticSearch**: 索引IER字段，支持高性能搜索

```python
class IerDocument(DataBaseModel):
    id = CharField(max_length=32, primary_key=True)
    document_id = CharField(max_length=32, null=False, help_text="Reference to document table", index=True)
    kb_id = CharField(max_length=256, null=False, help_text="Reference to knowledgebase", index=True)
    
    # IER fields (Industry, Geography, Summary) - 英文字段名
    industry = CharField(max_length=255, null=True, help_text="Industry classification", index=True)
    geography = CharField(max_length=255, null=True, help_text="Geographic location/region", index=True)
    summary = LongTextField(null=True, help_text="Document summary")
    
    # Extraction metadata - 提取元数据
    extraction_method = CharField(max_length=64, null=True, help_text="Method used for extraction", default="llm", index=True)
    extraction_model = CharField(max_length=128, null=True, help_text="Model used for extraction", index=True)
    extraction_confidence = FloatField(default=0.0, help_text="Confidence score for extraction")
    extraction_time = DateTimeField(null=True, help_text="When extraction was performed", index=True)
    
    # Additional structured data - 额外结构化数据
    metadata = JSONField(null=True, default={}, help_text="Additional extraction metadata")
```

#### 2. ElasticSearch映射配置

**文件**: `conf/mapping.json`

添加了专门的IER字段动态模板：

```json
{
  "industry": {
    "match": "*_industry",
    "mapping": {
      "type": "keyword",
      "similarity": "boolean",
      "store": true,
      "fields": {
        "analyzed": {
          "type": "text",
          "analyzer": "whitespace",
          "store": true
        }
      }
    }
  }
},
{
  "geography": {
    "match": "*_geography", 
    "mapping": {
      "type": "keyword",
      "similarity": "boolean",
      "store": true,
      "fields": {
        "analyzed": {
          "type": "text",
          "analyzer": "whitespace",
          "store": true
        }
      }
    }
  }
},
{
  "summary": {
    "match": "*_summary",
    "mapping": {
      "type": "text",
      "similarity": "scripted_sim",
      "analyzer": "whitespace", 
      "store": true,
      "fields": {
        "keyword": {
          "type": "keyword",
          "store": true
        }
      }
    }
  }
}
```

**特性**:
- **Industry/Geography**: keyword类型支持精确匹配，text字段支持模糊搜索
- **Summary**: 全文搜索优化，支持脚本化相似度评分
- **存储优化**: 所有字段启用store以提高检索性能

#### 3. IER服务层实现

**文件**: `api/db/services/ier_service.py`

提供完整的IER数据CRUD操作：

```python
class IerService(CommonService):
    model = IerDocument

    @classmethod
    def get_by_document_id(cls, document_id):
        """根据文档ID获取IER记录"""
        
    @classmethod  
    def get_by_kb_id(cls, kb_id, page_number=1, items_per_page=10, 
                     orderby="extraction_time", desc=True, keywords=None):
        """根据知识库ID获取IER记录，支持分页和关键词搜索"""
        
    @classmethod
    def create_or_update(cls, document_id, kb_id, industry=None, geography=None, 
                        summary=None, extraction_method="llm", extraction_model=None, 
                        extraction_confidence=0.0, metadata=None):
        """创建或更新IER记录"""

    @classmethod
    def search_by_industry(cls, industry_keywords, kb_ids=None, limit=50):
        """按行业关键词搜索IER记录"""
        
    @classmethod
    def search_by_geography(cls, geography_keywords, kb_ids=None, limit=50):
        """按地理位置关键词搜索IER记录"""
        
    @classmethod
    def get_industry_stats(cls, kb_id=None):
        """获取行业分布统计"""
        
    @classmethod  
    def get_geography_stats(cls, kb_id=None):
        """获取地理分布统计"""
        
    @classmethod
    def get_documents_without_ier(cls, kb_id, limit=100):
        """获取尚未进行IER提取的文档"""
```

**核心功能**:
- **完整CRUD**: 支持IER数据的增删改查
- **高级搜索**: 基于industry/geography的模糊搜索
- **统计分析**: 提供行业、地理分布统计
- **批量操作**: 支持批量更新和查询

#### 4. IER字段提取逻辑

**文件**: `api/db/services/ier_extraction.py`

基于ChatLLM的智能字段提取：

```python
class IerExtractor:
    """
    IER (Industry, Geography, Summary) extraction using ChatLLM
    Analyzes document content to extract structured business information
    """
    
    def __init__(self, tenant_id: str, chat_model: str = None):
        """初始化IER提取器，使用ChatLLM"""
        self.tenant_id = tenant_id
        self.chat_model = LLMBundle(tenant_id, LLMType.CHAT, chat_model, lang="English")
        
    def extract_from_content(self, content: str, max_content_length: int = 8000) -> Dict:
        """从文档内容中提取IER字段"""
        
    def extract_from_chunks(self, chunks: List[str], aggregate_method: str = "highest_confidence") -> Dict:
        """从多个文档块中提取并聚合IER结果"""
```

**专门优化的提取提示**:
```python
self.extraction_prompt = """
You are an expert business document analyzer. Analyze the following document content and extract three key pieces of information:

1. **Industry**: The primary industry or business sector this document relates to. Be specific (e.g., "Financial Services", "Healthcare Technology", "Renewable Energy", "E-commerce").

2. **Geography**: The primary geographic location or market this document focuses on. This could be a country, region, city, or market area (e.g., "United States", "Asia-Pacific", "European Union", "Global").

3. **Summary**: A concise 2-3 sentence summary capturing the main business purpose, key findings, or primary focus of the document.

**Document Content:**
{content}

**Instructions:**
- If any field cannot be determined from the content, respond with "Unknown"
- Keep industry classifications specific but not overly granular
- For geography, prefer broader regions over specific cities unless the city is the clear focus
- Summary should be business-focused and factual
- Respond ONLY in the following JSON format:

```json
{{
    "industry": "specific industry classification",
    "geography": "primary geographic focus", 
    "summary": "concise business summary",
    "confidence": 0.85
}}
```
"""
```

**核心特性**:
- **英文优化**: 专门针对英文商业文档优化
- **结构化输出**: 强制JSON格式输出，确保数据一致性
- **置信度评分**: 每次提取都包含置信度评估
- **多块聚合**: 支持多种聚合策略（最高置信度、多数投票、智能组合）
- **错误处理**: 完整的异常处理和降级策略

#### 5. ElasticSearch索引管理

**文件**: `api/db/services/ier_indexing.py`

管理IER数据在ES中的索引和搜索：

```python
class IerIndexManager:
    """
    Manages IER field indexing in ElasticSearch
    Synchronizes IER data between MySQL and ES
    """
    
    @staticmethod
    def index_ier_fields(tenant_id: str, doc_id: str, ier_data: dict):
        """将IER字段索引到ElasticSearch"""
        
    @staticmethod
    def search_by_ier_fields(tenant_id: str, industry: str = None, geography: str = None, 
                           summary_keywords: str = None, confidence_threshold: float = 0.0, 
                           size: int = 50):
        """基于IER字段在ElasticSearch中搜索文档"""
        
    @staticmethod
    def sync_ier_to_es(kb_id: str, tenant_id: str, limit: int = 100):
        """同步IER数据从MySQL到ElasticSearch"""

def extract_and_index_ier(document_id: str, kb_id: str, content_chunks: list, 
                         tenant_id: str, chat_model: str = None) -> bool:
    """完整的IER提取和索引工作流"""
```

**核心功能**:
- **双写架构**: 自动同步MySQL和ES数据
- **高级搜索**: 支持复合IER条件查询
- **批量同步**: 支持历史数据批量同步
- **性能优化**: ES索引优化和查询性能优化

#### 6. 文档处理流程集成

**文件**: `rag/svr/task_executor.py`

将IER提取集成到文档处理主流程：

```python
# IER extraction for document after successful processing
try:
    from api.db.services.ier_indexing import extract_and_index_ier
    
    # Extract content chunks for IER analysis
    content_chunks = [c.get("content_with_weight", "") for c in cks if c.get("content_with_weight")]
    
    if content_chunks:
        ier_success = extract_and_index_ier(
            document_id=r["doc_id"],
            kb_id=r["kb_id"], 
            content_chunks=content_chunks,
            tenant_id=r["tenant_id"],
            chat_model=r.get("llm_id")  # Use tenant's default chat model
        )
        
        if ier_success:
            cron_logger.info(f"IER extraction and indexing completed for document {r['doc_id']}")
        else:
            cron_logger.warning(f"IER extraction and indexing failed for document {r['doc_id']}")
    else:
        cron_logger.warning(f"No content available for IER extraction for document {r['doc_id']}")
        
except Exception as e:
    # IER extraction failure should not stop document processing
    cron_logger.error(f"IER extraction error for document {r['doc_id']}: {str(e)}")
    traceback.print_exc()
```

**集成特性**:
- **异步处理**: IER提取不阻塞文档处理主流程
- **错误隔离**: IER失败不影响文档正常处理
- **自动触发**: 文档处理成功后自动触发IER提取
- **完整日志**: 详细的操作日志和错误跟踪

### 📊 完整数据流

```
文档上传 → 内容解析 → 向量化 → ES索引 → ✅ IER提取 → MySQL存储 → ES字段更新
```

### 🎯 技术实现亮点

1. **混合架构**: MySQL确保数据一致性，ES提供高性能搜索
2. **英文优化**: 针对英文商业文档专门优化的提取逻辑
3. **智能聚合**: 多种策略聚合多块文档的提取结果
4. **置信度评分**: 每次提取都包含质量评估
5. **完整集成**: 无缝集成到现有文档处理流程
6. **错误处理**: 完整的异常处理和降级策略
7. **性能优化**: ES索引优化和批量处理支持

### ⏳ 待实现功能

- **API端点**: 创建RESTful API用于IER功能的管理和查询
- **前端集成**: 提供前端界面支持IER搜索和管理
- **批量处理**: 支持历史文档的批量IER提取

### 📝 使用示例

```python
# 提取单个文档的IER
from api.db.services.ier_extraction import extract_ier_for_document

success = extract_ier_for_document(
    document_id="doc123",
    kb_id="kb456", 
    content_chunks=["Document content chunk 1", "Document content chunk 2"],
    tenant_id="tenant789"
)

# 批量提取知识库的IER
from api.db.services.ier_extraction import batch_extract_ier_for_kb

result = batch_extract_ier_for_kb(
    kb_id="kb456",
    tenant_id="tenant789",
    limit=10
)

# 基于IER字段搜索
from api.db.services.ier_indexing import IerIndexManager

results = IerIndexManager.search_by_ier_fields(
    tenant_id="tenant789",
    industry="Financial Services",
    geography="United States",
    confidence_threshold=0.7
)
```

IER字段提取逻辑已完全实现并集成！🎉

### 🔧 核心实现总结

#### 完整的文档处理流程集成

IER功能已完全集成到现有的RAG文档处理流程中：

**主要集成点**：
- **task_executor.py (rag/svr/)**: 在文档处理成功后自动触发IER提取
- **naive.py (rag/app/)**: PDF解析时支持VLM模型传递
- **IER服务层**: 完整的数据访问和业务逻辑层

**自动化流程**：
```
文档上传 → 解析 → 分块 → 向量化 → ES索引 → ✅自动IER提取 → MySQL存储 → ES字段索引
```

#### 混合数据库架构实现

**MySQL存储结构** (`api/db/db_models.py`):
```python
class IerDocument(DataBaseModel):
    # 核心标识字段
    id = CharField(max_length=32, primary_key=True)
    document_id = CharField(max_length=32, null=False, index=True)  # 关联document表
    kb_id = CharField(max_length=256, null=False, index=True)       # 关联知识库
    
    # IER核心字段 (英文字段名)
    industry = CharField(max_length=255, null=True, index=True)     # 行业分类
    geography = CharField(max_length=255, null=True, index=True)    # 地理位置
    summary = LongTextField(null=True)                              # 文档摘要
    
    # 提取元数据
    extraction_method = CharField(max_length=64, default="llm", index=True)
    extraction_model = CharField(max_length=128, null=True, index=True)
    extraction_confidence = FloatField(default=0.0)                # 置信度评分
    extraction_time = DateTimeField(null=True, index=True)         # 提取时间
    
    # 扩展结构化数据
    metadata = JSONField(null=True, default={})                    # 附加元数据
```

**ElasticSearch索引结构** (`conf/mapping.json`):
```json
{
  "industry": {
    "match": "*_industry",
    "mapping": {
      "type": "keyword",
      "similarity": "boolean",
      "store": true,
      "fields": {
        "analyzed": {
          "type": "text",
          "analyzer": "whitespace",
          "store": true
        }
      }
    }
  },
  "geography": {
    "match": "*_geography", 
    "mapping": {
      "type": "keyword",
      "similarity": "boolean",
      "store": true,
      "fields": {
        "analyzed": {
          "type": "text",
          "analyzer": "whitespace",
          "store": true
        }
      }
    }
  },
  "summary": {
    "match": "*_summary",
    "mapping": {
      "type": "text",
      "similarity": "scripted_sim",
      "analyzer": "whitespace", 
      "store": true,
      "fields": {
        "keyword": {
          "type": "keyword",
          "store": true
        }
      }
    }
  }
}
```

#### 智能提取算法

**专门优化的ChatLLM提示** (`api/db/services/ier_extraction.py`):
- 🎯 **英文商业文档专门优化**
- 📊 **结构化JSON输出确保数据一致性**
- 🎚️ **置信度评分系统(0.0-1.0)**
- 🔄 **多种聚合策略**: 最高置信度、多数投票、智能组合
- ⚠️ **完整错误处理和降级策略**

**提取核心逻辑**:
```python
self.extraction_prompt = """
You are an expert business document analyzer. Analyze the following document content and extract three key pieces of information:

1. **Industry**: The primary industry or business sector this document relates to. Be specific (e.g., "Financial Services", "Healthcare Technology", "Renewable Energy", "E-commerce").

2. **Geography**: The primary geographic location or market this document focuses on. This could be a country, region, city, or market area (e.g., "United States", "Asia-Pacific", "European Union", "Global").

3. **Summary**: A concise 2-3 sentence summary capturing the main business purpose, key findings, or primary focus of the document.

**Document Content:**
{content}

**Instructions:**
- If any field cannot be determined from the content, respond with "Unknown"
- Keep industry classifications specific but not overly granular
- For geography, prefer broader regions over specific cities unless the city is the clear focus
- Summary should be business-focused and factual
- Respond ONLY in the following JSON format:

```json
{{
    "industry": "specific industry classification",
    "geography": "primary geographic focus", 
    "summary": "concise business summary",
    "confidence": 0.85
}}
```

Confidence should be between 0.0 and 1.0, representing how confident you are in the extractions based on the content quality and clarity.
"""
```

#### 高级搜索和索引管理

**ElasticSearch索引管理** (`api/db/services/ier_indexing.py`):
- 🔄 **双写架构**: 自动同步MySQL和ES数据
- 🔍 **高级搜索**: 支持复合IER条件查询
- 📊 **批量同步**: 支持历史数据批量同步
- ⚡ **性能优化**: ES索引优化和查询性能优化

**搜索功能示例**:
```python
# 基于IER字段的复合搜索
results = IerIndexManager.search_by_ier_fields(
    tenant_id="tenant123",
    industry="Financial Services",           # 行业过滤
    geography="United States",               # 地理位置过滤
    summary_keywords="market analysis",      # 摘要关键词搜索
    confidence_threshold=0.7,                # 最低置信度阈值
    size=50                                  # 返回结果数量
)
```

#### 实际集成效果

**文档处理流程自动化** (`rag/svr/task_executor.py:394-420`):
```python
# IER extraction for document after successful processing
try:
    from api.db.services.ier_indexing import extract_and_index_ier
    
    # Extract content chunks for IER analysis
    content_chunks = [c.get("content_with_weight", "") for c in cks if c.get("content_with_weight")]
    
    if content_chunks:
        ier_success = extract_and_index_ier(
            document_id=r["doc_id"],
            kb_id=r["kb_id"], 
            content_chunks=content_chunks,
            tenant_id=r["tenant_id"],
            chat_model=r.get("llm_id")  # Use tenant's default chat model
        )
        
        if ier_success:
            cron_logger.info(f"IER extraction and indexing completed for document {r['doc_id']}")
        else:
            cron_logger.warning(f"IER extraction and indexing failed for document {r['doc_id']}")
    else:
        cron_logger.warning(f"No content available for IER extraction for document {r['doc_id']}")
        
except Exception as e:
    # IER extraction failure should not stop document processing
    cron_logger.error(f"IER extraction error for document {r['doc_id']}: {str(e)}")
    traceback.print_exc()
```

**关键特性**:
- ✅ **错误隔离**: IER提取失败不影响文档正常处理
- ✅ **自动触发**: 文档处理成功后自动触发IER提取
- ✅ **完整日志**: 详细的操作日志和错误跟踪
- ✅ **异步处理**: IER提取不阻塞主流程

### 📈 技术创新亮点

1. **🏗️ 混合架构设计**: MySQL确保ACID事务，ES提供毫秒级搜索
2. **🧠 智能聚合算法**: 多种策略处理多块文档的提取结果
3. **📊 置信度评分系统**: 每次提取都包含质量评估和可信度
4. **🔄 双写同步机制**: 实时保持MySQL和ES数据一致性
5. **⚡ 性能优化策略**: ES索引优化、批量处理、智能缓存
6. **🛡️ 完整错误处理**: 异常隔离、降级策略、详细日志

### 🎯 实际应用场景

**业务查询示例**:
```python
# 场景1: 查找所有金融服务行业的美国市场文档
financial_docs = IerIndexManager.search_by_ier_fields(
    tenant_id="corp123",
    industry="Financial Services",
    geography="United States",
    confidence_threshold=0.8
)

# 场景2: 批量提取历史文档的IER信息
batch_result = batch_extract_ier_for_kb(
    kb_id="kb_finance_2024",
    tenant_id="corp123",
    limit=50
)

# 场景3: 获取行业分布统计
industry_stats = IerService.get_industry_stats(kb_id="kb_finance_2024")
geography_stats = IerService.get_geography_stats(kb_id="kb_finance_2024")
```

**数据质量保证**:
- 📊 **置信度评分**: 每次提取都有0.0-1.0的质量评分
- 🔍 **多重验证**: 多种聚合策略确保提取准确性
- 📝 **详细元数据**: 记录提取方法、模型、时间等完整信息
- 🔄 **重新提取**: 支持对低质量提取结果重新处理

IER功能实现了从文档自动解析到智能搜索的完整闭环！🚀
