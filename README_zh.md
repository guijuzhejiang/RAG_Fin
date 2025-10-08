# RAG_Fin 智能文档检索与分析系统

<div align="center">

![RAG_Fin](https://img.shields.io/badge/RAG_Fin-v2.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.11+-green.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)
![Build](https://img.shields.io/badge/Build-Docker-blue.svg)

基于 RAGFlow 的二代英文文档智能检索系统，集成视觉语言模型的先进文档解析技术

[English](README.md) | [中文文档](README_zh.md) | [日本語](README_ja.md) | [한국어](README_ko.md)

</div>

## 📖 项目概述

RAG_Fin 是一个专为英文文档处理优化的**第二代 RAG 系统**，基于原始的 RAGFlow 架构进行了重大改进。该系统特别针对金融、投资和商业分析领域的文档处理需求，提供高精度的智能检索和分析服务。

### 🎯 核心特性

- **🧠 VLM 增强解析**: 集成视觉语言模型（VLM）革命性提升 PDF 解析质量
- **🌍 英文优化处理**: 专门针对英文文档的 NLP 处理和分词优化
- **⚡ 双服务架构**: API 服务与计算服务分离，提供更好的可扩展性
- **📊 智能分析**: 支持 IER（行业-经济-地区）三维智能检索
- **🔍 多模态理解**: 图像、表格、文本的统一理解和处理
- **📈 实时处理**: 异步任务处理，实时进度反馈

## 🚀 相比原版 RAGFlow 的主要改进

### 1. 语言处理升级
- **主要目标**: 英文文档和查询（原为中文）
- **NLP 引擎**: 集成 NLTK 英文处理组件
  - PorterStemmer 词干提取
  - WordNetLemmatizer 词形还原
  - `word_tokenize` 英文分词
- **混合分词器**: `rag/nlp/rag_tokenizer.py` 保留中文字典但增加英文检测和处理

### 2. VLM PDF 解析器创新
- **文件位置**: `deepdoc/parser/pdf_parser_unstructured_VLM.py`
- **核心优势**: 用先进的视觉语言模型替代传统 OCR
- **关键功能**:
  - Unstructured 库集成，提供高质量 PDF 解析
  - VLM 图像描述和理解
  - 图像预处理（对比度、锐度、降噪）
  - 智能页眉页脚过滤
  - 表格提取与 HTML 结构保持
  - 感知哈希图像去重

### 3. 待升级组件
- Word 解析器 (`docx_parser.py`) - 仍使用原始逻辑
- PowerPoint 解析器 (`ppt_parser.py`) - 尚未更新
- Excel 解析器 (`excel_parser.py`) - 等待 VLM 集成

## 🏗️ 系统架构

### 双服务架构设计

RAG_Fin 采用**双服务架构**实现更好的可扩展性和资源管理：

```
用户请求 → API 服务 (9380) 
         ↓
    Redis Streams
         ↓
    Task Executor
         ↓
    Document Parsing (naive.py)
         ↓
    VLM Processing (for PDFs)
         ↓
    ElasticSearch Index
```

#### 1. API 服务 (`api/ragflow_server.py`)
- Flask 基础的 REST API 服务器
- 处理端口 9380 的 HTTP 请求
- 管理用户认证和会话
- 调度文档处理任务
- 实时更新处理进度

#### 2. RAG 计算服务 (`rag/svr/task_executor.py`)
- 异步任务处理引擎
- 从 Redis 队列消费任务
- 执行文档解析、分块和向量化
- 管理三种模型类型：
  - **嵌入模型**: 文本向量化
  - **对话模型**: 文本生成和摘要
  - **VLM 模型** (IMAGE2TEXT): 图像理解和描述
- 将结果存储到 ElasticSearch

### 核心组件结构

```
api/                 # Flask REST API 层
├── apps/           # API 端点
├── db/             # 数据库模型和服务
├── schemas/        # Pydantic 模式
└── ragflow_server.py  # 主 API 服务器入口

rag/                # RAG 核心引擎
├── nlp/           # NLP 处理（英文优化）
├── app/           # 文档处理模板
│   └── naive.py   # 主文档解析调度器
├── llm/           # LLM 和嵌入模型
│   └── cv_model.py # 视觉模型包括 OllamaCV
└── svr/           
    └── task_executor.py # 异步任务处理服务

deepdoc/           # 文档解析引擎
├── parser/        # 格式特定解析器
│   └── pdf_parser_unstructured_VLM.py  # 新增: VLM 增强解析器
└── vision/        # 布局识别和 OCR
```

### 服务依赖

- **ElasticSearch**: 向量存储和全文搜索
- **MySQL**: 业务数据和元数据
- **Redis**: 缓存和任务队列管理（使用 Redis Streams 进行任务分发）
- **MinIO**: 文档和提取图像的对象存储
- **Ollama**: 本地 VLM 部署（通过 OllamaCV 类）

## 🛠️ 快速开始

### 系统要求

- **Python**: >= 3.11, < 3.12
- **Node.js**: >= 18.0 (前端开发)
- **Docker**: >= 24.0.0
- **Docker Compose**: >= 2.26.1
- **硬件**: CPU >= 4核, RAM >= 16GB, 磁盘 >= 50GB

### 🐳 Docker 部署（推荐）

#### 1. 克隆项目
```bash
git clone <repository-url>
cd RAG_Fin
```

#### 2. 启动完整系统
```bash
cd docker
docker compose up -d
```

#### 3. 访问服务
- **Web 界面**: http://localhost
- **API 接口**: http://localhost:9380
- **MinIO 控制台**: http://localhost:9000

### 🔧 本地开发环境

#### 1. 启动基础服务
```bash
cd docker
docker compose -f docker-compose-base.yml up -d
```

#### 2. 配置主机映射
```bash
echo "127.0.0.1 es01 mysql minio redis" >> /etc/hosts
```

#### 3. 安装 Python 依赖
```bash
# 使用 Poetry 安装依赖
poetry install --sync --no-root

# 激活虚拟环境
source .venv/bin/activate
export PYTHONPATH=$(pwd)
```

#### 4. 启动后端服务
```bash
# 启动 API 服务
python api/ragflow_server.py

# 新终端启动计算服务
python rag/svr/task_executor.py
```

#### 5. 启动前端服务
```bash
cd web
npm install --force
npm run dev
```

## 📋 使用指南

### VLM PDF 解析器

#### 核心功能
```python
from deepdoc.parser.pdf_parser_unstructured_VLM import PdfParserVLM

# 初始化解析器（可选 VLM 模型）
parser = PdfParserVLM(vlm_mdl=your_vlm_model)

# 解析 PDF 文档
sections, tables = parser("document.pdf", need_image=True)
print(f"提取了 {len(sections)} 个段落和 {len(tables)} 个表格")
```

#### 处理流程
1. **PDF 加载**: 使用 unstructured 的 `partition_pdf`，策略为 `hi_res`
2. **元素分类**: 文本保持原样，图像用 VLM 描述，表格提取 HTML
3. **图像增强**: VLM 识别前的预处理
4. **内容提取**: 图像用 VLM，表格保持 HTML，文本直接提取
5. **去重策略**: 感知哈希避免重复图像/表格

### 英文 NLP 处理

#### 分词器配置
```python
from rag.nlp.rag_tokenizer import RagTokenizer

tokenizer = RagTokenizer()

# 自动检测英文文本（>60% ASCII 字符）
english_text = "Financial analysis of market trends"
tokens = tokenizer.tokenize(english_text)
print(f"分词结果: {tokens}")
```

#### 语言检测特性
- 自动检测英文 vs 中文内容
- 英文文本绕过中文分词
- 使用 NLTK 工具进行英文特定处理

### IER 智能检索系统

系统支持**行业-经济-地区**三维智能检索：

#### 演示界面
```bash
# 访问演示页面
open demo/ier_search_demo.html

# 或使用本地服务器
cd demo
python -m http.server 8000
# 访问 http://localhost:8000/ier_search_demo.html
```

#### API 使用
```javascript
// POST /api/ier/search
{
  "industry": "technology",
  "geography": "north-america", 
  "parties": "Microsoft, Activision",
  "question": "并购案例分析",
  "language": "zh"
}
```

## 🧪 测试与调试

### 服务健康检查
```bash
# 检查 API 服务
curl http://localhost:9380/api/health

# 检查 ElasticSearch
curl http://localhost:9200/_cluster/health

# 检查 Redis
redis-cli ping

# 检查 MinIO
mc admin info minio/

# 检查 Ollama VLM 模型
curl http://localhost:11434/api/tags
```

### VLM 解析器测试
```python
# 测试新的 VLM PDF 解析器
from deepdoc.parser.pdf_parser_unstructured_VLM import PdfParserVLM

parser = PdfParserVLM(vlm_mdl=your_vlm_model)
sections, tables = parser("test.pdf", need_image=True)
print(f"提取了 {len(sections)} 个段落和 {len(tables)} 个表格")
```

### 英文 NLP 测试
```python
from rag.nlp.rag_tokenizer import RagTokenizer

tokenizer = RagTokenizer()
english_text = "This is an English document about financial analysis."
tokens = tokenizer.tokenize(english_text)
print(f"分词结果: {tokens}")
```

### 任务监控
```bash
# 监控 Redis Streams 任务队列
redis-cli xinfo stream rag_task_stream

# 查看待处理任务
redis-cli xlen rag_task_stream

# 查看消费者组信息
redis-cli xinfo groups rag_task_stream
```

## 📊 性能指标

### 处理性能
- **PDF VLM 解析**: 2-5 秒/页
- **PDF 传统解析**: 0.5-1 秒/页
- **英文分词**: 10-20ms/段落
- **向量搜索**: 50-200ms/查询

### 资源使用
- **VLM 处理**: 2-4GB GPU 内存
- **嵌入生成**: 100-200MB / 1000 块
- **ElasticSearch 索引**: 约 1.5 倍原文档大小

## 🔧 配置参数

### 核心配置文件
- `docker/service_conf.yaml`: 服务配置包括数据库连接
- `docker/.env`: Docker 环境变量（需与 service_conf.yaml 同步）
- `conf/conf.py`: Python 配置管理

### 服务端口
- RAGFlow API: 9380
- Web 界面: 80 (通过 nginx)
- ElasticSearch: 9200
- MySQL: 3306
- MinIO: 9000
- Redis: 6379

### VLM 模型配置
```yaml
# 配置 VLM 进行图像描述
vlm_config:
  model: "llava"  # 或其他 VLM 模型
  temperature: 0.7
  max_tokens: 500
```

### 英文文档优化
```yaml
# 针对英文文档优化
chunk_size: 512  # 英文较小（中文为 1000）
chunk_overlap: 50  # 英文重叠较少
similarity_threshold: 0.75  # 英文精度更高
```

## 🛣️ 开发路线图

### Phase 1: 后端集成（已完成）
- ✅ VLM PDF 解析器实现
- ✅ 英文 NLP 分词器集成
- ✅ 混合语言检测

### Phase 2: 文档解析器（进行中）
- ⏳ Word 文档解析器 VLM 集成
- ⏳ PowerPoint 解析器增强
- ⏳ Excel 解析器表格理解

### Phase 3: 完全英文迁移（计划中）
- 📋 替换中文字典为英文
- 📋 英文特定分块优化
- 📋 查询理解增强
- 📋 英文重排序模型

### Phase 4: 高级功能（未来）
- 📋 多模态搜索
- 📋 实时协作
- 📋 API 版本管理
- 📋 企业级安全

## 🤝 贡献指南

欢迎贡献代码和提出改进建议！

### 开发流程
1. Fork 项目仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

### 代码规范
- 使用 2 个空格缩进
- 函数和变量使用驼峰命名
- 添加适当的注释说明
- 保持代码整洁和可读性
- 遵循 PEP 8 标准

### 测试要求
```bash
# Python 测试
pytest

# 前端代码检查
cd web
npm run lint

# 前端测试
npm test
```

## 🚨 故障排除

### 常见问题

#### VLM 相关问题
1. **VLM 模型无法加载**:
   - 检查模型路径和权限
   - 验证 CUDA/GPU 可用性
   - 解决方案: 使用 CPU 回退或较小的 VLM 模型

2. **VLM 处理缓慢**:
   - 问题: 大型 PDF 包含多个图像
   - 解决方案: 批处理图像，实现缓存

#### 英文 NLP 问题
1. **NLTK 数据缺失**:
   ```bash
   python -m nltk.downloader punkt
   python -m nltk.downloader wordnet
   python -m nltk.downloader averaged_perceptron_tagger
   ```

2. **混合语言检测失败**:
   - 问题: 包含代码或特殊字符的文档
   - 解决方案: 调整语言检测阈值

#### 系统集成问题
1. **ElasticSearch 连接**:
   - 检查: `curl http://localhost:9200`
   - 修复: `sysctl vm.max_map_count=262144`

2. **Redis 任务队列**:
   - 检查: `redis-cli ping`
   - 修复: 确保 Redis 容器正在运行

## 📄 许可证

本项目采用 Apache 2.0 许可证。详情请参阅 [LICENSE](LICENSE) 文件。

## 📞 联系方式

- **项目维护者**: RAG_Fin 开发团队
- **技术支持**: [创建 Issue](../../issues)
- **功能建议**: [提交 Feature Request](../../issues/new)
- **文档问题**: [文档反馈](../../issues/new?labels=documentation)

## 🙏 致谢

感谢以下技术和项目的支持：
- [RAGFlow](https://github.com/infiniflow/ragflow) 开源项目
- [Unstructured](https://github.com/Unstructured-IO/unstructured) 文档解析库
- [NLTK](https://www.nltk.org/) 自然语言处理工具包
- [Ollama](https://github.com/ollama/ollama) 本地 LLM 部署
- 开源社区的贡献和反馈

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请给我们一个 Star！**

*最后更新：2024年10月*

*该项目是基于 RAGFlow 的第二代 RAG 系统，专注于英文文档处理和 VLM 增强解析。*

</div>