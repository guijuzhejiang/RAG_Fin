# RAG_Fin - Intelligent Document Retrieval & Analysis System

<div align="center">

![RAG_Fin](https://img.shields.io/badge/RAG_Fin-v2.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.11+-green.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)
![Build](https://img.shields.io/badge/Build-Docker-blue.svg)

Second-generation English-focused RAG system based on RAGFlow with advanced VLM-enhanced document parsing

[English](README.md) | [中文文档](README_zh.md) | [日本語](README_ja.md) | [한국어](README_ko.md)

</div>

## 📖 Project Overview

RAG_Fin is a **second-generation RAG system** optimized for English document processing, built upon the original RAGFlow architecture with significant improvements. This system is specifically designed for financial, investment, and business analysis document processing needs, providing high-precision intelligent retrieval and analysis services.

### 🎯 Key Features

- **🧠 VLM-Enhanced Parsing**: Revolutionary PDF parsing quality improvement using Vision Language Models (VLM)
- **🌍 English-Optimized Processing**: Specialized NLP processing and tokenization optimization for English documents
- **⚡ Dual-Service Architecture**: API service and computation service separation for better scalability
- **📊 Intelligent Analysis**: Support for IER (Industry-Economy-Region) three-dimensional intelligent retrieval
- **🔍 Multimodal Understanding**: Unified understanding and processing of images, tables, and text
- **📈 Real-time Processing**: Asynchronous task processing with real-time progress feedback

## 🚀 Major Improvements Over Original RAGFlow

### 1. Language Processing Upgrade
- **Primary Target**: English documents and queries (previously Chinese)
- **NLP Engine**: Integration of NLTK English processing components
  - PorterStemmer for word stemming
  - WordNetLemmatizer for lemmatization
  - `word_tokenize` for English tokenization
- **Hybrid Tokenizer**: `rag/nlp/rag_tokenizer.py` retains Chinese dictionary but adds English detection and processing

### 2. VLM PDF Parser Innovation
- **File Location**: `deepdoc/parser/pdf_parser_unstructured_VLM.py`
- **Core Advantage**: Replace traditional OCR with advanced Vision Language Model processing
- **Key Features**:
  - Unstructured library integration for high-quality PDF parsing
  - VLM-based image description and understanding
  - Image preprocessing (contrast, sharpness, noise reduction)
  - Smart header/footer filtering
  - Table extraction with HTML structure preservation
  - Perceptual hashing for image deduplication

### 3. Pending Updates
- Word parser (`docx_parser.py`) - Still using original logic
- PowerPoint parser (`ppt_parser.py`) - Not yet updated
- Excel parser (`excel_parser.py`) - Awaiting VLM integration

## 🏗️ System Architecture

### Dual-Service Architecture Design

RAG_Fin employs a **dual-service architecture** for better scalability and resource management:

```
User Request → API Service (:9380) 
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

#### 1. API Service (`api/ragflow_server.py`)
- Flask-based REST API server
- Handles HTTP requests on port 9380
- Manages user authentication and sessions
- Schedules document processing tasks
- Updates processing progress in real-time

#### 2. RAG Computation Service (`rag/svr/task_executor.py`)
- Asynchronous task processing engine
- Consumes tasks from Redis queue
- Performs document parsing, chunking, and vectorization
- Manages three types of models:
  - **Embedding Model**: For text vectorization
  - **Chat Model**: For text generation and summarization
  - **VLM Model** (IMAGE2TEXT): For image understanding and description
- Stores results in ElasticSearch

### Core Component Structure

```
api/                 # Flask REST API layer
├── apps/           # API endpoints
├── db/             # Database models and services
├── schemas/        # Pydantic schemas
└── ragflow_server.py  # Main API server entry

rag/                # RAG core engine
├── nlp/           # NLP processing (English-focused)
├── app/           # Document processing templates
│   └── naive.py   # Main document parser dispatcher
├── llm/           # LLM and embedding models
│   └── cv_model.py # Vision models including OllamaCV
└── svr/           
    └── task_executor.py # Async task processing service

deepdoc/           # Document parsing engine
├── parser/        # Format-specific parsers
│   └── pdf_parser_unstructured_VLM.py  # NEW: VLM-enhanced parser
└── vision/        # Layout recognition and OCR
```

### Service Dependencies

- **ElasticSearch**: Vector storage and full-text search
- **MySQL**: Business data and metadata
- **Redis**: Caching and task queue management (uses Redis Streams for task distribution)
- **MinIO**: Object storage for documents and extracted images
- **Ollama**: Local VLM deployment for image understanding (via OllamaCV class)

## 🛠️ Quick Start

### System Requirements

- **Python**: >= 3.11, < 3.12
- **Node.js**: >= 18.0 (for frontend development)
- **Docker**: >= 24.0.0
- **Docker Compose**: >= 2.26.1
- **Hardware**: CPU >= 4 cores, RAM >= 16GB, Disk >= 50GB

### 🐳 Docker Deployment (Recommended)

#### 1. Clone the Project
```bash
git clone <repository-url>
cd RAG_Fin
```

#### 2. Start Complete System
```bash
cd docker
docker compose up -d
```

#### 3. Access Services
- **Web Interface**: http://localhost
- **API Interface**: http://localhost:9380
- **MinIO Console**: http://localhost:9000

### 🔧 Local Development Environment

#### 1. Start Base Services
```bash
cd docker
docker compose -f docker-compose-base.yml up -d
```

#### 2. Configure Host Mapping
```bash
echo "127.0.0.1 es01 mysql minio redis" >> /etc/hosts
```

#### 3. Install Python Dependencies
```bash
# Install dependencies using Poetry
poetry install --sync --no-root

# Activate virtual environment
source .venv/bin/activate
export PYTHONPATH=$(pwd)
```

#### 4. Start Backend Services
```bash
# Start API service
python api/ragflow_server.py

# Start computation service in new terminal
python rag/svr/task_executor.py
```

#### 5. Start Frontend Service
```bash
cd web
npm install --force
npm run dev
```

## 📋 Usage Guide

### VLM PDF Parser

#### Core Functionality
```python
from deepdoc.parser.pdf_parser_unstructured_VLM import PdfParserVLM

# Initialize parser (optional VLM model)
parser = PdfParserVLM(vlm_mdl=your_vlm_model)

# Parse PDF document
sections, tables = parser("document.pdf", need_image=True)
print(f"Extracted {len(sections)} sections and {len(tables)} tables")
```

#### Processing Pipeline
1. **PDF Loading**: Uses unstructured's `partition_pdf` with `hi_res` strategy
2. **Element Classification**: Text preserved as-is, images described with VLM, tables extracted as HTML
3. **Image Enhancement**: Preprocessing before VLM recognition
4. **Content Extraction**: VLM for images, HTML for tables, direct text extraction
5. **Deduplication Strategy**: Perceptual hashing to avoid duplicate images/tables

### English NLP Processing

#### Tokenizer Configuration
```python
from rag.nlp.rag_tokenizer import RagTokenizer

tokenizer = RagTokenizer()

# Automatic English text detection (>60% ASCII characters)
english_text = "Financial analysis of market trends"
tokens = tokenizer.tokenize(english_text)
print(f"Tokenization result: {tokens}")
```

#### Language Detection Features
- Automatic detection of English vs Chinese content
- English text bypasses Chinese tokenization
- Uses NLTK tools for English-specific processing

### IER Intelligent Retrieval System

The system supports **Industry-Economy-Region** three-dimensional intelligent retrieval:

#### Demo Interface
```bash
# Access demo page
open demo/ier_search_demo.html

# Or use local server
cd demo
python -m http.server 8000
# Visit http://localhost:8000/ier_search_demo.html
```

#### API Usage
```javascript
// POST /api/ier/search
{
  "industry": "technology",
  "geography": "north-america", 
  "parties": "Microsoft, Activision",
  "question": "M&A case analysis",
  "language": "en"
}
```

## 🧪 Testing & Debugging

### Service Health Checks
```bash
# Check API service
curl http://localhost:9380/api/health

# Check ElasticSearch
curl http://localhost:9200/_cluster/health

# Check Redis
redis-cli ping

# Check MinIO
mc admin info minio/

# Check Ollama VLM models
curl http://localhost:11434/api/tags
```

### VLM Parser Testing
```python
# Test the new VLM PDF parser
from deepdoc.parser.pdf_parser_unstructured_VLM import PdfParserVLM

parser = PdfParserVLM(vlm_mdl=your_vlm_model)
sections, tables = parser("test.pdf", need_image=True)
print(f"Extracted {len(sections)} sections and {len(tables)} tables")
```

### English NLP Testing
```python
from rag.nlp.rag_tokenizer import RagTokenizer

tokenizer = RagTokenizer()
english_text = "This is an English document about financial analysis."
tokens = tokenizer.tokenize(english_text)
print(f"Tokenization result: {tokens}")
```

### Task Monitoring
```bash
# Monitor Redis Streams task queue
redis-cli xinfo stream rag_task_stream

# View pending tasks
redis-cli xlen rag_task_stream

# View consumer group info
redis-cli xinfo groups rag_task_stream
```

## 📊 Performance Metrics

### Processing Performance
- **PDF VLM Parsing**: 2-5 seconds per page
- **PDF Traditional Parsing**: 0.5-1 second per page
- **English Tokenization**: 10-20ms per paragraph
- **Vector Search**: 50-200ms per query

### Resource Usage
- **VLM Processing**: 2-4GB GPU memory
- **Embedding Generation**: 100-200MB per 1000 chunks
- **ElasticSearch Index**: ~1.5x original document size

## 🔧 Configuration

### Core Configuration Files
- `docker/service_conf.yaml`: Service configurations including database connections
- `docker/.env`: Docker environment variables (must sync with service_conf.yaml)
- `conf/conf.py`: Python configuration management

### Service Ports
- RAGFlow API: 9380
- Web Interface: 80 (via nginx)
- ElasticSearch: 9200
- MySQL: 3306
- MinIO: 9000
- Redis: 6379

### VLM Model Configuration
```yaml
# Configure VLM for image description
vlm_config:
  model: "llava"  # or other VLM models
  temperature: 0.7
  max_tokens: 500
```

### English Document Optimization
```yaml
# Optimize for English documents
chunk_size: 512  # Smaller for English (vs 1000 for Chinese)
chunk_overlap: 50  # Less overlap needed for English
similarity_threshold: 0.75  # Higher threshold for English precision
```

## 🛣️ Development Roadmap

### Phase 1: Backend Integration (Completed)
- ✅ VLM PDF parser implementation
- ✅ English NLP tokenizer integration
- ✅ Hybrid language detection

### Phase 2: Document Parsers (In Progress)
- ⏳ Word document parser with VLM integration
- ⏳ PowerPoint parser enhancement
- ⏳ Excel parser with table understanding

### Phase 3: Full English Migration (Planned)
- 📋 Replace Chinese dictionaries with English
- 📋 English-specific chunking optimization
- 📋 Query understanding enhancement
- 📋 English-focused re-ranking models

### Phase 4: Advanced Features (Future)
- 📋 Multimodal search
- 📋 Real-time collaboration
- 📋 API versioning
- 📋 Enterprise-grade security

## 🤝 Contributing

Welcome code contributions and improvement suggestions!

### Development Workflow
1. Fork the project repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Create a Pull Request

### Code Standards
- Use 2-space indentation
- Use camelCase for functions and variables
- Add appropriate comments
- Keep code clean and readable
- Follow PEP 8 standards

### Testing Requirements
```bash
# Python tests
pytest

# Frontend linting
cd web
npm run lint

# Frontend tests
npm test
```

## 🚨 Troubleshooting

### Common Issues

#### VLM-Related Issues
1. **VLM Model Not Loading**:
   - Check model path and permissions
   - Verify CUDA/GPU availability
   - Solution: Use CPU fallback or smaller VLM model

2. **Slow VLM Processing**:
   - Issue: Large PDFs with many images
   - Solution: Batch process images, implement caching

#### English NLP Issues
1. **NLTK Data Missing**:
   ```bash
   python -m nltk.downloader punkt
   python -m nltk.downloader wordnet
   python -m nltk.downloader averaged_perceptron_tagger
   ```

2. **Mixed Language Detection Failure**:
   - Issue: Documents with code or special characters
   - Solution: Adjust language detection threshold

#### System Integration Issues
1. **ElasticSearch Connection**:
   - Check: `curl http://localhost:9200`
   - Fix: `sysctl vm.max_map_count=262144`

2. **Redis Task Queue**:
   - Check: `redis-cli ping`
   - Fix: Ensure Redis container is running

## 📄 License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

## 📞 Contact

- **Project Maintainers**: RAG_Fin Development Team
- **Technical Support**: [Create Issue](../../issues)
- **Feature Requests**: [Submit Feature Request](../../issues/new)
- **Documentation Issues**: [Documentation Feedback](../../issues/new?labels=documentation)

## 🙏 Acknowledgments

Thanks to the following technologies and projects for their support:
- [RAGFlow](https://github.com/infiniflow/ragflow) open source project
- [Unstructured](https://github.com/Unstructured-IO/unstructured) document parsing library
- [NLTK](https://www.nltk.org/) natural language processing toolkit
- [Ollama](https://github.com/ollama/ollama) local LLM deployment
- Open source community contributions and feedback

---

<div align="center">

**⭐ If this project helps you, please give us a Star!**

*Last Updated: October 2024*

*This project is a second-generation RAG system based on RAGFlow, focusing on English document processing and VLM-enhanced parsing.*

</div>