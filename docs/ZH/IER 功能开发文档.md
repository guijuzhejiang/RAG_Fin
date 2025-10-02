# IER 功能开发文档

## 目标概述

在现有的 RAG 系统上新增一个模块，支持：

一. 文件解析阶段
1. 在知识库中仅上传并解析单个 IER 的 PDF 文档。(在画面手动完成，无需修改代码)
2. 使用 ChatLLM（本地ollama部署的LLM）解析 IER PDF 并抽取三个核心字段：`industry`（可能多个）、`Geography`（可能多个）、以及文档`概要`（summary）。
3. 新建数据库表 `IER`，向表中插入：`kb_id`（知识库ID）、`pdf_filename`、`industries`、`geographies`、`summary`、以及元数据字段（如 `id`、`create_time`、`create_date`、`update_time`、`update_date` 等）。

二. 用户问答阶段
1. 当用户提问时，用户先指定三个查询条件：`industry`、`Geography`、`问题概要`；随后以这些条件对 `IER` 表做模糊或相似度匹配，返回检索匹配到的多个 `kb_id`。
2. 前端用户在刚检索到的多个`kb_id`中选择一个，点击“检索”，发送给后端一个指定的`kb_id`， 后端使用该知识库作为RAG的检索来源，执行 RAG 问答并返回给用户。
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
