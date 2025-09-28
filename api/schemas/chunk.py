# coding=utf-8
# @Time : 2024/12/6 16:05
# @File : chunk.py
from marshmallow import Schema, fields, validate
from api.schemas.common import CommonResponseSchema


class ChunkSchema(Schema):
    chunk_id = fields.String()
    content_ltks = fields.String()
    content_with_weight = fields.String()
    doc_id = fields.String()
    docnm_kwd = fields.String()
    img_id = fields.String()
    important_kwd = fields.List(fields.String(), )
    kb_id = fields.String()
    positions = fields.List(fields.String(), )
    similarity = fields.Float()
    term_similarity = fields.Float()
    vector_similarity = fields.Float()

class DocAggsSchema(Schema):
    count = fields.Integer()
    doc_id = fields.String()
    doc_name = fields.String()

class ChunkDataSchema(Schema):
    chunks = fields.List(fields.Nested(ChunkSchema))
    doc_aggs = fields.List(fields.Nested(DocAggsSchema))
    total = fields.Integer()

"""检索 chunk/retrieval_test"""

# 请求
class ChunkRetrievalTestReqSchema(Schema):
    kb_id = fields.List(fields.Str(description="知识库id"), allow_none=True)
    question = fields.Str(description="问题", required=True)
    doc_ids = fields.List(fields.Str(description="文档id"), allow_none=True)
    page = fields.Int(dump_default=1, description="页码")
    size = fields.Int(dump_default=30, description="分页size")
    rerank_id = fields.Str(description="rerank id")
    search_depth = fields.Float(dump_default=0.2, description="search_depth")
    similarity_threshold = fields.Float(dump_default=0.3, description="相似度阈值")
    vector_similarity_weight = fields.Float(dump_default=0.4, description="关键字相似度权重")
    top_k = fields.Int(dump_default=1024, description="top_k")
    highlight = fields.Bool(allow_none=True, description="highlight")


# 返回
class ChunkRetrievalTestResSchema(CommonResponseSchema):
    data = fields.Nested(ChunkDataSchema, allow_none=True)
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['success',
                                 "Knowledgebase not found!",
                                 "Only owner of knowledgebase authorized for this operation.",
                                 "No chunk found! Check the chunk status please!"
                                 ]),
        dump_default="success"
    )