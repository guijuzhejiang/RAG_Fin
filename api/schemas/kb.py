# coding=utf-8
# @Time : 2024/12/3 17:16
# @File : system.py
from marshmallow import Schema, fields, validate
from sympy import false

from api.schemas.common import CommonResponseSchema
from api.schemas.db_models import KbSchema

"""创建知识库 kb/create"""
# 请求
class KbCreateReqSchema(Schema):
    name = fields.Str(
        description="知识库名称",
        required=True
    )



# 返回
class KbCreateResSchema(CommonResponseSchema):
    data = fields.Nested(KbSchema, description="kb data")


"""更新知识库 kb/update"""

class KbRaptorSchema(Schema):
    use_raptor = fields.Bool(dump_default=false, description="使用召回增强RAPTOR策略, 请参考 https://huggingface.co/papers/2401.18059")

class KbParserConfigSchema(Schema):
    chunk_token_num = fields.Integer(dump_default=128, description="每个chunk的token数量")
    delimiter = fields.Str(dump_default="\\n!?;.。；！？", description="分割符")
    html4excel = fields.Bool(dump_default=False, description="Excel 是否将被解析为 HTML 表。如果为 FALSE，Excel 中的每一行都将形成一个块。")
    layout_recognize = fields.Bool(dump_default=True, description="使用视觉模型进行布局分析，以更好地识别文档结构，找到标题、文本块、图像和表格的位置。 如果没有此功能，则只能获取 PDF 的纯文本。")
    raptor = fields.Nested(KbRaptorSchema, description="使用召回增强RAPTOR策略")

# 请求
class KbUpdateReqSchema(Schema):
    avatar = fields.Str(allow_none=True, description="头像")
    description = fields.Str(allow_none=True, description="描述")
    embd_id = fields.Str(description="Embedding ID, e.g.,  nomic-embed-text:v1.5@Ollama")
    kb_id = fields.Str(description="知识库 ID")
    language = fields.Str(
        validate=validate.OneOf(["Japanese", "English", "Chinese"]),
        dump_default="Japanese",
        required=True,
        description="语言"
    )
    name = fields.Str(description="知识库名称")
    parser_id = fields.Str(
        description="解析方法",
        dump_default="naive",
        # validate=validate.OneOf(['naive', "table"]),
    )
    permission = fields.Str(
        description="权限",
        dump_default="me",
        validate=validate.OneOf(['me', "team"]),
    )
    parser_config = fields.Nested(KbParserConfigSchema, description="解析配置")



# 返回
class KbUpdateResSchema(CommonResponseSchema):
    data = fields.Nested(KbSchema, description="kb data")


"""获取知识库信息 kb/detail"""
class KbDetailSchema(KbUpdateReqSchema):
    chunk_num = fields.Integer(dump_default=0, description="chunk数量")
    doc_num = fields.Integer(dump_default=0, description="文档数量")
    token_num = fields.Integer(dump_default=0, description="token数量")

# 请求
class KbDetailReqSchema(Schema):
    kb_id = fields.Str(required=True, description="知识库 id")

# 返回
class KbDetailResSchema(CommonResponseSchema):
    data = fields.Nested(KbDetailSchema, description="kb detail data")


"""获取知识库列表 kb/list"""


class KbListKbsItemSchema(KbDetailSchema):
    create_date = fields.DateTime(description="创建时间")
    create_time = fields.Integer(description="创建时间戳")
    created_by = fields.Str(description="创建用户id")
    similarity_threshold = fields.Float(description="相似度阈值")
    vector_similarity_weight = fields.Float(description="关键字相似度权重")
    status = fields.Str(description="开启状态")
    tenant_id = fields.Str(description="tenant_id")
    update_date = fields.Str(description="更新时间")
    update_time = fields.Int(description="更新时间戳")

# 请求
class KbListKbsReqSchema(Schema):
    page = fields.Integer(allow_none=True, dump_default=1, description="页码")
    page_size = fields.Integer(allow_none=True, dump_default=150, description="页大小")
    orderby = fields.Str(allow_none=True, dump_default="create_time", description="排序列")
    desc = fields.Bool(allow_none=True, dump_default=True, description="降序")

# 返回
class KbListKbsResSchema(CommonResponseSchema):
    data = fields.List(fields.Nested(KbListKbsItemSchema), required=True, description="List of kb")


"""删除知识库 kb/rm"""

# 请求
class KbRmReqSchema(Schema):
    kb_id = fields.Str(required=True, description="知识库 id")

# 返回
class KbRmResSchema(CommonResponseSchema):
    data = fields.Bool(description="bool result of kb remove")
