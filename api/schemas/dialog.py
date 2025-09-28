# coding=utf-8
# @Time : 2024/12/3 17:16
# @File : system.py
from marshmallow import Schema, fields, validate

from api.schemas.common import CommonResponseSchema
from api.schemas.db_models import DialogSchema


"""新建或修改聊天助理 dialog/set"""

class DialogSetPromptConfigParametersSchema(Schema):
    key = fields.Str(dump_default="knowledge", description="key")
    optional = fields.Bool(dump_default=True, description="是否启用")


class DialogSetPromptConfigSchema(Schema):
    empty_response = fields.Str(dump_default="Answer not found in the knowledge base.", description="空回复")
    prologue = fields.Str(
        dump_default="Hi! I'm your assistant, what can I do for you?",
        description="开场白")
    quote = fields.Bool(dump_default=True, description="是否应该显示原文出处？")
    refine_multiturn = fields.Bool(dump_default=True,
                                   description="多轮对话优化 在多轮对话的中，对去知识库查询的问题进行优化。会调用大模型额外消耗token。")
    tts = fields.Bool(dump_default=False, description="tts")
    system = fields.Str(
        dump_default="あなたは日本語で応答を出力するAIアシスタントです。\n      必ず日本語で回答してください。\n      質問に答えるために、知識ベースの内容を要約し、詳しく回答してください。\n      質問に関連する知識ベースのデータを列挙して回答してください。\n      回答にはチャット履歴を考慮し、知識ベースの内容に完全に基づいて答えてください。\n      内容を創作することは避けてください。\n       以下は知識ベースです：\n       {knowledge} \n      以上が知識ベースです。",
        description="llm的system")
    parameters = fields.List(fields.Nested(DialogSetPromptConfigParametersSchema), description="参数")


class DialogSetLlmSettingSchema(Schema):
    frequency_penalty = fields.Float(dump_default=0.7)
    max_tokens = fields.Int(dump_default=8192)
    presence_penalty = fields.Float(dump_default=0.4)
    temperature = fields.Float(dump_default=0.1)
    top_p = fields.Float(dump_default=0.3)


# 请求
class DialogSetReqSchema(Schema):
    dialog_id = fields.Str(allow_none=True, description="助理id,存在时修改助理,不存在时新建助理")
    llm_id = fields.Str(dump_default="大语言聊天模型", allow_none=True,
                        description="llm id,存在时修改助理,不存在时新建助理")
    rerank_id = fields.Str(dump_default="bge-reranker-v2-m3",
                           description="rerank 模型id")
    language = fields.Str(dump_default="English", description="助理名称")
    name = fields.Str(description="助理名称")
    description = fields.Str(description="助理描述")
    icon = fields.Str(description="助理头像")
    top_n = fields.Int(
        dump_default=2,
        description="Top N 并非所有相似度得分高于“相似度阈值”的块都会被提供给大语言模型。 LLM 只能看到这些“Top N”块。"
    )
    top_k = fields.Int(
        dump_default=1024,
        description="Top K 块将被送入Rerank型号。"
    )
    similarity_threshold = fields.Float(
        dump_default=0.3,
        description="相似度阈值 我们使用混合相似度得分来评估两行文本之间的距离。 它是加权关键词相似度和向量余弦相似度。 如果查询和块之间的相似度小于此阈值，则该块将被过滤掉。"
    )
    search_depth = fields.Float(
        dump_default=0.2,
        description="search_depth dialog/set没有用"
    )
    vector_similarity_weight = fields.Float(
        dump_default=0.6,
        description="关键字相似度权重 我们使用混合相似性评分来评估两行文本之间的距离。它是加权关键字相似性和矢量余弦相似性或rerank得分（0〜1）。两个权重的总和为1.0。"
    )
    llm_setting = fields.Nested(DialogSetLlmSettingSchema, description="llm setting")
    kb_ids = fields.List(fields.Str(), description="知识库ids")
    prompt_config = fields.Nested(DialogSetPromptConfigSchema, description="prompt setting")


# 返回
class DialogSetResSchema(CommonResponseSchema):
    data = fields.Nested(DialogSchema,
                         description="dialog 信息"
                         )


"""获取dialog(助理) dialog/get"""

# 请求
class DialogGetReqSchema(Schema):
    dialog_id = fields.Str(allow_none=True, description="助理id")


# 返回
class DialogGetResSchema(CommonResponseSchema):
    data = fields.Nested(DialogSchema,
                         description="dialog 信息"
                         )


"""list dialog(助理),返回对话列表 dialog/list"""

# 返回
class DialogListResSchema(CommonResponseSchema):
    data = fields.List(fields.Nested(DialogSchema), description="dialog list")


"""删除dialog(助理) dialog/rm"""

# 请求
class DialogRmReqSchema(Schema):
    dialog_ids = fields.List(fields.Str(description="助理id"),
                               description="dialog 信息",
                               allow_none=True
                               )


# 返回
class DialogRmResSchema(CommonResponseSchema):
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['success',
                                 'Only owner of dialog authorized for this operation.',
                                 "No chunk found, please upload file and parse it."
                                 ]),
        dump_default="success"
    )
