# coding=utf-8
# @Time : 2024/12/3 17:16
# @File : system.py
from accelerate.commands.config.default import description
from marshmallow import Schema, fields, validate

from api.schemas.common import CommonResponseSchema
from api.settings import RetCode

"""获取可用的模型 llm/factories"""

# {
#     "data": {
#         "Ollama": [
#             {
#                 "available": true,
#                 "fid": "Ollama",
#                 "llm_name": "gte-Qwen2-7B-instruct.Q5_K_M",
#                 "model_type": "embedding"
#             },
#             {
#                 "available": true,
#                 "fid": "Ollama",
#                 "llm_name": "qwen2.5:14b-instruct-q5_K_M",
#                 "model_type": "chat"
#             }
#         ]
#     },
#     "retcode": 0,
#     "retmsg": "success"
# }

class ModelSchema(Schema):
    available = fields.Bool(description="Indicates if the model is available")
    fid = fields.Str(description="FID of the model")
    llm_name = fields.Str(description="Name of the LLM (Large Language Model)")
    model_type = fields.Str(description="Type of the model (e.g., 'embedding', 'chat')")

class LlmDataSchema(Schema):
    # Using Dict to allow dynamic keys with a list of ModelSchema as values
    dynamic_keys = fields.Dict(
        keys=fields.Str(description="Dynamic keys representing different categories"),
        values=fields.List(fields.Nested(ModelSchema), description="List of models for the category"),
        required=True
    )

# 获取可用的模型
class LlmFactoriesResSchema(CommonResponseSchema):
    data = fields.Nested(LlmDataSchema, description="Main data section with dynamic keys")


"""设置模型对应的api_key llm/set_api_key"""

# 请求
class LlmSetApiKeyReqSchema(Schema):
    llm_factory = fields.Str(required=True, description="模型名")
    api_key = fields.Str(required=True, description="api key")

# 返回
class LlmSetApiKeyResSchema(CommonResponseSchema):
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['success', 'Sorry! Data missing!', "Fail to access model"
                                 ]),
        dump_default="success"
    )



"""设置模型对应的api_key llm/add_llm"""
# 请求
class AddLlmReqSchema(Schema):
    model_type = fields.Str(
        validate=validate.OneOf(
            ["chat","embedding", "rerank", "image2text", "audio2text", "text2audio"]),
        required=True,
        description="model type")
    api_base = fields.Str(required=True, description="模型 base url")
    api_key = fields.Str(allow_none=True, description="api key")
    llm_name = fields.Str(required=True, description="model name")
    llm_factory = fields.Str(required=True, description="model name")

# 返回
class AddLlmResSchema(CommonResponseSchema):
    retcode = fields.Integer(
        dump_default=RetCode.SUCCESS,
        description="返回码"
    )
    retmsg = fields.Str(
        description="返回信息",
        dump_default="success"
    )


"""刪除模型 llm/delete_llm"""
# 请求
class DeleteLlmReqSchema(Schema):
    llm_factory = fields.Str(required=True, description="llm factory e.g., Ollama")
    llm_name = fields.Str(required=True, description="model name e.g., Ollama nomic-embed-text:v1.5")

# 返回
class DeleteLlmResSchema(CommonResponseSchema):
    retcode = fields.Integer(
        validate=validate.OneOf(
            [RetCode.SUCCESS]),
        dump_default=RetCode.SUCCESS,
        description="返回码"
    )
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['success']),
        dump_default="success")


"""刪除模型工厂 llm/delete_factory"""

# 请求
class DeleteFactoryReqSchema(Schema):
    llm_factory = fields.Str(required=True, description="llm factory e.g., Ollama")

# 返回
class DeleteFactoryResSchema(Schema):
    data = fields.Bool(allow_none=True, description="bool result")
    retcode = fields.Integer(
        dump_default=RetCode.SUCCESS,
        description="返回码"
    )
    retmsg = fields.Str(
        description="返回信息",
        dump_default="success")


"""获取当前用户设置了的模型 llm/my_llms"""


class LlmSchema(Schema):
    name = fields.Str(description="Name of the model")
    type = fields.Str(description="Type of the model (e.g., 'embedding', 'chat')")
    used_token = fields.Int(description="Number of tokens used")

class CategorySchema(Schema):
    llm = fields.List(fields.Nested(LlmSchema), description="List of models under the category")
    tags = fields.Str(description="Tags associated with the category")

class MyLlmsSchema(Schema):
    dynamic_keys = fields.Dict(
        keys=fields.Str(description="Dynamic keys representing categories"),
        values=fields.Nested(CategorySchema),
        description="Dynamic categories with LLM details and tags"
    )

# 返回
class MyLlmsResSchema(Schema):
    data = fields.Nested(MyLlmsSchema, allow_none=True, description="返回模型列表")
    retcode = fields.Integer(
        dump_default=RetCode.SUCCESS,
        description="返回码"
    )
    retmsg = fields.Str(
        description="返回信息",
        dump_default="success")


"""获取可用的模型 llm/list"""


# 获取可用的模型
class LlmListResSchema(CommonResponseSchema):
    data = fields.Nested(LlmDataSchema, description="Main data section with dynamic keys")