# coding=utf-8
# @Time : 2024/12/2 16:20
# @File : login.py
from accelerate.commands.config.default import description
from marshmallow import Schema, fields, validate

from api.schemas.common import CommonResponseSchema
from api.schemas.db_models import UserSchema, TenantSchema
from api.settings import RetCode

"""浏览器登录 user/login"""


# 浏览器登录请求
class LoginReqSchema(Schema):
    email = fields.Str(required=True, description="邮箱")
    password = fields.Str(required=True, description="密码")


# 浏览器登录返回
class LoginResSchema(Schema):
    retcode = fields.Integer(
        validate=validate.OneOf([RetCode.SUCCESS, RetCode.AUTHENTICATION_ERROR, RetCode.EXCEPTION_ERROR]),
        description="返回码"
    )
    data = fields.Nested(UserSchema, description="User data object")
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['Unauthorized!', 'Email: {email} is not registered!', 'Fail to crypt password', 'Email and password do not match!', 'success']),
        dump_default="success"
    )


"""当前用户设置 user/setting"""


# 设置请求
class SettingReqSchema(Schema):
    avatar = fields.Str(description="头像")
    color_schema = fields.Str(
        validate=validate.OneOf(["Bright", "Dark"]),
        dump_default="Bright",
        description="主题"
    )
    email = fields.Str(description="邮箱")
    language = fields.Str(
        validate=validate.OneOf(["Japanese", "English", "Chinese", "Traditional Chinese"]),
        dump_default="Japanese",
        required=True,
        description="语言"
    )
    nickname = fields.Str(description="昵称")
    timezone = fields.Str(description="时区", dump_default="UTC+8\tAsia/Tokyo")

    password = fields.Str(description="原密码", allow_none=True)
    new_password = fields.Str(description="新密码", allow_none=True)

# 设置返回
class SettingResSchema(Schema):
    # {"data":true,"retcode":0,"retmsg":"success"}
    data = fields.Bool(
        # validate=validate.OneOf([True, False]),
        description="头像")
    retcode = fields.Integer(
        validate=validate.OneOf([RetCode.AUTHENTICATION_ERROR, RetCode.EXCEPTION_ERROR, RetCode.SUCCESS]),
        description="返回码"
    )
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['Update failure!', 'Password error!', 'success']),
        dump_default="success"
    )


"""获取当前用户设置 user/info"""

# 获取当前用户信息返回
class InfoResSchema(Schema):
    data = fields.Nested(UserSchema, required=True, description="User data object")


"""用户注册 user/register"""

# 用户注册请求
class RegisterReqSchema(Schema):
    email = fields.Str(required=True, description="邮箱")
    password = fields.Str(required=True, description="密码")
    nickname = fields.Str(required=True, description="昵称")

# 用户注册返回
class RegisterResSchema(Schema):
    data = fields.Nested(UserSchema, description="User data object")
    retcode = fields.Integer(
        validate=validate.OneOf([RetCode.ARGUMENT_ERROR, RetCode.OPERATING_ERROR, RetCode.EXCEPTION_ERROR, RetCode.SUCCESS]),
        description="返回码"
    )
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(["required argument are missing: {}; ", "required argument values: {}", 'Invalid email address: {email_address}!', 'Email: {email_address} has already registered!', "{nickname}, welcome aboard!", 'User registration failure, error: {str(e)}', 'success']),
        dump_default="success"
    )


"""获取当前用户模型设置 user/tenant_info"""


# 获取当前用户模型设置返回
class TenantInfoResSchema(CommonResponseSchema):
    data = fields.Nested(TenantSchema, description="Tenant data object")
    retcode = fields.Integer(
        validate=validate.OneOf(
            [RetCode.SUCCESS, RetCode.UNAUTHORIZED, RetCode.EXCEPTION_ERROR]),
        description="返回码"
    )


"""设置当前用户模型 user/set_tenant_info"""


# 设置当前用户模型请求
class SetTenantInfoReqSchema(Schema):
    asr_id = fields.Str(description="ASR model identifier, e.g., 'paraformer-realtime-8k-v1'")
    embd_id = fields.Str(description="Embedding model identifier, e.g., 'nomic-embed-text:v1.5@Ollama'")
    img2txt_id = fields.Str(description="Image-to-text model identifier, e.g., 'blaifa/InternVL3_5:8b@Ollama'")
    llm_id = fields.Str(description="Large Language Model identifier, e.g., 'qwen3-14B-think-Q4_K_M@Ollama'")
    name = fields.Str(description="Name of the entity, e.g., 'ray‘s Kingdom'")
    rerank_id = fields.Str(description="Re-rank model identifier, e.g., 'bge-reranker-v2-m3'")
    tenant_id = fields.Str(description="Unique tenant identifier, e.g., '390df030641b11ef8a0a00d861bc2f9a'")
    tts_id = fields.Str(allow_none=True, description="Text-to-speech model identifier, nullable")

# 设置当前用户模型返回
class SetTenantInfoResSchema(CommonResponseSchema):
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['success',
                                 "required argument are missing: {}; ", "required argument values: {}",
                                 ]),
        dump_default="success"
    )