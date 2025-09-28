# coding=utf-8
# @Time : 2024/12/6 16:20
# @File : api.py

from marshmallow import Schema, fields, validate

from api.schemas.common import CommonResponseSchema
from api.schemas.conversation import ConversationMessageSchema
from api.schemas.db_models import ConversationSchema

"""new_token api/new_token"""

# 请求
class ApiNewTokenReqSchema(Schema):
    dialog_id = fields.Str(description="助理id")
    canvas_id = fields.Str(description="canvas_id")


# 返回
class ApiNewTokenResSchema(CommonResponseSchema):
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['success',
                                 "Tenant not found!",
                                 "No chunk found! Check the chunk status please!"
                                 ]),
        dump_default="success"
    )



"""token_list api/token_list"""

# 请求
class ApiTokenListReqSchema(Schema):
    dialog_id = fields.Str(description="助理id")
    canvas_id = fields.Str(description="canvas_id")


# 返回
class ApiTokenListResSchema(CommonResponseSchema):
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['success',
                                 "Tenant not found!",
                                 "No chunk found! Check the chunk status please!"
                                 ]),
        dump_default="success"
    )


"""rm api/rm"""

# 请求
class ApiRmReqSchema(Schema):
    tenant_id = fields.Str(description="tenant id")
    dialog_id = fields.Str(description="dialog id")
    tokens = fields.List(fields.Str(description="token"), description="tokens")


"""stats api/stats"""
class ApiStatsSchema(Schema):
    pv = fields.List(fields.List(fields.Str()))
    uv = fields.List(fields.List(fields.Str()))
    speed = fields.List(fields.List(fields.Str()))
    tokens = fields.List(fields.List(fields.Str()))
    round = fields.List(fields.List(fields.Str()))
    thumb_up = fields.List(fields.List(fields.Str()))


# 请求
class ApiStatsReqSchema(Schema):
    from_date = fields.Str(description="from_date %Y-%m-%d 00:00:00")
    to_date = fields.Str(description="to_date %Y-%m-%d 00:00:00")
    canvas_id = fields.Str(description="canvas_id")


# 返回
class ApiStatsResSchema(CommonResponseSchema):
    data = fields.Nested(ApiStatsSchema)
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['success',
                                 "Tenant not found!",
                                 "No chunk found! Check the chunk status please!"
                                 ]),
        dump_default="success"
    )



"""新对话 api/new_conversation"""

# 请求
class ApiNewConversationReqSchema(Schema):
    user_id = fields.Str(description="user_id")


# 返回
class ApiNewConversationResSchema(CommonResponseSchema):
    data = fields.Nested(ConversationSchema)
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['success',
                                 "Token is not valid!\"",
                                 "canvas not found.",
                                 "Dialog not found",
                                 "No chunk found! Check the chunk status please!"
                                 ]),
        dump_default="success"
    )


"""对话 api/conversation/<conversation_id>"""

# 返回
class ApiCompletionIdResSchema(CommonResponseSchema):
    data = fields.Nested(ConversationSchema)
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['success',
                                 "Token is not valid!\"",
                                 "Conversation not found!",
                                 "Token is not valid for this conversation_id!\"",
                                 "No chunk found! Check the chunk status please!"
                                 ]),
        dump_default="success"
    )


"""新对话 api/completion"""

# 请求
class ApiCompletionReqSchema(Schema):
    conversation_id = fields.Str(description="conversation_id")
    messages = fields.List(fields.Nested(ConversationMessageSchema), allow_none=True, description="聊天记录")
    quote = fields.Bool(description="显示引用")
    stream = fields.Bool(description="流式传输")


# 返回
class ApiCompletionResSchema(CommonResponseSchema):
    data = fields.Nested(ConversationSchema)
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['success',
                                 "Token is not valid!\"",
                                 "Conversation not found!",
                                 "canvas not found.",
                                 "No chunk found! Check the chunk status please!"
                                 ]),
        dump_default="success"
    )


"""上传 api/document/upload"""

# 请求
class ApiDocumentUploadReqSchema(Schema):
    kb_name = fields.Bool(description="显示引用")
    stream = fields.Bool(description="流式传输")


# 返回
class ApiDocumentUploadResSchema(CommonResponseSchema):
    data = fields.Nested(ConversationSchema)
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['success',
                                 "Token is not valid!\"",
                                 "Conversation not found!",
                                 "canvas not found.",
                                 "No chunk found! Check the chunk status please!"
                                 ]),
        dump_default="success"
    )


