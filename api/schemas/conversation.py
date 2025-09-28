# coding=utf-8
# @Time : 2024/12/3 17:16
# @File : system.py
from marshmallow import Schema, fields, validate, ValidationError

from api.schemas.common import CommonResponseSchema
from api.schemas.db_models import DialogSchema, ConversationSchema

class ConversationChunkPositions(fields.Field):
    def _deserialize(self, value, attr, data, **kwargs):
        if isinstance(value, list):
            if value == [""] or all(isinstance(i, int) for i in value):
                return value
        raise ValidationError("Invalid list format. Expected [int, int, ...] or [''].")

class ConversationMessageReference(fields.Field):
    def _deserialize(self, value, attr, data, **kwargs):
        if isinstance(value, list):
            return value  # 空数组
        elif isinstance(value, dict):
            # if isinstance(value["chunks"], list) and isinstance(value["doc_aggs"], list):
            return value  # 符合预期的字典
        raise ValidationError("Invalid field format. Expected [] or {chunks: [], doc_aggs: []}.")

class ConversationChunkSchema(Schema):
    chunk_id = fields.String()
    content_ltks = fields.String()
    content_with_weight = fields.String()
    doc_id = fields.String()
    docnm_kwd = fields.String()
    kb_id = fields.String()
    important_kwd = fields.List(fields.String(), allow_none=True)
    img_id = fields.String()
    similarity = fields.Float()
    vector_similarity = fields.Float()
    term_similarity = fields.Float()
    positions = ConversationChunkPositions()

class ConversationDocAggsSchema(Schema):
    doc_name = fields.String()
    doc_id = fields.String()
    count = fields.Integer()

class ConversationReferenceSchema(Schema):
    total = fields.Integer()
    chunks = fields.List(fields.Nested(ConversationChunkSchema))
    doc_aggs = fields.List(fields.Nested(ConversationDocAggsSchema))

class ConversationMessageSchema(Schema):
    role = fields.Str(description="角色 assistant还是user")
    content = fields.Str(description="内容")
    doc_ids = fields.List(fields.Str(), allow_none=True, description="文档id")
    id = fields.Str(allow_none=True, description="id")
    prompt = fields.Str(allow_none=True, description="prompt")
    # reference = fields.Nested(ConversationReferenceSchema, allow_none=True, description="引用")
    reference = ConversationMessageReference(allow_none=True, description="引用")
    audio_binary = fields.Raw(allow_none=True, description="音频")


"""对话设置 conversation/set"""

# 请求
class ConversationSetReqSchema(Schema):
    conversation_id = fields.Str(required=True, description="对话 id")
    dialog_id = fields.Str(required=True, description="助理 id")
    is_new = fields.Bool(required=True, description="是不是新的")
    message = fields.List(fields.Nested(ConversationMessageSchema), allow_none=True, description="聊天记录")
    name = fields.Str(allow_none=True, description="名称")
    id = fields.String(description="对话id")
    create_date = fields.String(description="创建时间")
    create_time = fields.Integer(description="创建时间戳")
    reference = fields.List(fields.Nested(ConversationReferenceSchema))
    update_date = fields.String(description="更新时间")
    update_time = fields.Integer(description="更新时间戳")

# 返回
class ConversationSetResSchema(CommonResponseSchema):
    data = fields.Nested(ConversationSchema, allow_none=True)
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['success',
                                 "Conversation not found!",
                                 "Dialog not found",
                                 "Fail to new a conversation!",
                                 "No chunk found, please upload file and parse it."
                                 ]),
        dump_default="success"
    )


"""获取conversion conversion/get"""

# 请求
class ConversationGetReqSchema(Schema):
    conversation_id = fields.Str(required=True, description="对话id")


# 返回
class ConversationGetResSchema(CommonResponseSchema):
    data = fields.Nested(ConversationSchema, allow_none=True)
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['success',
                                 "Conversation not found!",
                                 'Only owner of conversation authorized for this operation.',
                                 "No chunk found, please upload file and parse it."
                                 ]),
        dump_default="success"
    )


"""list conversation conversation/list"""

# 请求
class ConversationListReqSchema(Schema):
    dialog_id = fields.Str(description="dialog id")

# 返回
class ConversationListResSchema(CommonResponseSchema):
    data = fields.List(fields.Nested(ConversationSchema), description="dialog list")
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['success',
                                 'Only owner of conversation authorized for this operation.',
                                 "No chunk found, please upload file and parse it."
                                 ]),
        dump_default="success"
    )


"""删除conversation conversation/rm"""

# 请求
class ConversationRmReqSchema(Schema):
    conversation_ids = fields.List(fields.Str(description="对话id"), allow_none=True)
    dialog_id = fields.Str(description="助理id", allow_none=True)


# 返回
class ConversationRmResSchema(CommonResponseSchema):
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['success',
                                 "Conversation not found!",
                                 'Only owner of conversation authorized for this operation.',
                                 "No chunk found, please upload file and parse it."
                                 ]),
        dump_default="success"
    )



"""聊天对话 conversation/completion"""
class ConversationAnswerSchema(Schema):
    answer = fields.String()
    reference = fields.Dict()  # Empty dictionary or other key-value pairs
    audio_binary = fields.Raw()  # Can handle binary or None
    id = fields.String()


# 请求
class ConversationCompletionReqSchema(Schema):
    conversation_id = fields.Str(description="对话id", allow_none=True)
    messages = fields.List(fields.Nested(ConversationMessageSchema), allow_none=True, description="聊天记录")


# 返回
class ConversationCompletionResSchema(CommonResponseSchema):
    data = fields.Nested(ConversationAnswerSchema, allow_none=True)
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['success',
                                 "Conversation not found!",
                                 "Dialog not found!",
                                 "",
                                 'Only owner of conversation authorized for this operation.',
                                 "No chunk found, please upload file and parse it."
                                 ]),
        dump_default="success"
    )



"""删除某条聊天信息 conversation/delete_msg"""

# 请求
class ConversationDeleteMsgReqSchema(Schema):
    conversation_id = fields.Str(description="对话id", required=True)
    message_id = fields.Str(description="信息id", required=True)


# 返回
class ConversationDeleteMsgResSchema(CommonResponseSchema):
    data = fields.Nested(ConversationSchema, allow_none=True)
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['success',
                                 "Conversation not found!",
                                 "No chunk found, please upload file and parse it."
                                 ]),
        dump_default="success"
    )

