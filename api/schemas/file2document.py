# coding=utf-8
# @Time : 2024/12/5 11:43
# @File : file2document.py
from marshmallow import Schema, fields, validate

from api.schemas.common import CommonResponseSchema

"""链接知识库 file2document/convert"""
class File2documentConvertSchema(Schema):
    create_date = fields.Str(required=True, description="Creation date in GMT format")
    create_time = fields.Int(required=True, description="Creation time in epoch milliseconds")
    document_id = fields.Str(required=True, description="Unique ID of the document")
    file_id = fields.Str(required=True, description="Unique ID of the associated file")
    id = fields.Str(required=True, description="Unique identifier of the entry")
    update_date = fields.Str(required=True, description="Last update date in GMT format")
    update_time = fields.Int(required=True, description="Last update time in epoch milliseconds")


# 请求
class File2documentConvertReqSchema(Schema):
    file_ids = fields.List(fields.Str(),
                           description="file_ids",
                           required=True
                           )
    kb_ids = fields.List(fields.Str(),
                         description="kb_ids",
                         required=True
                         )


# 返回
class File2documentConvertResSchema(CommonResponseSchema):
    data = fields.List(fields.Nested(File2documentConvertSchema), description="kb data")
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['success', "Document not found!",
                                 "Tenant not found!",
                                 "Database error (Document removal)!",
                                 "Can't find this knowledgebase!",
                                 "Can't find this file!",
                                 "No chunk found, please upload file and parse it."
                                 ]),
        dump_default="success"
    )


"""删除 file2document/rm"""

# 请求
class File2documentRmReqSchema(Schema):
    file_ids = fields.List(fields.Str(),
                           description="file_ids",
                           required=True
                           )


# 返回
class File2documentRmResSchema(CommonResponseSchema):
    data = fields.List(fields.Nested(File2documentConvertSchema), description="kb data")
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['success', "Document not found!",
                                 "Tenant not found!",
                                 "Inform not found!",
                                 "Database error (Document removal)!",
                                 'Lack of "Files ID"',
                                 "No chunk found, please upload file and parse it."
                                 ]),
        dump_default="success"
    )
