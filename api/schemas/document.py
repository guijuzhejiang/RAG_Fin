# coding=utf-8
# @Time : 2024/12/5 11:43
# @File : file2document.py
from flask_smorest.fields import Upload
from marshmallow import Schema, fields, validate

from api.schemas.common import CommonResponseSchema, CommonPaginationRequestSchema
from api.schemas.db_models import FileSchema

"""知识库内上传文件 document/upload"""


# 请求
class DocumentUploadFormReqSchema(Schema):
    kb_id = fields.Str(description="知识库 id", required=True)


class DocumentUploadFilesReqSchema(Schema):
    file = fields.List(Upload(), description="Binary file data", load_only=True)


# 返回
class DocumentUploadResSchema(CommonResponseSchema):
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['success',
                                 'Lack of "KB ID"',
                                 'No file part!',
                                 'No file selected!',
                                 "No chunk found, please upload file and parse it."
                                 ]),
        dump_default="success"
    )


"""根据url爬取web转成pdf保存到某知识库？？？ document/web_crawl"""


# 请求
class DocumentWebCrawlFormReqSchema(Schema):
    kb_id = fields.Str(description="知识库 id", required=True)
    name = fields.Str(description="保存文件名", required=True)
    url = fields.Str(description="url", required=True)


# 返回
class DocumentWebCrawlResSchema(CommonResponseSchema):
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['success',
                                 'Lack of "KB ID"',
                                 'The URL format is invalid',
                                 "No chunk found, please upload file and parse it."
                                 ]),
        dump_default="success"
    )


"""知识库内创建空文件 document/create"""


# 请求
class DocumentCreateReqSchema(Schema):
    kb_id = fields.Str(description="知识库 id", required=True)
    name = fields.Str(description="保存文件名", required=True)


# 返回
class DocumentCreateResSchema(CommonResponseSchema):
    data = fields.Nested(FileSchema, allow_none=True, description="文件信息")
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['success',
                                 'Lack of "KB ID"',
                                 "Can't find this knowledgebase!",
                                 "Duplicated document name in the same knowledgebase.",
                                 "No chunk found, please upload file and parse it."
                                 ]),
        dump_default="success"
    )


"""知识库获取文件列表 document/list"""


class DocumentListResItemSchema(Schema):
    docs = fields.List(fields.Nested(FileSchema), description="文件信息list")


# 请求
class DocumentListReqSchema(CommonPaginationRequestSchema):
    kb_id = fields.Str(
        description="知识库id",
        allow_none=True
    )


# 返回
class DocumentListResSchema(CommonResponseSchema):
    data = fields.Nested(DocumentListResItemSchema, description="文件信息")
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['success',
                                 'Lack of "KB ID"',
                                 "Only owner of knowledgebase authorized for this operation.",
                                 "No chunk found, please upload file and parse it."
                                 ]),
        dump_default="success"
    )


"""获取文件信息 document/infos"""


# 请求
class DocumentDocinfosReqSchema(Schema):
    doc_ids = fields.List(fields.Str(),
                          description="文檔id list",
                          allow_none=True
                          )


# 返回
class DocumentDocinfosResSchema(CommonResponseSchema):
    data = fields.List(fields.Nested(FileSchema), description="文件信息")


"""缩略图 document/thumbnails"""


# 请求
class DocumentThumbnailsReqSchema(Schema):
    doc_ids = fields.List(fields.Str(),
                          description="文檔id list",
                          allow_none=True
                          )


# 返回
class DocumentThumbnailsResSchema(CommonResponseSchema):
    data = fields.Dict(keys=fields.Str(description="文件id"),
                       values=fields.Nested(FileSchema, description="文件信息"),
                       description="文件信息")


"""change_status document/change_status"""


# 请求
class DocumentChangeStatusReqSchema(Schema):
    doc_id = fields.List(fields.Str(),
                         description="文檔id",
                         allow_none=True
                         )
    status = fields.Int(
        description="状态 0 或 1",
        allow_none=True
    )


# 返回
class DocumentChangeStatusResSchema(CommonResponseSchema):
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['success',
                                 '"Status" must be either 0 or 1!',
                                 "Document not found!",
                                 "Can't find this knowledgebase!",
                                 "Database error (Document update)!",
                                 "No chunk found, please upload file and parse it."
                                 ]),
        dump_default="success"
    )


"""刪除文件 document/rm"""


# 请求
class DocumentRmReqSchema(Schema):
    doc_id = fields.List(fields.Str(),
                         description="文檔id",
                         allow_none=True
                         )


# 返回
class DocumentRmResSchema(CommonResponseSchema):
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['success',
                                 "Document not found!",
                                 "Tenant not found!",
                                 "Database error (Document update)!",
                                 "No chunk found, please upload file and parse it."
                                 ]),
        dump_default="success"
    )



"""解析文檔 document/run"""


# 请求
class DocumentRunReqSchema(Schema):
    doc_ids = fields.List(fields.Str(),
                         description="文檔ids",
                         allow_none=True
                         )
    run = fields.Int(description="run", required=True)

# 返回
class DocumentRunResSchema(CommonResponseSchema):
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['success',
                                 "Tenant not found!",
                                 "No chunk found, please upload file and parse it."
                                 ]),
        dump_default="success"
    )


"""重命名文檔 document/rename"""


# 请求
class DocumentRenameReqSchema(Schema):
    doc_id = fields.Str(description="文檔ids", required=True)
    name = fields.Str(description="文檔名", required=True)

# 返回
class DocumentRenameResSchema(CommonResponseSchema):
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['success',
                                 "Document not found!",
                                 "The extension of file can't be changed",
                                 "Duplicated document name in the same knowledgebase.",
                                 "Database error (Document rename)!",
                                 "No chunk found, please upload file and parse it."
                                 ]),
        dump_default="success"
    )


"""下载文檔 document/get/<doc_id>"""



"""修改解析方法 document/change_parser"""

class DocumentRaptorSchema(Schema):
    use_raptor = fields.Bool()

class DocumentParserConfigSchema(Schema):
    chunk_token_num = fields.Integer(dump_default=128, description="每个chunk的token数量")
    delimiter = fields.Str(dump_default="\\n!?;.。；！？", description="分割符")
    pages = fields.List(fields.Int(), allow_none=True)
    raptor = fields.Nested(DocumentRaptorSchema, description="使用召回增强RAPTOR策略")
    html4excel = fields.Bool(allow_none=True, description="")

# 请求
class DocumentChangeParserReqSchema(Schema):
    doc_id = fields.Str(description="文檔ids", required=True)
    parser_id = fields.Str(description="解析方法", required=True)
    parser_config = fields.Nested(DocumentParserConfigSchema, description="解析設置")

# 返回
class DocumentChangeParserResSchema(CommonResponseSchema):
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['success',
                                 "Document not found!",
                                 "The extension of file can't be changed",
                                 "Duplicated document name in the same knowledgebase.",
                                 "Database error (Document rename)!",
                                 "No chunk found, please upload file and parse it."
                                 ]),
        dump_default="success"
    )


"""下载图片 document/image/<image_id>"""


"""上传并且解析 document/upload_and_parse"""

# 请求
class DocumentUploadAndParseFormReqSchema(Schema):
    conversation_id = fields.Str(description="对话id conversation_id",
                         allow_none=True)

class DocumentUploadAndParseFilesReqSchema(Schema):
    conversation_id = fields.Str(description="对话id conversation_id",
                         allow_none=True)


# 返回
class DocumentUploadAndParseResSchema(CommonResponseSchema):
    data = fields.List(fields.Int(), description="成功则返回文檔id 列表，失败返回false")
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['success',
                                 "No file part!",
                                 "No file selected!",
                                 "No chunk found, please upload file and parse it."
                                 ]),
        dump_default="success"
    )