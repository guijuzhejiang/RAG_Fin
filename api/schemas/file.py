# coding=utf-8
# @Time : 2024/12/3 17:16
# @File : system.py
from marshmallow import Schema, fields, validate
from api.schemas.common import CommonResponseSchema, CommonPaginationRequestSchema
from api.schemas.db_models import FileSchema
from flask_smorest.fields import Upload

"""上传文件 file/upload"""
# 请求
class FileUploadFilesReqSchema(Schema):
    file = fields.List(Upload(), description="Binary file data", load_only=True)

class FileUploadFormReqSchema(Schema):
    parent_id = fields.Str(
        description="parent_id",
        allow_none=True
    )
    kb_id = fields.Str(
        description="知识库id",
        allow_none=True
    )
    path = fields.Str(
        description="",
        allow_none=True
    )

# 返回
class FileUploadResSchema(CommonResponseSchema):
    data = fields.Bool(allow_none=True, description="bool result")


"""創建文件 file/create"""

# 请求
class FileCreateReqSchema(Schema):
    parent_id = fields.Str(
        description="parent_id",
        allow_none=True
    )
    type = fields.Str(
        description="input_file_type",
        allow_none=True
    )
    kb_id = fields.Str(
        description="知识库id",
        allow_none=True
    )
    name = fields.Str(
        description="文件名",
        required=True
    )

# 返回
class FileCreateResSchema(CommonResponseSchema):
    data = fields.Nested(FileSchema, description="文件信息")


"""文件管理 获取所有文件 file/list"""


class FileListFilesFolderSchema(Schema):
    create_date = fields.Str(required=True, description="Creation date in GMT format")
    create_time = fields.Int(required=True, description="Creation time in epoch milliseconds")
    created_by = fields.Str(required=True, description="ID of the creator")
    id = fields.Str(required=True, description="Unique identifier of the folder")
    location = fields.Str(allow_none=True, description="Location of the folder")
    name = fields.Str(required=True, description="Name of the folder")
    parent_id = fields.Str(required=True, description="Parent folder ID")
    size = fields.Int(required=True, description="Size of the folder in bytes")
    source_type = fields.Str(allow_none=True, description="Source type of the folder")
    tenant_id = fields.Str(required=True, description="Tenant ID associated with the folder")
    type = fields.Str(required=True, description="Type of the folder (e.g., folder, file)")
    update_date = fields.Str(required=True, description="Last update date in GMT format")
    update_time = fields.Int(required=True, description="Last update time in epoch milliseconds")

class FileListFilesResDataSchema(Schema):
    files = fields.List(fields.Nested(FileSchema), description="List of models for the category")
    parent_folder = fields.Nested(FileListFilesFolderSchema, description="文件夹信息")
    total = fields.Int(description="文件总数")


# 请求
class FileListFilesReqSchema(CommonPaginationRequestSchema):
    parent_id = fields.Str(
        description="parent_id",
        allow_none=True
    )

# 返回
class FileListFilesResSchema(CommonResponseSchema):
    data = fields.Nested(FileListFilesResDataSchema, description="文件信息")


"""获取root_folder file/root_folder"""

class FileListFilesResDataRootFolderSchema(Schema):
    id = fields.Str(description="file_id")
    parent_id = fields.Str(description="parent_id")
    tenant_id = fields.Str(description="tenant_id")
    created_by = fields.Str(description="created_by")
    name = fields.Str(dump_default='/', description="name ")
    type = fields.Str(dump_default='folder', description="文件类型", validate=validate.OneOf(["pdf", "doc", "visual", 'aural', 'virtual', 'folder', 'other']))
    size = fields.Integer(dump_default=0, description="文件size")
    location = fields.Str(dump_default="", description="location")

class FileListFilesResDataSchema(Schema):
    root_folder = fields.Nested(FileListFilesResDataRootFolderSchema, description="FileListFilesResDataRootFolder")

# 返回
class FileGetRootFolderResSchema(CommonResponseSchema):
    data = fields.Nested(FileListFilesResDataSchema, description="文件信息")


"""获取parent_folder file/parent_folder"""

class ParentFolderSchema(Schema):
    parent_folder = fields.Nested(FileSchema, description="文件夾信息")

# 请求
class FileGetParentFolderReqSchema(Schema):
    file_id = fields.Str(
        description="file_id",
        allow_none=True
    )

# 返回
class FileGetParentFolderResSchema(CommonResponseSchema):
    data = fields.Nested(ParentFolderSchema, description="文件夾信息")
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['success', "Folder not found!", "No chunk found, please upload file and parse it."
                                 ]),
        dump_default="success"
    )

"""获取全部parent_folder file/all_parent_folder"""

class ParentFoldersSchema(Schema):
    parent_folders = fields.List(fields.Nested(FileSchema), description="文件夾信息")

# 请求
class FileGetParentFolderReqSchema(Schema):
    file_id = fields.Str(
        description="file_id",
        allow_none=True
    )

# 返回
class FileGetParentFolderResSchema(CommonResponseSchema):
    data = fields.Nested(ParentFoldersSchema, description="文件夾信息")
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['success', "Folder not found!", "No chunk found, please upload file and parse it."
                                 ]),
        dump_default="success"
    )


"""删除文件 file/rm"""

# 请求
class FileRmReqSchema(Schema):
    file_ids = fields.List(fields.Str(),
        description="file_id",
        allow_none=True
    )
    parent_id = fields.Str(
        description="parent_id",
        allow_none=True
    )

# 返回
class FileRmResSchema(CommonResponseSchema):
    data = fields.Nested(ParentFoldersSchema, description="文件夾信息")
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['success', "Folder not found!", "No chunk found, please upload file and parse it."
                                 ]),
        dump_default="success"
    )


"""文件重命名 file/rename"""

# 请求
class FileRenameReqSchema(Schema):
    file_id = fields.Str(
        description="file_id",
        required=True
    )
    name = fields.Str(
        description="name",
        required=True
    )

# 返回
class FileRenameResSchema(CommonResponseSchema):
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['success', "Database error (Document rename)!", "File not found!", "The extension of file can't be changed", "Duplicated file name in the same folder.", "No chunk found, please upload file and parse it."
                                 ]),
        dump_default="success"
    )


"""获取文件 file/get/<file_id>"""

# 返回
class FileGetResSchema(CommonResponseSchema):
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['success', "Document not found!", "No chunk found, please upload file and parse it."
                                 ]),
        dump_default="success"
    )


"""文件移动 file/mv"""

# 请求
class FileMvReqSchema(Schema):
    src_file_ids = fields.List(fields.Str(),
        description="file_ids",
        required=True
    )
    dest_file_id = fields.Str(
        description="dest_file_id",
        required=True
    )

# 返回
class FileMvResSchema(CommonResponseSchema):
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['success', "File or Folder not found!", "File not found!", "Parent Folder not found!", "No chunk found, please upload file and parse it."
                                 ]),
        dump_default="success"
    )