
# coding=utf-8
# @Time : 2024/12/5 09:39
# @File : common.py
from marshmallow import Schema, fields, validate

from api.settings import RetCode


class CommonResponseSchema(Schema):
    data = fields.Bool(allow_none=True, description="bool result")
    retcode = fields.Integer(
        validate=validate.OneOf(
            [RetCode.EXCEPTION_ERROR, RetCode.SUCCESS]),
        dump_default=RetCode.SUCCESS,
        allow_none=True,
        description="返回码"
    )
    retmsg = fields.Str(
        description="返回信息",
        allow_none=True,
        validate=validate.OneOf(['success', "No chunk found, please upload file and parse it."
                                 ]),
        dump_default="success"
    )


class CommonPaginationRequestSchema(Schema):
    keywords = fields.Str(
        description="关键词",
        allow_none=True
    )
    page_size = fields.Integer(
        description="分页大小",
        dump_default=10
    )
    page = fields.Integer(
        description="页码",
        dump_default=1
    )