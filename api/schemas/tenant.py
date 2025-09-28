# coding=utf-8
# @Time : 2024/12/3 15:35
# @File : tenant.py.py
from marshmallow import Schema, fields, validate

from api.schemas.common import CommonResponseSchema
from api.schemas.db_models import TenantSchema, UserSchema
from api.settings import RetCode


""" 获取当前用户模型设置列表 tenant/list """


class TenantListResSchema(CommonResponseSchema):
    data = fields.Nested(TenantSchema, description="Tenant data object")
    retcode = fields.Integer(
        validate=validate.OneOf(
            [RetCode.SUCCESS, RetCode.UNAUTHORIZED, RetCode.EXCEPTION_ERROR]),
        description="返回码"
    )


""" 获取tenant_id对应的用户列表 tenant/user/list """


class TenantUserListResSchema(CommonResponseSchema):
    data = fields.Nested(UserSchema, description="User data object")
    retcode = fields.Integer(
        validate=validate.OneOf(
            [RetCode.SUCCESS, RetCode.UNAUTHORIZED, RetCode.EXCEPTION_ERROR, ]),
        description="返回码"
    )


""" 获取user_id对应的tenant tenant/user """


class TenantCreateReqSchema(Schema):
    user_id = fields.Str(required=True, description="User id")


class TenantCreateResDataSchema(Schema):
    id = fields.Str(required=True, description="User id")

class TenantCreateResSchema(CommonResponseSchema):
    data = fields.Nested(TenantCreateResDataSchema, description="User id object")
    retcode = fields.Integer(
        validate=validate.OneOf(
            [RetCode.ARGUMENT_ERROR, RetCode.EXCEPTION_ERROR, RetCode.SUCCESS]),
        description="返回码"
    )
    retmsg = fields.Str(
        description="返回信息",
        validate=validate.OneOf(['success',
                                 "No chunk found, please upload file and parse it.",
                                 'Lack of "USER ID"',
                                 ]),
        required=True,
        dump_default="success"
    )


""" 删除user_id,tenant_id对应的tenant teanant/<tenant_id>/user/<user_id> """


class TenantRmResSchema(CommonResponseSchema):
    data = fields.Bool(description="bool result")