# coding=utf-8
# @Time : 2024/12/3 17:16
# @File : system.py
from marshmallow import Schema, fields, validate

from api.schemas.common import CommonResponseSchema
from api.settings import RetCode

"""获取系统版本 system/version"""

# 获取系统版本返回
class SystemVersionResSchema(Schema):
    data = fields.Str(
        description="系统版本",
        dump_default="dev"
    )


"""获取系统版本 system/status"""

# 获取系统状态

class DatabaseSchema(Schema):
    database = fields.Str(description="Database type, e.g., 'mysql'")
    elapsed = fields.Str(description="Time elapsed in seconds")
    status = fields.Str(description="Status of the database, e.g., 'green'")

class ESSchema(Schema):
    active_primary_shards = fields.Int(description="Number of active primary shards")
    active_shards = fields.Int(description="Number of active shards")
    active_shards_percent_as_number = fields.Float(description="Percentage of active shards")
    cluster_name = fields.Str(description="Name of the Elasticsearch cluster")
    delayed_unassigned_shards = fields.Int(description="Number of delayed unassigned shards")
    elapsed = fields.Str(description="Time elapsed in seconds")
    initializing_shards = fields.Int(description="Number of initializing shards")
    number_of_data_nodes = fields.Int(description="Number of data nodes in the cluster")
    number_of_in_flight_fetch = fields.Int(description="Number of in-flight fetch operations")
    number_of_nodes = fields.Int(description="Number of nodes in the cluster")
    number_of_pending_tasks = fields.Int(description="Number of pending tasks in the cluster")
    relocating_shards = fields.Int(description="Number of relocating shards")
    status = fields.Str(description="Status of the Elasticsearch cluster, e.g., 'green'")
    task_max_waiting_in_queue_millis = fields.Int(description="Maximum waiting time in task queue (ms)")
    timed_out = fields.Bool(description="Whether the cluster timed out")
    unassigned_shards = fields.Int(description="Number of unassigned shards")

class RedisSchema(Schema):
    elapsed = fields.Str(description="Time elapsed in seconds")
    status = fields.Str(description="Status of Redis, e.g., 'green'")

class StorageSchema(Schema):
    elapsed = fields.Str(description="Time elapsed in seconds")
    status = fields.Str(description="Status of the storage, e.g., 'green'")
    storage = fields.Str(description="Storage type, e.g., 'minio'")

class TaskExecutorSchema(Schema):
    elapsed = fields.Dict(keys=fields.Str(), values=fields.List(fields.Float()), description="Elapsed time for each task consumer")
    status = fields.Str(description="Status of the task executor, e.g., 'green'")

class DataSchema(Schema):
    database = fields.Nested(DatabaseSchema, required=True)
    es = fields.Nested(ESSchema, required=True)
    redis = fields.Nested(RedisSchema, required=True)
    storage = fields.Nested(StorageSchema, required=True)
    task_executor = fields.Nested(TaskExecutorSchema, required=True)

class SystemStatusResSchema(CommonResponseSchema):
    data = fields.Nested(DataSchema, description="All system status data")
