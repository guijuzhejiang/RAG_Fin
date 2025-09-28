# coding=utf-8
# @Time : 2024/12/3 15:32
# @File : db_models.py
from marshmallow import Schema, fields, validate


# 用户信息字段
class UserSchema(Schema):
    id = fields.Str(description="Primary key, user ID (max length: 32)")
    access_token = fields.Str(allow_none=True, description="Access token for the user (max length: 255)")
    nickname = fields.Str(description="Nickname (max length: 100)")
    password = fields.Str(allow_none=True, description="User password (max length: 255)")
    email = fields.Str(description="User email (max length: 255)")
    avatar = fields.Str(allow_none=True, description="Avatar as base64 string")
    language = fields.Str(
        allow_none=True,
        validate=validate.OneOf(["English", "Chinese", "Japanese"]),
        dump_default="Japanese",
        description="Preferred language (default: Japanese)"
    )
    color_schema = fields.Str(
        allow_none=True,
        validate=validate.OneOf(["Bright", "Dark"]),
        dump_default="Bright",
        description="Preferred color scheme (default: Bright)"
    )
    timezone = fields.Str(
        allow_none=True,
        dump_default="UTC+8\tAsia/Shanghai",
        description="Timezone (default: UTC+8\tAsia/Shanghai)"
    )
    last_login_time = fields.DateTime(allow_none=True, description="Last login time")
    is_authenticated = fields.Str(

        validate=validate.OneOf(["0", "1"]),
        dump_default="1",
        description="Is user authenticated (0: no, 1: yes)"
    )
    is_active = fields.Str(
        validate=validate.OneOf(["0", "1"]),
        dump_default="1",
        description="Is user active (0: no, 1: yes)"
    )
    is_anonymous = fields.Str(

        validate=validate.OneOf(["0", "1"]),
        dump_default="0",
        description="Is user anonymous (0: no, 1: yes)"
    )
    login_channel = fields.Str(allow_none=True, description="The channel through which the user logged in")
    status = fields.Str(
        allow_none=True,
        validate=validate.OneOf(["0", "1"]),
        dump_default="1",
        description="Validation status (0: wasted, 1: validate)"
    )
    is_superuser = fields.Bool(allow_none=True, dump_default=False,
                               description="Is the user a superuser (default: False)")


# 模型设置
class TenantSchema(Schema):
    asr_id = fields.Str(description="ASR ID, e.g., 'paraformer-realtime-8k-v1'")
    embd_id = fields.Str(description="Embedding ID, e.g., 'nomic-embed-text:v1.5@Ollama'")
    img2txt_id = fields.Str(description="Image-to-Text model ID, e.g., 'blaifa/InternVL3_5:8b@Ollama'")
    llm_id = fields.Str(description="LLM ID, e.g., 'qwen3-14B-think-Q4_K_M@Ollama'")
    name = fields.Str(description="Name, e.g., 'ray’s Kingdom'")
    parser_ids = fields.Str(description="Parser IDs as a comma-separated string, e.g., 'naive:General,qa:Q&A,...'")
    rerank_id = fields.Str(
        description="Re-ranker model ID, e.g., 'bge-reranker-v2-m3'")
    role = fields.Str(description="Role of the user, e.g., 'owner'")
    tenant_id = fields.Str(description="Tenant ID, e.g., '390df030641b11ef8a0a00d861bc2f9a'")
    tts_id = fields.Str(description="TTS ID, nullable field")


# 知识库
class KbRaptorConfigSchema(Schema):
    use_raptor = fields.Bool(description="Whether to use Raptor for parsing")


class KbParserConfigSchema(Schema):
    chunk_token_num = fields.Int(description="Number of tokens per chunk")
    delimiter = fields.Str(description="Delimiters for parsing")
    html4excel = fields.Bool(description="Whether to use HTML for Excel parsing")
    layout_recognize = fields.Bool(description="Whether to recognize layout")
    raptor = fields.Nested(KbRaptorConfigSchema, description="Raptor configuration")


class KbSchema(Schema):
    avatar = fields.Str(description="URL or path to the avatar")
    chunk_num = fields.Int(description="Number of chunks")
    create_date = fields.Str(description="Creation date in GMT format")
    create_time = fields.Int(description="Creation time in epoch milliseconds")
    created_by = fields.Str(description="ID of the creator")
    description = fields.Str(description="Description of the entity")
    doc_num = fields.Int(description="Number of documents")
    embd_id = fields.Str(description="Embedding ID")
    id = fields.Str(description="Unique identifier")
    language = fields.Str(description="Language of the entity")
    name = fields.Str(description="Name of the entity")
    parser_config = fields.Nested(KbParserConfigSchema, description="Parser configuration details")
    parser_id = fields.Str(description="ID of the parser used")
    permission = fields.Str(description="Permission level (e.g., 'me', 'all')")
    similarity_threshold = fields.Float(description="Similarity threshold")
    status = fields.Str(description="Status of the entity")
    tenant_id = fields.Str(description="Tenant ID")
    token_num = fields.Int(description="Number of tokens")
    update_date = fields.Str(description="Update date in GMT format")
    update_time = fields.Int(description="Update time in epoch milliseconds")
    vector_similarity_weight = fields.Float(description="Weight for vector similarity calculation")



# file
class FileParserConfigSchema(Schema):
    field_map = fields.Dict(
        keys=fields.Str(description="Field map key"),
        values=fields.Str(description="Field map value"),
        description="Mapping of fields"
    )
    pages = fields.List(
        fields.List(fields.Int(), validate=lambda x: len(x) == 2),
        description="Page ranges (start, end)"
    )

class FileSchema(Schema):
    chunk_num = fields.Int(description="Number of chunks")
    create_date = fields.Str(description="Creation date")
    create_time = fields.Int(description="Creation time in epoch")
    created_by = fields.Str(description="ID of the creator")
    id = fields.Str(description="Unique identifier")
    kb_id = fields.Str(description="Knowledge Base ID")
    location = fields.Str(description="Location info", allow_none=True)
    name = fields.Str(description="Name of the resource")
    parser_config = fields.Nested(FileParserConfigSchema, description="Parser configuration details")
    parser_id = fields.Str(description="ID of the parser")
    process_begin_at = fields.Str(allow_none=True, description="Start time of processing")
    process_duation = fields.Float(description="Processing duration in seconds")
    progress = fields.Float(description="Progress as a percentage")
    progress_msg = fields.Str(allow_none=True, description="Progress message")
    run = fields.Str(description="Run state")
    size = fields.Int(description="Size of the file/resource")
    source_type = fields.Str(description="Type of source")
    status = fields.Str(description="Status of the resource")
    thumbnail = fields.Str(allow_none=True, description="Thumbnail URL or ID")
    token_num = fields.Int(description="Number of tokens")
    type = fields.Str(description="Type of resource")
    update_date = fields.Str(description="Update date")
    update_time = fields.Int(description="Update time in epoch")


# Dialog

class LLMSettingSchema(Schema):
    frequency_penalty = fields.Float(validate=validate.Range(min=0.0))
    max_tokens = fields.Integer(validate=validate.Range(min=1))
    presence_penalty = fields.Float(validate=validate.Range(min=0.0))
    temperature = fields.Float(validate=validate.Range(min=0.0, max=1.0))
    top_p = fields.Float(validate=validate.Range(min=0.0, max=1.0))

class PromptConfigParameterSchema(Schema):
    key = fields.String()
    optional = fields.Boolean()

class PromptConfigSchema(Schema):
    empty_response = fields.String()
    parameters = fields.List(fields.Nested(PromptConfigParameterSchema), )
    prologue = fields.String()
    quote = fields.Boolean()
    refine_multiturn = fields.Boolean()
    system = fields.String()
    tts = fields.Boolean()

class DialogSchema(Schema):
    create_date = fields.String(validate=validate.Regexp(r'.*\d{4}.*GMT$'))
    create_time = fields.Integer()
    description = fields.String()
    do_refer = fields.String(validate=validate.OneOf(["0", "1"]))
    icon = fields.String()
    id = fields.String()
    kb_ids = fields.List(fields.String(), )
    kb_names = fields.List(fields.String(), )
    language = fields.String()
    llm_id = fields.String()
    llm_setting = fields.Nested(LLMSettingSchema, )
    name = fields.String()
    prompt_config = fields.Nested(PromptConfigSchema, )
    prompt_type = fields.String()
    rerank_id = fields.String()
    similarity_threshold = fields.Float(validate=validate.Range(min=0.0, max=1.0))
    status = fields.String(validate=validate.OneOf(["0", "1"]))
    tenant_id = fields.String()
    top_k = fields.Integer(validate=validate.Range(min=1))
    top_n = fields.Integer(validate=validate.Range(min=1))
    update_date = fields.String(validate=validate.Regexp(r'.*\d{4}.*GMT$'))
    update_time = fields.Integer()
    vector_similarity_weight = fields.Float(validate=validate.Range(min=0.0, max=1.0))


# conversation

class ConversationReferenceSchema(Schema):
    chunks = fields.List(fields.Raw(), )
    doc_aggs = fields.List(fields.Raw(), )
    total = fields.Integer()

class ConversationMessageSchema(Schema):
    content = fields.String()
    role = fields.String()
    id = fields.String()
    doc_ids = fields.List(fields.String(), required=False, allow_none=True)  # Optional for "doc_ids"
    prompt = fields.String(allow_none=True)  # Optional for "prompt"

class ConversationSchema(Schema):
    create_date = fields.String()
    create_time = fields.Integer()
    dialog_id = fields.String()
    id = fields.String()
    message = fields.List(fields.Nested(ConversationMessageSchema))
    name = fields.String()
    reference = fields.List(fields.Nested(ConversationReferenceSchema))
    update_date = fields.String()
    update_time = fields.Integer()