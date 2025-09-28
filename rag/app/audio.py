#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import io
import re
import numpy as np

from api.db import LLMType
from api.db.services.llm_service import LLMBundle
from rag.nlp import naive_merge, rag_tokenizer, num_tokens_from_string, tokenize_chunks


def chunk(filename, binary, tenant_id, lang, callback=None, **kwargs):
    # chat_mdl = kwargs.get('chat_mdl')
    doc = {
        "docnm_kwd": filename,
        "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", filename))
    }
    doc["title_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["title_tks"])
    chunk_token_num = 128
    delimiter = "\n!?;.。；！？"
    # is it English
    eng = lang.lower() == "english"  # is_english(sections)
    try:
        callback(0.1, "USE Sequence2Txt LLM to transcription the audio")
        seq2txt_mdl = LLMBundle(tenant_id, LLMType.SPEECH2TEXT, lang=lang)
        ans = seq2txt_mdl.transcription(binary)
        ans = ans.replace("\n", "")
        # ans = chat_gen(ans, chat_mdl)
        sections = parser_txt(ans, chunk_token_num, delimiter)
        callback(0.8, "Sequence2Txt LLM respond: %s ..." % ans[:32])
        chunks = naive_merge(sections, chunk_token_num, delimiter)
        res = tokenize_chunks(chunks, doc, eng)
        return res
    except Exception as e:
        callback(prog=-1, msg=str(e))

    return []

def chat_gen(text, chat_mdl):
    # 定义你想删除的特殊字符
    # pattern = r'[。？！；!?;+.]+'
    # text = re.sub(pattern, '', text)
    if len(text)<2:
        return
    prompt_user = f"""
            以下のテキストはWhisperモデルによって音声からテキストに変換された結果:
            {text}
            上記のテキストは、Whisperモデルによって音声からテキストに変換されたテキストです。
            音声認識には誤認や余分な言葉が含まれる可能性があります。文脈に基づいて、誤りがある場合は正しいテキストに修正してください。
            また、挨拶、世間話、感謝の言葉など、実際の内容と関係のない部分は省略してください。
            日本語で出力しなければならない。
            修正後の内容のみを出力してください。説明や補足情報は一切含めないでください。
            余計な前置きや説明を出力しないでください。
            テキストに含まれるUnicodeシンボル（例えば表情符号や特殊記号など）や無関係なコンテンツはすべて削除してください。必要な情報のみを残してください。
            """
    prompt_system = "あなたは断片化された様々な情報を整理し、そこから有用な情報を抽出するのが得意なAIアシスタントです。"
    res = chat_mdl.chat(prompt_system,
                        [{"role": "user", "content": prompt_user}],
                        {"temperature": 0.0}
                        )
    return res

def parser_txt(txt, chunk_token_num=128, delimiter="\n!?;.。；！？"):
    if not isinstance(txt, str):
        raise TypeError("txt type should be str!")
    cks = [""]
    tk_nums = [0]
    # 使用正则表达式匹配标点符号和换行符，但确保不会误将字母 'n' 作为分隔符
    # 将换行符作为单独的条件，其他标点符号分开处理
    # delimiter_pattern = r"[\n!?;。；！？]"
    delimiter = delimiter.replace("\\n", "\n")
    delimiter_pattern = f"[{re.escape(delimiter)}]"

    def add_chunk(t):
        nonlocal cks, tk_nums, delimiter
        tnum = num_tokens_from_string(t)
        if tnum < 8:
            pos = ""
        if tk_nums[-1] > chunk_token_num:
            cks.append(t)
            tk_nums.append(tnum)
        else:
            cks[-1] += t
            tk_nums[-1] += tnum

    # 分割文本时使用正则表达式，匹配换行符和标点符号
    chunks = re.split(f"({delimiter_pattern})", txt)
    merged_chunks = []
    for i in range(0, len(chunks), 2):  # 遍历分割的内容
        sentence = chunks[i].strip()
        if i + 1 < len(chunks):
            # 如果有分隔符，把它加到句子末尾
            separator = chunks[i + 1]
            sentence += separator
        if sentence:
            merged_chunks.append(sentence)

    for chunk in merged_chunks:
        add_chunk(chunk)

    return [[c, ""] for c in cks]
