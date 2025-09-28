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
from deepdoc.parser.utils import get_text
from rag.nlp import num_tokens_from_string
import re


class RAGFlowTxtParser:
    def __call__(self, fnm, binary=None, chunk_token_num=128, delimiter="\n!?;.。；！？"):
        txt = get_text(fnm, binary)
        return self.parser_txt(txt, chunk_token_num, delimiter)

    @classmethod
    def parser_txt(cls, txt, chunk_token_num=128, delimiter="\n!?;.。；！？"):
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

        return [[c,""] for c in cks]