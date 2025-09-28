# -*- coding: utf-8 -*-
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
import base64
from io import BytesIO

from PIL import Image
from bs4 import BeautifulSoup
from unstructured.partition.pdf_image.ocr import get_table_tokens
from unstructured_inference.models import tables
from unstructured_inference.models.tables import cells_to_html

from deepdoc.vision import surya_ocr
from rag.nlp import find_codec
import readability
import html_text
import chardet

def get_encoding(file):
    with open(file,'rb') as f:
        tmp = chardet.detect(f.read())
        return tmp['encoding']

class RAGFlowHtmlParser:
    def __init__(self, chat_mdl):
        super().__init__()
        self.chat_mdl = chat_mdl

    def __call__(self, fnm, binary=None):
        txt = ""
        if binary:
            encoding = find_codec(binary)
            txt = binary.decode(encoding, errors="ignore")
        else:
            with open(fnm, "r",encoding=get_encoding(fnm)) as f:
                txt = f.read()

        # 使用BeautifulSoup解析HTML
        soup = BeautifulSoup(txt, "html.parser")
        # ocr
        img_tags = soup.find_all("img")
        for img in img_tags:
            # 检查 img 的 src 是否为 base64 编码的图片
            src_value = img.get("src", '')
            if src_value.startswith('data:image'):
                image_data = base64.b64decode(src_value.split(',')[1])
                # 使用 BytesIO 将二进制数据转换为 PIL Image 对象
                image = Image.open(BytesIO(image_data))

                layout_dets = surya_ocr.surya_model.layout_detection(image)
                # 使用 bboxes 属性检查是否检测到表格
                if hasattr(layout_dets, 'bboxes') and len(layout_dets.bboxes) > 0:
                    # 假设第一个 bbox 是表格（如果它被标记为表格）
                    # 注意：您可能需要进一步检查如何确定某个 bbox 是表格
                    if hasattr(layout_dets.bboxes[0], 'label') and layout_dets.bboxes[0].label == 'Table':
                        ocr_agent = surya_ocr.OCRAgentSurya()
                        table_tokens = get_table_tokens(
                            table_element_image=image,
                            ocr_agent=ocr_agent,
                            extracted_regions=None,
                            table_element=None,
                        )
                        tatr_cells = tables.tables_agent.predict(
                            image, ocr_tokens=table_tokens, result_format="cells"
                        )
                        if tatr_cells:
                            text_as_html = "" if tatr_cells == "" else cells_to_html(tatr_cells)
                            img.replace_with(text_as_html)
                    else:
                        p_tag = soup.new_tag('p')
                        p_tag.string = surya_ocr.surya_model.predict(image, self.chat_mdl)
                        # 用 p 标签替换 img 标签
                        img.replace_with(p_tag)
                else:
                    p_tag = soup.new_tag('p')
                    p_tag.string = surya_ocr.surya_model.predict(image, self.chat_mdl)
                    # 用 p 标签替换 img 标签
                    img.replace_with(p_tag)

        return self.parser_txt(txt)

    @classmethod
    def parser_txt(cls, txt):
        if type(txt) != str:
            raise TypeError("txt type should be str!")
        html_doc = readability.Document(txt)
        title = html_doc.title()
        content = html_text.extract_text(html_doc.summary(html_partial=True))
        if title == '[no-title]':
            txt = f"{content}"
        else:
            txt = f"{title}\n{content}"
        sections = txt.split("\n")
        return sections
