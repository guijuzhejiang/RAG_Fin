import base64
import os.path
import random
import re
import string
import traceback
# import magic
# import mimetypes
from bs4 import BeautifulSoup
from spire.xls import Workbook as spireWorkbook
from spire.xls.common import *
from aspose.cells import Workbook, HtmlSaveOptions
import subprocess
from io import BytesIO
from PIL import Image
import tempfile
import os

from unstructured.partition.pdf_image.ocr import get_table_tokens
from unstructured_inference.inference.elements import TextRegion
from unstructured_inference.inference.layoutelement import LayoutElement
from unstructured_inference.models import tables
from unstructured_inference.models.tables import cells_to_html

from deepdoc.vision import surya_ocr

from rag.nlp import find_codec
from itertools import zip_longest


def get_text(fnm: str, binary=None) -> str:
    txt = ""
    if binary:
        encoding = find_codec(binary)
        txt = binary.decode(encoding, errors="ignore")
    else:
        with open(fnm, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                txt += line
    return txt



def clean_text(text):
    """
       清理文本中的不必要字符，包括多余的空格、换行符和特殊字符。
       该函数首先移除中日文字符之间的多余空格，然后根据文本是否以换行符结尾来决定如何处理换行符，
       最后移除所有非中日文字符、非空白字符和非字母数字字符。

       参数:
       text (str): 需要清理的文本字符串。

       返回:
       str: 清理后的文本字符串。
   """
    # 正则表达式，匹配中文或日文字符之间的空格
    space_pattern = r'([\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff])\s+([\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff])'

    # 循环替换，直到没有更多的匹配项
    while re.search(space_pattern, text):
        text = re.sub(space_pattern, r'\1\2', text)

    # 正则表达式，匹配句子中的换行符
    if text.endswith("\n"):  # 检查末尾是否有换行符
        text = re.sub(r"[\n]+", "", text[:-1]) + "\n"  # 删除中间的空格和换行符，保留末尾的换行符
    else:
        text = re.sub(r"[\n]+", "", text)  # 删除所有空格和换行符

    # 正则表达式，匹配特殊字符，但保留正常的标点符号
    special_chars_pattern = r'[^\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\s\w.。,，、?？!！;；:："“”‘’()（）…]'
    text = re.sub(special_chars_pattern, '', text)
    return text


def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    random_string = "".join(random.choice(characters) for _ in range(length))
    return random_string


def is_empty(tag):
    # 判断一个标签是否为空（没有内容或只有空格）
    return not tag.get_text(strip=True)


def clean_table(table):
    # 遍历所有的 <table>、<tr>、<td>、<th> 标签
    for tag in table.find_all(["table", "tr", "td", "th"]):
        # 创建一个新的属性字典，只保留 colspan 和 rowspan
        new_attrs = {}
        if "colspan" in tag.attrs:
            new_attrs["colspan"] = tag.attrs["colspan"]
        if "rowspan" in tag.attrs:
            new_attrs["rowspan"] = tag.attrs["rowspan"]
        tag.attrs = new_attrs

    # 遍历并删除所有的 <font> 标签，但保留其内容
    for font_tag in table.find_all(["font", "div", "span"]):
        font_tag.unwrap()  # 去掉 <font> 标签但保留其中的内容

    # 移除空的单元格（列）
    for col in table.find_all(["col", "img", "?if supportMisalignedColumns?", "?endif?"]):
        col.decompose()

    # 找到所有的行
    rows = table.find_all("tr")
    #
    # if chunk_rows > 0:
    #     reserve_len = (len(rows) - 1) // chunk_rows + 1
    #     rows = rows[:reserve_len]

    # 找到每一列的所有单元格
    columns = list(zip(*[row.find_all('td') for row in rows]))

    # 检查每一列是否为空（即所有单元格都是空的）
    for i, col in enumerate(columns):
        if all(cell.get_text(strip=True) == "" for cell in col):
            # 如果该列所有单元格都为空，则删除该列
            for cell in col:
                cell.extract()

    # 移除空的行
    for row in rows:
        if all(is_empty(cell) for cell in row.find_all(["td", "th"])):
            row.decompose()  # 移除空的行
        else:
            # 合并空的td
            empty_cell_count = 0
            start_cell = None
            # start_cell_idx = None
            for cell_idx, cell in enumerate(row.find_all("td")):
                empty_cell = is_empty(cell)
                if empty_cell:
                    empty_cell_count = empty_cell_count + 1
                    if start_cell:
                        cell.decompose()
                    else:
                        start_cell = cell
                else:
                    if empty_cell_count > 0:
                        start_cell.attrs = {"colspan": empty_cell_count}
                    empty_cell_count = 0
                    start_cell = None
            else:
                if start_cell:
                    if empty_cell_count > 0:
                        start_cell.attrs = {"colspan": empty_cell_count}

                # 最后一个cell是空的删除
                cur_row_last_cell = row.find_all("td")[-1]
                if is_empty(cur_row_last_cell):
                    cur_row_last_cell.decompose()

    return table


def extract_table_from_html(excel_path):
    # 读取HTML文件
    with open(excel_path, "r", encoding="utf-8") as file:
        html_content = file.read()

    # 使用BeautifulSoup解析HTML
    soup = BeautifulSoup(html_content, "html.parser")

    # 找到所有的table标签
    tables = soup.find_all("table")
    # 处理并清理每一个表格
    cleaned_tables = [clean_table(table) for table in tables]

    return cleaned_tables


# def get_content_type(file_binary, file_name=None):
#     mime = magic.Magic(mime=True)
#     content_type = mime.from_buffer(file_binary)
#     # 如果有文件名，进一步验证
#     if file_name:
#         guessed_type, _ = mimetypes.guess_type(file_name)
#         if guessed_type:
#             content_type = guessed_type
#     return content_type


class SpireConverter:
    def __init__(self):
        pass

    @staticmethod
    def SaveToHtml(fnm):
        # Create a Workbook instance
        wb = spireWorkbook()
        # Load a sample Excel file
        if isinstance(fnm, str):
            wb.LoadFromFile(fnm)
        else:
            wb.LoadFromStream(Stream(fnm))

        # sheet name and cache path
        res = []
        for sheet in wb.Worksheets:
            cache_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f"tmp_{generate_random_string(6)}.html",
            )
            sheet.SaveToHtml(cache_path)

            res.append((sheet.Name, cache_path))
        return res


class AsposeConverter:
    def __init__(self):
        pass

    @staticmethod
    def SaveToHtml(fnm, chat_mdl):
        # Load a sample Excel file
        if isinstance(fnm, str):
            wb = Workbook(fnm)
        else:
            wb = Workbook(BytesIO(fnm))

        # 创建 HtmlSaveOptions 对象
        html_options = HtmlSaveOptions()
        # 启用导出所有工作表为一个 HTML 文件
        html_options.export_active_worksheet_only = False
        html_options.save_as_single_file = True
        cache_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"tmp_{generate_random_string(6)}.html",
        )
        wb.save(cache_path, html_options)
        # 读取HTML文件
        with open(cache_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        # 使用BeautifulSoup解析HTML
        soup = BeautifulSoup(html_content, "html.parser")

        # remove eval warning
        for tag in soup.find_all(string="Evaluation Only. Created with Aspose.Cells for Python via .NET. Copyright 2003 - 2024 Aspose Pty Ltd."):
            tag.parent.decompose()
        for div in soup.find_all('div', attrs={'sheetname': 'Evaluation Warning'}):
            div.decompose()

        divs = soup.find_all('div', id=re.compile(r'^table_\d+$'))
        # 找到所有的table标签
        tables = soup.find_all("table")

        # ocr
        try:
            from deepdoc.vision.surya_ocr import surya_model

            for table in tables:
                for img in table.find_all('img'):
                    # 获取 src 属性的值
                    img_data = img['src']
                    # 去掉前缀 'data:image/png;base64,'，获取纯 base64 数据
                    base64_data = img_data.split(',')[1]
                    if len(base64_data) > 0:
                        # 解码 base64 数据
                        image_blob = base64.b64decode(base64_data)
                        # 将解码后的数据转换为 PIL Image 对象
                        img_parser = ImageParser(chat_mdl)
                        ocr_text = img_parser.get_picture(image_blob)
                        if ocr_text:
                            img.replace_with(ocr_text)

        except Exception:
            print(traceback.format_exc())

        # 保存为 HTML 文件
        # wb.save(cache_path, html_options)
        # sheet name and cache path
        res = []
        for sheet, div in zip_longest(wb.worksheets, divs):
            if not div:
                break
            sheet_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f"tmp_{generate_random_string(6)}.html",
            )
            with open(sheet_path, "w", encoding="utf-8") as file:
                file.write(str(div))
            res.append((sheet.name, sheet_path))

        if os.path.exists(cache_path):
            os.remove(cache_path)
        return res

class ImageParser():
    def __init__(self, chat_mdl):
        self.chat_mdl = chat_mdl

    def wmf_to_png(self, wmf_data):
        # 保存为 JPEG 文件
        # Specify the desired filename
        desired_filename = "tmp"

        # Create a temporary directory to ensure the file is unique and safe
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_wmf_path = os.path.join(temp_dir, desired_filename+".wmf")
            with open(temp_wmf_path, 'wb') as f:
                f.write(wmf_data)

            temp_png_path = os.path.join(temp_dir, desired_filename+'.png')

            subprocess.check_call([
                "libreoffice",
                "--headless",
                "--convert-to",
                "png",
                temp_wmf_path,
                "--outdir",
                temp_dir,
            ])
            return Image.open(temp_png_path).convert('RGB')


    def get_picture(self, image_blob):
        """
        Extract the image from a shape in a slide.
        """
        try:
            # image = Image.open(BytesIO(image_blob)).convert('RGB')
            image = Image.open(BytesIO(image_blob))
            if (image.format == 'WMF'):
                # 将 WMF 转换为 PNG
                image = self.wmf_to_png(image_blob)
            image = image.convert('RGB')
            # have table
            layout_pred = surya_ocr.surya_model.det_target(image, 'Table')
            if layout_pred:
                table_region = LayoutElement.from_coords(
                    layout_pred.bbox[0],
                    layout_pred.bbox[1],
                    layout_pred.bbox[2],
                    layout_pred.bbox[3],
                    text=None,
                    type=layout_pred.label,
                    prob=None,
                    source="surya"
                )

                # run ocr
                img_pred = surya_ocr.surya_model.run_ocr(
                    [image], batch_size=8
                )[0]

                text_regions: list["TextRegion"] = []
                for t in img_pred.text_lines:
                    if t.text:
                        text_region = LayoutElement.from_coords(
                            t.bbox[0],
                            t.bbox[1],
                            t.bbox[2],
                            t.bbox[3],
                            text=None,
                            type='Text',
                            prob=None,
                            source="surya"
                        )
                        text_regions.append(text_region)

                # table struct infer
                table_tokens = get_table_tokens(
                    table_element_image=image,
                    ocr_agent=surya_ocr.OCRAgentSurya(),
                    extracted_regions=None if len(text_regions) < 1 else text_regions,
                    table_element=table_region,
                )
                tatr_cells = tables.tables_agent.predict(
                    image, ocr_tokens=table_tokens, result_format="cells"
                )
                if tatr_cells:
                    text_from_ocr = "" if tatr_cells == "" else cells_to_html(tatr_cells)
                else:
                    text_from_ocr = surya_ocr.surya_model.predict(image, self.chat_mdl)
            else:
                text_from_ocr = surya_ocr.surya_model.predict(image, self.chat_mdl)


            return text_from_ocr
        except Exception as e:
            print(f"Failed to extract image: {e}")
            return None