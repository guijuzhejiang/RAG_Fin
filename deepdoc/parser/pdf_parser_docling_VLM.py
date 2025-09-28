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
import os
import re
import warnings
import logging
from typing import List, Tuple, Dict, Any, Optional, Set
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
from pathlib import Path
import time
import tempfile
from io import BytesIO
from PIL import Image


# 修复ONNX Runtime TensorRT警告
warnings.filterwarnings("ignore", message=".*TensorRT.*")
os.environ['ORT_PROVIDERS'] = 'CUDAExecutionProvider,CPUExecutionProvider'
os.environ['ORT_TENSORRT_UNAVAILABLE'] = '1'

# Docling v2正确的导入方式
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.base_models import InputFormat, DocumentStream
from docling_core.types.doc import TextItem, TableItem, PictureItem

logging.getLogger("docling").setLevel(logging.WARNING)


class PdfParserVLM:
    """
    Docling v2原生PDF解析器 - 基于DocumentConverter的正确实现
    
    核心特性:
    - 使用Docling v2的DocumentConverter进行文档结构解析
    - 基于DoclingDocument的TextItem/TableItem/PictureItem进行内容类型判断
    - 保持正确的文档阅读顺序：按页面分组，页面内严格按序处理
    - 图像使用外部VLM模型进行语义分析
    - 高精度表格检测和OCR支持
    - 智能并行优化：页面内VLM并行处理，保持内容顺序
    - 与现有RAG系统完全兼容
    
    处理策略:
    - TextItem: 直接提取文本内容，应用后处理
    - TableItem: 提取表格结构，转换为Markdown/HTML格式
    - PictureItem: 提取图像数据并使用外部VLM进行语义描述
    - 保持与unstructured版本相同的接口和返回格式
    
    VLM集成:
    - 支持外部VLM模型（如OllamaCV）
    - 仅对图像进行VLM语义分析
    - 批处理和并行处理优化
    - 智能缓存和去重
    """
    
    def __init__(self, 
                 vlm_mdl=None,
                 enable_vlm=True,
                 image_batch_size=5, 
                 enable_gc=True, 
                 max_workers=4, 
                 page_chunk_size=10):
        """
        初始化解析器
        
        Args:
            vlm_mdl: 外部VLM模型实例（如OllamaCV），用于图像语义分析
            enable_vlm: 是否启用VLM处理图像
            image_batch_size: 图片批处理大小
            enable_gc: 是否启用强制垃圾回收
            max_workers: 并行处理线程数
            page_chunk_size: 页面分块大小（保持兼容性）
        """
        self.vlm_mdl = vlm_mdl
        self.enable_vlm = enable_vlm and vlm_mdl is not None
        self.image_batch_size = image_batch_size
        self.enable_gc = enable_gc
        self.max_workers = max_workers
        self.page_chunk_size = page_chunk_size
        
        # 性能优化配置
        self.skip_small_images = True
        self.skip_decorative_images = True
        self.min_image_area_ratio = 0.06
        self.max_similar_images = 5
        
        # VLM结果缓存
        self._vlm_cache = {}
        self._cache_size_limit = 1000
        
        # 临时文件管理
        self.temp_dir = None

        # 临时文件管理
        self.docling_models_dir = "/media/zzg/GJ_disk01/pretrained_model/docling/models"
        
        # 初始化DocumentConverter
        self._setup_document_converter()
        
        logging.info(f"DoclingNativeVLM v2初始化完成: VLM={'启用' if self.enable_vlm else '禁用'}, "
                    f"VLM模型={'已配置' if vlm_mdl else '未配置'}")

    def _setup_document_converter(self):
        """
        设置优化的DocumentConverter配置
        
        配置项：
        - 启用高精度表格检测（TableFormerMode.ACCURATE）
        - 启用OCR支持（适用于扫描PDF）
        - 启用布局分析（保持阅读顺序）
        - 优化表格单元格匹配
        """
        try:
            # 配置高精度PDF处理选项
            pipeline_options = PdfPipelineOptions(
                do_ocr=True,  # 启用OCR支持扫描PDF
                do_table_structure=True,  # 启用高精度表格检测
                do_figure=True,  # 启用图形检测
                do_paragraph_segmentation=True,  # 启用段落分割
                do_layout_structure=True,  # 启用布局结构分析
                artifacts_path=self.docling_models_dir,
                generate_picture_images=True,
                images_scale=1.0,
            )
            
            # 设置TableFormer为高精度模式
            pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
            pipeline_options.table_structure_options.do_cell_matching = True

            # 创建DocumentConverter实例
            self.converter = DocumentConverter(
                allowed_formats=[InputFormat.PDF],  # 只处理PDF
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                },
            )
            
            logging.info("DocumentConverter配置完成: 高精度表格检测 + OCR支持 + 布局分析")
            
        except Exception as e:
            logging.error(f"DocumentConverter初始化失败: {e}")
            # 回退到默认配置
            self.converter = DocumentConverter()
            logging.warning("使用默认DocumentConverter配置")

    def _setup_temp_directory(self):
        """设置临时目录用于图像处理"""
        if self.temp_dir is None:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="docling_vlm_"))
            logging.debug(f"创建临时目录: {self.temp_dir}")

    def _cleanup_temp_directory(self):
        """清理临时目录"""
        if self.temp_dir and self.temp_dir.exists():
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
                logging.debug(f"清理临时目录: {self.temp_dir}")
            except Exception as e:
                logging.warning(f"清理临时目录失败: {e}")
            finally:
                self.temp_dir = None

    def _describe_image_with_external_vlm(self, image_obj):
        """
        使用外部VLM模型对图像进行语义描述
        
        保持与原有VLM调用接口的兼容性
        
        Args:
            image_obj: 图像对象（PIL Image或字节数据）
            
        Returns:
            str: VLM解析的图像描述
        """
        if not self.vlm_mdl:
            return "VLM模型未配置，无法解析图像内容"
        
        try:
            # 调用外部VLM模型的describe方法
            vlm_result = self.vlm_mdl.describe(image_obj)
            
            # 处理VLM返回值的不同格式
            if isinstance(vlm_result, tuple):
                if len(vlm_result) >= 2:
                    description, _ = vlm_result  # (description, token_count)
                else:
                    description = vlm_result[0]  # 只有描述
            else:
                description = vlm_result  # 直接是描述字符串
                
            return description.strip() if description else "VLM解析结果为空"
            
        except Exception as e:
            logging.error(f"外部VLM解析失败: {e}")
            return f"VLM解析失败: {e}"

    def _get_cached_vlm_result(self, image_hash: str) -> Optional[str]:
        """从缓存获取VLM结果"""
        return self._vlm_cache.get(image_hash)

    def _cache_vlm_result(self, image_hash: str, description: str):
        """缓存VLM结果，实现LRU策略"""
        if len(self._vlm_cache) >= self._cache_size_limit:
            # 简单LRU：删除最旧的一半
            keys_to_remove = list(self._vlm_cache.keys())[:self._cache_size_limit // 2]
            for key in keys_to_remove:
                del self._vlm_cache[key]
        
        self._vlm_cache[image_hash] = description

    def restore_spaces_and_punct(self, text: str) -> str:
        """
        轻量文本后处理
        
        保持与原有处理逻辑一致
        """
        if not text:
            return text
            
        # 1. 规范空白
        t = re.sub(r'\s+', ' ', text).strip()

        # 2. 确保标点后有空格
        t = re.sub(r'(?<=[\.,;:])(?=[^\s])', r' ', t)

        # 3. 在小写/数字与大写之间插入空格
        t = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', t)

        # 4. 在字母与数字之间插入空格
        t = re.sub(r'(?<=[A-Za-z])(?=[0-9])', ' ', t)
        t = re.sub(r'(?<=[0-9])(?=[A-Za-z])', ' ', t)

        # 5. 再次折叠重复空格
        t = re.sub(r'\s{2,}', ' ', t).strip()

        return t

    def _header_footer_text_filter(self, text: str) -> bool:
        """
        判断文本是否为页眉/页脚/页码等需要过滤的内容
        """
        if not text or len(text.strip()) <= 4:
            return True

        # 常见页眉页脚关键词
        keywords = [r'page\s*\d+', 'copyright', '©', 'confidential']
        for kw in keywords:
            if re.search(kw, text, re.IGNORECASE):
                return True
                
        return False

    def _line_tag(self, bbox_info, ZM=3):
        """
        生成位置标签，保持与原有格式兼容
        
        Args:
            bbox_info: 包含位置信息的字典
            ZM: 缩放比例
            
        Returns:
            str: 位置标签字符串
        """
        try:
            page_num = bbox_info.get("page_number", 1) - 1  # 转换为0基索引
            tag = "@@{}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}##".format(
                page_num,
                bbox_info.get("x0", 0), bbox_info.get("x1", 0), 
                bbox_info.get("top", 0), bbox_info.get("bottom", 0)
            )
            return tag
        except Exception as e:
            logging.debug(f"生成位置标签失败: {e}")
            return "@@0\t0.0\t0.0\t0.0\t0.0##"

    def _extract_position_from_item(self, item) -> Tuple[int, float, float, float, float]:
        """
        从DoclingDocument元素中提取位置信息
        
        Args:
            item: TextItem/TableItem/PictureItem等
            
        Returns:
            tuple: (page_number, x0, x1, top, bottom) - 与unstructured格式兼容
        """
        page_number = 0  # 0基索引，与unstructured保持一致
        x0, x1, top, bottom = 0.0, 100.0, 0.0, 100.0
        
        try:
            # 尝试从item的prov属性中提取位置信息
            if hasattr(item, 'prov') and item.prov:
                for prov in item.prov:
                    # 提取页码
                    if hasattr(prov, 'page_no'):
                        page_number = prov.page_no - 1  # 转换为0基索引
                    
                    # 提取边界框
                    if hasattr(prov, 'bbox'):
                        bbox = prov.bbox
                        if hasattr(bbox, 'l') and hasattr(bbox, 'r'):
                            x0 = bbox.l
                            x1 = bbox.r  
                            top = bbox.t
                            bottom = bbox.b
                        break
                        
        except Exception as e:
            logging.debug(f"提取位置信息失败: {e}")
            
        return (page_number, x0, x1, top, bottom)

    def _process_text_item(self, item: TextItem, sections: List[Tuple[str, str]]):
        """
        处理TextItem，提取文本内容并应用后处理
        
        Args:
            item: DoclingDocument中的TextItem
            sections: 输出的文本段落列表
        """
        try:
            if hasattr(item, 'text') and item.text:
                text_content = item.text.strip()
                
                if text_content and len(text_content) > 2:
                    # 过滤页眉页脚
                    if self._header_footer_text_filter(text_content):
                        logging.debug(f"跳过页眉页脚文本: '{text_content[:30]}...'")
                        return
                    
                    # 应用文本后处理
                    text_clean = self.restore_spaces_and_punct(text_content)
                    
                    if text_clean.strip():
                        # 提取位置信息 - 现在返回元组格式
                        position_tuple = self._extract_position_from_item(item)
                        # 为了兼容现有的_line_tag，转换为字典格式
                        bbox_info = {
                            "page_number": position_tuple[0] + 1,  # 转换回1基索引给_line_tag使用
                            "x0": position_tuple[1],
                            "x1": position_tuple[2], 
                            "top": position_tuple[3],
                            "bottom": position_tuple[4]
                        }
                        position_tag = self._line_tag(bbox_info)
                        
                        # 添加到sections
                        sections.append((text_clean.strip(), position_tag))
                        logging.debug(f"处理文本: '{text_clean[:50]}...'")
                        
        except Exception as e:
            logging.warning(f"处理TextItem失败: {e}")

    # def _process_table_item(self, item: TableItem, tbls: List[List[str]]):
    def _process_table_item(self, item: TableItem, sections: List[Tuple[str, str]]):
        """
        处理TableItem，提取表格结构并转换格式
        
        Args:
            item: DoclingDocument中的TableItem
            tbls: 输出的表格列表
        """
        try:
            table_content = None
            
            # 尝试多种表格内容提取方式
            try:
                # 方法1: 导出为DataFrame然后转换为Markdown
                if hasattr(item, 'export_to_dataframe'):
                    df = item.export_to_dataframe()
                    if df is not None and not df.empty:
                        # table_content = df.to_markdown(index=False)
                        table_content = df.to_html(index=False)
                        logging.debug("使用export_to_dataframe提取表格")
            except Exception as df_err:
                logging.debug(f"DataFrame导出失败: {df_err}")
            
            if not table_content:
                try:
                    # 方法2: 导出为HTML
                    if hasattr(item, 'export_to_html'):
                        html_content = item.export_to_html()
                        if html_content and html_content.strip():
                            table_content = html_content.strip()
                            logging.debug("使用export_to_html提取表格")
                except Exception as html_err:
                    logging.debug(f"HTML导出失败: {html_err}")
            
            if table_content and table_content.strip():
                # 提取位置信息 - 现在返回元组格式
                position_tuple = self._extract_position_from_item(item)
                
                # 按照unstructured格式：((img, content), [position_tuples])
                # tbls.append(((None, table_content.strip()), [position_tuple]))
                sections.append((table_content.strip(), position_tuple))
                logging.debug(f"处理表格: {table_content[:50]}...")
            
        except Exception as e:
            logging.warning(f"处理TableItem失败: {e}")

    def _extract_image_from_picture_item(self, doc, item: PictureItem):
        """
        从PictureItem中提取图像数据
        
        Args:
            doc: DoclingDocument对象
            item: DoclingDocument中的PictureItem
            
        Returns:
            图像数据（PIL Image），如果提取失败返回None
        """
        try:
            attr_value = item.get_image(doc)
            # 保存图片
            # if attr_value:
                # self.save_pil_image(attr_value, out_path)
            return attr_value
        except Exception as e:
            logging.error(f"从PictureItem提取图像失败: {e}")
            return None
            
    def save_pil_image(self, img: Image.Image, out_path: Path=Path("/home/zzg/workspace/pycharm/RAG_Fin/figures/")):
        out_path.mkdir(parents=True, exist_ok=True)
        out_path = out_path / f"{time.time()}.png"
        img.save(out_path)

    def _get_page_size_from_docling_item(self, item) -> Tuple[Optional[float], Optional[float]]:
        """
        尝试从DoclingDocument元素中获取页面尺寸信息
        
        Args:
            item: DoclingDocument中的元素
            
        Returns:
            tuple: (page_width, page_height) 或 (None, None)
        """
        try:
            # 尝试从item的prov属性中获取页面尺寸信息
            if hasattr(item, 'prov') and item.prov:
                for prov in item.prov:
                    if hasattr(prov, 'page_no'):
                        # 这里可以尝试获取页面尺寸，但docling的API可能不直接提供
                        # 暂时返回None，使用默认值
                        pass
                        
            # 如果无法从docling获取，返回None使用默认值
            return None, None
            
        except Exception as e:
            logging.debug(f"从docling元素获取页面尺寸失败: {e}")
            return None, None

    def _should_skip_image_by_size(self, pil_img: Image.Image, item: PictureItem) -> tuple:
        """
        根据图像尺寸和面积判断是否应该跳过图像处理
        
        参考unstructured版本的过滤逻辑：
        - 绝对像素阈值：宽度<200或高度<150
        - 面积比阈值：图像面积<页面面积的6%
        
        Args:
            pil_img: PIL图像对象
            item: PictureItem元素
            
        Returns:
            tuple: (should_skip: bool, reason: str)
        """
        try:
            w, h = pil_img.size
            
            # 规则1: 绝对像素阈值
            if w < 200 or h < 150:
                return True, f"图像尺寸过小({w}x{h})"
            
            # 规则1b: 过滤装饰性小图标
            if w < 100 and h < 100:
                return True, "疑似装饰性小图标"
                
            # 规则1c: 过滤细长的线条或分隔符
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio > 10 and min(w, h) < 20:
                return True, "疑似线条或分隔符"
            
            # 规则2: 面积比阈值（需要页面尺寸信息）
            try:
                # 尝试从item的位置信息中获取页面尺寸
                page_w, page_h = self._get_page_size_from_docling_item(item)
                
                # 如果无法获取页面尺寸，使用标准A4尺寸
                if not page_w or not page_h:
                    page_w, page_h = 612.0, 792.0  # 标准A4页面尺寸（点）
                
                img_area = float(w) * float(h)
                page_area = page_w * page_h
                
                if page_area > 0 and (img_area < 0.06 * page_area):
                    area_ratio = img_area / page_area
                    return True, f"图像面积比例过小({area_ratio:.2%})"
                    
            except Exception as e:
                logging.debug(f"计算面积比例失败: {e}")
            
            return False, "需要处理"
            
        except Exception as e:
            logging.warning(f"图像尺寸检查失败: {e}")
            return False, "检查失败，保守处理"

    def _process_picture_item(self, doc, item: PictureItem, sections: List[Tuple[str, str]]):
        """
        处理PictureItem，使用VLM进行语义分析
        
        Args:
            doc: DoclingDocument对象
            item: DoclingDocument中的PictureItem
            sections: 输出的文本段落列表
        """
        if not self.enable_vlm or not self.vlm_mdl:
            print("VLM未启用，跳过图像处理")
            return
        
        try:
            # 提取图像数据
            image_data = self._extract_image_from_picture_item(doc, item)
            if not image_data:
                print("无法从PictureItem提取图像数据")
                return
            
            # 检查图像是否为PIL Image对象
            if isinstance(image_data, Image.Image):
                # 根据尺寸和面积过滤图像
                should_skip, skip_reason = self._should_skip_image_by_size(image_data, item)
                if should_skip:
                    logging.debug(f"跳过图像处理: {skip_reason}")
                    return
                
            # 生成图像哈希用于去重和缓存
            if isinstance(image_data, bytes):
                image_hash = hashlib.md5(image_data).hexdigest()
            else:
                # 对于PIL Image或其他类型，转换为字符串进行哈希
                image_hash = hashlib.md5(str(image_data).encode()).hexdigest()
            
            # 检查缓存
            cached_description = self._get_cached_vlm_result(image_hash)
            if cached_description:
                logging.debug(f"使用缓存的VLM结果: {image_hash[:8]}...")
                description = cached_description
            else:
                # 使用VLM进行语义分析
                start_time_vlm = time.time()
                description = self._describe_image_with_external_vlm(image_data)
                end_time_vlm = time.time()
                print(f"574行：self._describe_image_with_external_vlm完成，耗时: {start_time_vlm - end_time_vlm:.2f}s, description:{description[:50]}")
                # 缓存结果
                if description and not description.startswith("VLM解析失败"):
                    self._cache_vlm_result(image_hash, description)
            
            if description and description.strip() and not description.startswith("VLM解析失败"):
                # 提取位置信息 - 现在返回元组格式
                position_tuple = self._extract_position_from_item(item)
                # 为了兼容现有的_line_tag，转换为字典格式
                bbox_info = {
                    "page_number": position_tuple[0] + 1,  # 转换回1基索引给_line_tag使用
                    "x0": position_tuple[1],
                    "x1": position_tuple[2], 
                    "top": position_tuple[3],
                    "bottom": position_tuple[4]
                }
                position_tag = self._line_tag(bbox_info)
                
                # 添加到sections
                sections.append((description.strip(), position_tag))
                logging.debug(f"VLM处理图像: {description[:50]}...")
            else:
                logging.debug(f"VLM解析失败或结果为空: {description}")
                    
        except Exception as e:
            logging.warning(f"处理PictureItem失败: {e}")

    def _group_items_by_page(self, items_with_levels):
        """
        按页面分组元素，同时保持每页内的正确顺序
        
        Args:
            items_with_levels: 从iterate_items()返回的(item, level)列表
            
        Returns:
            dict: {page_number: [(item, level, item_index)]} 按页面分组的元素
        """
        page_groups = {}
        
        for item_index, (item, level) in enumerate(items_with_levels):
            # 获取元素所在页码
            position_tuple = self._extract_position_from_item(item)
            page_number = position_tuple[0]  # 0基索引页码
            
            if page_number not in page_groups:
                page_groups[page_number] = []
            
            # 保存原始索引以维持顺序
            page_groups[page_number].append((item, level, item_index))
        
        # 确保每页内的元素按原始索引排序
        # for page_number in page_groups:
        #     page_groups[page_number].sort(key=lambda x: x[2])  # 按item_index排序
        
        return page_groups

    def _process_page_items(self, doc, page_items, sections, tbls):
        """
        按正确顺序处理单个页面的所有元素
        
        重要：确保页面内容按照在PDF中的出现顺序加入到sections和tbls中
        
        Args:
            page_items: 单个页面的元素列表 [(item, level, item_index)]
            sections: 输出文本段落列表
            tbls: 输出表格列表
        """
        # 严格按顺序处理每个元素
        for item, level, item_index in page_items:
            if isinstance(item, TextItem):
                self._process_text_item(item, sections)
            elif isinstance(item, TableItem):
                # self._process_table_item(item, tbls)
                self._process_table_item(item, sections)
            elif isinstance(item, PictureItem) and self.enable_vlm:
                self._process_picture_item(doc, item, sections)
        return

    def _process_pages(self, doc, items_with_levels, sections, tbls):
        """
        按页面分组处理元素，保持页面内的正确顺序
        
        Args:
            items_with_levels: 从iterate_items()返回的(item, level)列表
            sections: 输出文本段落列表
            tbls: 输出表格列表
        """
        # 按页面分组元素
        page_groups = self._group_items_by_page(items_with_levels)
        
        logging.info(f"文档分为 {len(page_groups)} 页进行处理")
        
        # 按页码顺序处理每一页
        for page_number in sorted(page_groups.keys()):
            start_time_page = time.time()
            page_items = page_groups[page_number]
            logging.debug(f"处理第 {page_number + 1} 页，包含 {len(page_items)} 个元素")
            # 按正确顺序处理页面内的所有元素
            self._process_page_items(doc, page_items, sections, tbls)
            end_time_page = time.time()
            print(f"第 {page_number + 1} 页处理完成，耗时: {end_time_page - start_time_page:.2f}s")

    def __call__(self, filename, need_image=True):
        """
        解析PDF文件 - 使用Docling v2 DocumentConverter
        
        Args:
            filename: PDF文件路径或字节流
            need_image: 是否需要处理图片（启用VLM处理）
            
        Returns:
            tuple: (sections, tbls) 与原格式完全兼容
                sections: List[Tuple[str, str]] - (文本内容, 位置标签)
                tbls: List[List[str]] - [[表格内容, 位置标签]]
        """
        sections = []
        tbls = []
        
        # 设置临时目录
        if need_image and self.enable_vlm:
            self._setup_temp_directory()
        
        try:
            # 动态调整VLM设置
            if need_image != self.enable_vlm:
                logging.info(f"动态调整VLM设置: {self.enable_vlm} -> {need_image}")
                self.enable_vlm = need_image
            start_time = time.time()
            
            # 使用DocumentConverter进行转换
            if isinstance(filename, str):
                # 文件路径
                logging.info(f"转换PDF文件: {filename}")
                result = self.converter.convert(filename)
            else:
                # 字节流
                logging.info("转换PDF字节流")
                buf = BytesIO(filename)
                source = DocumentStream(name="document.pdf", stream=buf)
                result = self.converter.convert(source)

            end_time_convert = time.time()
            print(f"716行：self.converter.convert完成，耗时: {end_time_convert - start_time:.2f}s")
            # 获取DoclingDocument
            doc = result.document
            logging.info(f"DocumentConverter转换完成，耗时: {time.time() - start_time:.2f}s")
            
            # 通过iterate_items()按正确阅读顺序获取所有元素
            items_with_levels = list(doc.iterate_items())
            logging.info(f"文档包含 {len(items_with_levels)} 个元素")
            
            # 统计元素类型
            text_count = sum(1 for item, _ in items_with_levels if isinstance(item, TextItem))
            table_count = sum(1 for item, _ in items_with_levels if isinstance(item, TableItem))
            picture_count = sum(1 for item, _ in items_with_levels if isinstance(item, PictureItem))
            print(f"元素统计: {text_count} 个文本, {table_count} 个表格, {picture_count} 个图像")
            
            # 按页面顺序处理所有元素（保持正确的阅读顺序）
            self._process_pages(doc, items_with_levels, sections, tbls)
            end_time_process_items = time.time()
            print(f"735行：self._process_pages所有页处理完成，耗时: {end_time_process_items - end_time_convert:.2f}s")
            # 后处理：过滤和格式化
            sections_filtered = []
            for item in sections:
                if isinstance(item, tuple) and len(item) == 2:
                    text, position_tag = item
                    text_str = str(text).strip() if text else ""
                    tag_str = str(position_tag) if position_tag else ""
                    if text_str and len(text_str) > 2:
                        sections_filtered.append((text_str + '\n', tag_str))
            
            logging.info(f"解析完成: {len(sections_filtered)} 个文本段落, {len(tbls)} 个表格")
            
            # 内存清理
            if self.enable_gc:
                gc.collect()
            
            return sections_filtered, tbls
            
        except Exception as e:
            logging.error(f"PDF解析失败: {str(e)}")
            import traceback
            logging.error(f"详细错误信息: {traceback.format_exc()}")
            return [("PDF解析失败", "")], []
            
        finally:
            # 清理临时目录
            if need_image and self.enable_vlm:
                self._cleanup_temp_directory()
    
    def crop(self, text, ZM=3, need_position=False):
        """保持兼容性的crop方法"""
        if need_position:
            return None, []
        return None
    
    def remove_tag(self, txt):
        """保持兼容性的标签移除方法"""
        return re.sub(r"@@[\t0-9.-]+?##", "", txt)

class PlainParser(object):
    """简单PDF解析器 - 基于Docling的纯文本提取（无VLM）"""
    
    def __call__(self, filename, from_page=0, to_page=100000, **kwargs):
        try:
            # 使用Docling进行简单文本提取（禁用VLM）
            parser = PdfParserVLM(enable_vlm=False)
            result = parser(filename, need_image=False)
            
            # 提取文本内容，忽略位置标签
            lines = []
            sections, _ = result
            for section, _ in sections:
                if section and section.strip():
                    lines.append(section.strip())
                    
            return [(line, "") for line in lines], []
            
        except Exception as e:
            logging.error(f"PlainParser解析失败: {str(e)}")
            return [("解析失败", "")], []
    
    def crop(self, ck, need_position):
        """PlainParser兼容性方法"""
        raise NotImplementedError
    
    @staticmethod
    def remove_tag(txt):
        return txt


if __name__ == "__main__":
    # 测试代码
    import sys
    
    print("=== Docling v2原生PDF解析器测试 ===")
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        print(f"解析文件: {test_file}")
        
        try:
            # 测试无VLM模式
            print("\n1. 测试无VLM模式...")
            parser_no_vlm = PdfParserVLM(enable_vlm=False)
            sections, tbls = parser_no_vlm(test_file, need_image=False)
            print(f"解析结果: {len(sections)} 个段落, {len(tbls)} 个表格")
            
            # 显示前几个段落
            for i, (content, tag) in enumerate(sections[:3]):
                print(f"\n段落 {i+1}: {content[:100]}...")
                print(f"位置标签: {tag}")
            
            # 显示表格信息
            if tbls:
                print(f"\n表格示例: {tbls[0][0][:100]}...")
            
            print("\n2. VLM模式测试需要VLM模型:")
            print("from rag.llm.cv_model import OllamaCV")
            print("vlm = OllamaCV('', 'blaifa/InternVL3_5:8b', base_url='http://localhost:11434')")
            print("parser_vlm = PdfParserVLM(vlm_mdl=vlm, enable_vlm=True)")
            print("sections, tbls = parser_vlm(test_file, need_image=True)")
            
        except Exception as e:
            print(f"解析失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("用法: python pdf_parser_docling_native_vlm.py <pdf文件路径>")
        print("\nDocling v2特性:")
        print("- 基于DocumentConverter的正确API使用")
        print("- TextItem/TableItem/PictureItem内容类型判断")
        print("- 高精度表格检测 + OCR支持")
        print("- 正确的文档阅读顺序")
        print("- VLM图像语义分析集成")
        print("- 并行处理和性能优化")