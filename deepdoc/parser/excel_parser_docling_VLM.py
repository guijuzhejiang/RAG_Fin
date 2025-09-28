"""
excel_parser_docling_VLM.py

基于Docling v2的Excel (.xls/.xlsx) 解析器，集成Ollama VLM语义分析
功能特点：
- 支持XLS自动转换为XLSX（通过LibreOffice）
- 基于docling v2 DocumentConverter的高精度解析
- 按doc.groups顺序处理多个Sheet，保持正确的阅读顺序
- 智能识别Text、Table、Picture，图片使用VLM语义分析
- 图像过滤机制，跳过装饰性小图
- 支持并行VLM处理优化性能

依赖：docling, docling_core, pillow, pandas, requests, libreoffice
"""

import os
import logging
import time
import tempfile
import subprocess
import hashlib
from pathlib import Path
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

import pandas as pd
from PIL import Image

# Docling v2 imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat, DocumentStream
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode

# docling_core types
try:
    from docling_core.types.doc import TextItem, TableItem, PictureItem
except ImportError:
    TextItem = None
    TableItem = None
    PictureItem = None

# Ollama optional
try:
    import ollama
except ImportError:
    ollama = None

logging.getLogger("docling").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class ExcelParserVLM:
    """
    基于Docling v2的Excel解析器，集成VLM语义分析
    
    功能特点:
    - 支持XLS转XLSX自动转换
    - 按doc.groups顺序处理多个Sheet
    - 智能识别Text、Table、Picture
    - VLM图像语义分析与缓存
    - 图像过滤机制
    - 并行处理优化
    """
    
    def __init__(self, 
                 vlm_mdl: Optional[Any] = None,
                 enable_vlm: bool = True,
                 docling_models_dir: str = "/media/zzg/GJ_disk01/pretrained_model/docling/models",
                 max_workers: int = 4,
                 min_image_area: int = 30000,  # 200x150像素
                 temp_image_dir: Optional[str] = None):
        """
        初始化Excel解析器
        
        Args:
            vlm_mdl: VLM模型实例（支持Ollama等）
            enable_vlm: 是否启用VLM图像分析
            docling_models_dir: Docling模型目录
            max_workers: 并行处理线程数
            min_image_area: 最小图像面积阈值
            temp_image_dir: 临时图像保存目录
        """
        self.vlm_mdl = vlm_mdl
        self.enable_vlm = bool(enable_vlm and vlm_mdl)
        self.docling_models_dir = Path(docling_models_dir)
        self.max_workers = max_workers
        self.min_image_area = min_image_area
        self.temp_image_dir = Path(temp_image_dir) if temp_image_dir else None
        
        # VLM缓存机制
        self._vlm_cache: Dict[str, str] = {}
        self._cache_size_limit = 1000
        
        # 初始化DocumentConverter
        self._setup_document_converter()

    def _setup_document_converter(self):
        """
        初始化Docling DocumentConverter，配置Excel解析选项
        """
        try:
            # 创建临时目录确保存在
            self.docling_models_dir.mkdir(parents=True, exist_ok=True)
            
            # Excel解析使用默认配置，主要依赖SimplePipeline
            self.converter = DocumentConverter(
                allowed_formats=[InputFormat.XLSX]  # 只支持XLSX格式
            )
            
            logger.info("DocumentConverter初始化完成，支持XLSX格式")
            
        except Exception as e:
            logger.error(f"DocumentConverter初始化失败: {e}")
            # 回退到基础配置
            self.converter = DocumentConverter()

    # ===== XLS转换工具 =====
    def _convert_xls_to_xlsx(self, xls_path: str, output_dir: Optional[str] = None) -> str:
        """
        使用LibreOffice将XLS转换为XLSX
        
        Args:
            xls_path: XLS文件路径
            output_dir: 输出目录，默认为输入文件同目录
            
        Returns:
            str: 转换后的XLSX文件路径
        """
        try:
            xls_file = Path(xls_path)
            if output_dir is None:
                output_dir = str(xls_file.parent)
            
            # 执行LibreOffice转换
            subprocess.run([
                "soffice", "--headless", "--convert-to", "xlsx", 
                "--outdir", output_dir, str(xls_file)
            ], check=True, timeout=300)
            
            xlsx_path = Path(output_dir) / (xls_file.stem + ".xlsx")
            if not xlsx_path.exists():
                raise FileNotFoundError(f"转换后的XLSX文件不存在: {xlsx_path}")
                
            logger.info(f"XLS转换成功: {xls_path} -> {xlsx_path}")
            return str(xlsx_path)
            
        except Exception as e:
            logger.error(f"XLS转XLSX转换失败: {e}")
            raise

    # ===== 图像处理工具 =====
    def _extract_image_from_picture_item(self, doc, item) -> Optional[Image.Image]:
        """
        从PictureItem中提取PIL图像
        
        Args:
            doc: DoclingDocument对象
            item: PictureItem对象
            
        Returns:
            PIL.Image对象或None
        """
        try:
            # 优先使用get_image方法
            if hasattr(item, "get_image"):
                img = item.get_image(doc)
                if img is not None and isinstance(img, Image.Image):
                    return img.convert("RGB")
                    
        except Exception as e:
            logger.debug(f"使用get_image提取图像失败: {e}")

        # 回退：尝试从item的其他属性获取图像
        try:
            img_ref = getattr(item, "image", None)
            if not img_ref:
                return None
                
            # 尝试从路径加载
            uri = getattr(img_ref, "uri", None) or getattr(img_ref, "path", None)
            if uri and isinstance(uri, str):
                img_path = Path(uri)
                if img_path.exists():
                    return Image.open(str(img_path)).convert("RGB")
            
            # 尝试从二进制数据加载
            data = getattr(img_ref, "data", None) or getattr(img_ref, "bytes", None)
            if data:
                if isinstance(data, str):
                    import base64
                    data = base64.b64decode(data)
                return Image.open(BytesIO(data)).convert("RGB")
                
        except Exception as e:
            logger.debug(f"回退方法提取图像失败: {e}")
            
        return None

    def _should_skip_image_by_size(self, pil_img: Image.Image) -> Tuple[bool, str]:
        """
        根据图像尺寸判断是否应该跳过
        
        Args:
            pil_img: PIL图像对象
            
        Returns:
            tuple: (should_skip, reason)
        """
        if pil_img is None:
            return True, "无图像数据"
            
        w, h = pil_img.size
        
        # 规则1: 绝对像素阈值
        if w < 200 or h < 150:
            return True, f"图像尺寸过小({w}x{h})"
            
        # 规则2: 面积阈值
        area = w * h
        if area < self.min_image_area:
            return True, f"图像面积过小({area}像素)"
            
        # 规则3: 装饰性小图标
        if w < 64 and h < 64:
            return True, "疑似装饰性图标"
            
        # 规则4: 细长的分隔线
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio > 10 and min(w, h) < 20:
            return True, "疑似分隔线或边框"
            
        return False, "需要处理"

    # ===== VLM缓存机制 =====
    def _compute_image_hash(self, pil_img: Image.Image) -> str:
        """计算图像哈希用于缓存"""
        try:
            # 使用图像数据的MD5作为缓存键
            img_bytes = BytesIO()
            pil_img.save(img_bytes, format='PNG')
            return hashlib.md5(img_bytes.getvalue()).hexdigest()
        except Exception:
            # 回退到基于尺寸的简单哈希
            return hashlib.md5(f"{pil_img.size}".encode()).hexdigest()
    
    def _get_cached_vlm_result(self, image_hash: str) -> Optional[str]:
        """获取缓存的VLM结果"""
        return self._vlm_cache.get(image_hash)
    
    def _cache_vlm_result(self, image_hash: str, description: str):
        """缓存VLM结果"""
        if len(self._vlm_cache) >= self._cache_size_limit:
            # 简单的LRU：移除最旧的一半
            keys_to_remove = list(self._vlm_cache.keys())[:len(self._vlm_cache)//2]
            for key in keys_to_remove:
                del self._vlm_cache[key]
        
        self._vlm_cache[image_hash] = description

    def _describe_image_with_vlm(self, pil_img: Image.Image) -> str:
        """
        使用VLM分析图像语义
        
        Args:
            pil_img: PIL图像对象
            
        Returns:
            图像描述文本
        """
        if not self.vlm_mdl:
            return "[VLM未配置]"
            
        try:
            # 检查缓存
            image_hash = self._compute_image_hash(pil_img)
            cached_result = self._get_cached_vlm_result(image_hash)
            if cached_result:
                logger.debug(f"使用缓存的VLM结果: {image_hash[:8]}...")
                return cached_result
            
            # 调用VLM
            description = self.vlm_mdl.describe(pil_img)
                
            # 缓存结果
            self._cache_vlm_result(image_hash, description)
            
            return description
            
        except Exception as e:
            logger.exception(f"VLM图像分析失败: {e}")
            return f"[VLM分析失败: {e}]"

    # ===== 内容项处理方法 =====
    def _get_sheet_name_from_item(self, item) -> str:
        """
        从Docling元素中提取Sheet名称
        """
        try:
            # 检查常见的sheet名称属性
            for attr in ("sheet_name", "sheet", "sheet_no", "sheet_index"):
                if hasattr(item, attr):
                    value = getattr(item, attr)
                    if value is not None:
                        return str(value)
            
            # 检查prov属性中的sheet信息
            if hasattr(item, "prov") and item.prov:
                for prov in item.prov:
                    if isinstance(prov, dict):
                        for key in ("sheet_name", "sheet"):
                            if key in prov and prov[key] is not None:
                                return str(prov[key])
                    elif hasattr(prov, "sheet_name"):
                        sheet_name = getattr(prov, "sheet_name")
                        if sheet_name:
                            return str(sheet_name)
                            
        except Exception as e:
            logger.debug(f"提取sheet名称失败: {e}")
            
        return "Sheet1"  # 默认名称

    def _process_text_item(self, item, doc, sheet_name: str) -> Dict[str, Any]:
        """
        处理TextItem元素
        
        Args:
            item: TextItem对象
            doc: DoclingDocument对象
            sheet_name: Sheet名称
            
        Returns:
            处理结果字典
        """
        try:
            text_content = ""
            
            # 尝试多种方式获取文本
            if hasattr(item, "get_text"):
                try:
                    text_content = item.get_text(doc)
                except Exception:
                    text_content = getattr(item, "text", "")
            else:
                text_content = getattr(item, "text", str(item))
            
            # 清理文本
            text_content = str(text_content).strip()
            
            return {
                "type": "text",
                "content": text_content,
                "sheet": sheet_name
            }
            
        except Exception as e:
            logger.warning(f"处理TextItem失败: {e}")
            return {
                "type": "text", 
                "content": "",
                "sheet": sheet_name,
                "error": str(e)
            }

    def _process_table_item(self, item, doc, sheet_name: str) -> Dict[str, Any]:
        """
        处理TableItem元素
        
        Args:
            item: TableItem对象
            doc: DoclingDocument对象
            sheet_name: Sheet名称
            
        Returns:
            处理结果字典
        """
        try:
            df = None
            markdown_content = ""
            
            # 尝试导出为DataFrame
            if hasattr(item, "export_to_dataframe"):
                try:
                    df = item.export_to_dataframe(doc)
                except Exception as e:
                    logger.debug(f"导出DataFrame失败: {e}")
            
            # 生成Markdown格式
            if df is not None and not df.empty:
                try:
                    markdown_content = df.to_markdown(index=False)
                except Exception:
                    # 回退到CSV格式
                    markdown_content = df.to_csv(index=False)
            else:
                # 尝试直接导出Markdown
                if hasattr(item, "export_to_markdown"):
                    try:
                        markdown_content = item.export_to_markdown(doc)
                    except Exception:
                        markdown_content = ""
            
            return {
                "type": "table",
                "content": markdown_content.strip(),
                "dataframe": df,
                "sheet": sheet_name
            }
            
        except Exception as e:
            logger.warning(f"处理TableItem失败: {e}")
            return {
                "type": "table",
                "content": "",
                "dataframe": None,
                "sheet": sheet_name,
                "error": str(e)
            }

    def _process_picture_item(self, item, doc, sheet_name: str, need_image: bool = True) -> Dict[str, Any]:
        """
        处理PictureItem元素
        
        Args:
            item: PictureItem对象
            doc: DoclingDocument对象
            sheet_name: Sheet名称
            need_image: 是否需要处理图像
            
        Returns:
            处理结果字典
        """
        try:
            if not need_image:
                return {
                    "type": "picture",
                    "content": "[图像处理已禁用]",
                    "sheet": sheet_name,
                    "skip_reason": "disabled"
                }
            
            # 提取图像
            pil_img = self._extract_image_from_picture_item(doc, item)
            if pil_img is None:
                return {
                    "type": "picture",
                    "content": "[无法提取图像]",
                    "sheet": sheet_name,
                    "skip_reason": "no_image"
                }
            
            # 检查是否应该跳过
            should_skip, skip_reason = self._should_skip_image_by_size(pil_img)
            if should_skip:
                logger.debug(f"跳过图像: {skip_reason}")
                return {
                    "type": "picture",
                    "content": "",
                    "sheet": sheet_name,
                    "skip_reason": skip_reason
                }
            
            # 保存图像（可选）
            image_path = None
            if self.temp_image_dir:
                self.temp_image_dir.mkdir(parents=True, exist_ok=True)
                timestamp = int(time.time() * 1000)
                filename = f"sheet_{sheet_name}_img_{timestamp}.png"
                image_path = self.temp_image_dir / filename
                pil_img.save(str(image_path))
                logger.debug(f"图像已保存: {image_path}")
            
            # VLM分析
            description = "[VLM未启用]"
            if self.enable_vlm:
                description = self._describe_image_with_vlm(pil_img)
            
            return {
                "type": "picture",
                "content": description,
                "sheet": sheet_name,
                "image_path": str(image_path) if image_path else None,
                "image_size": pil_img.size
            }
            
        except Exception as e:
            logger.exception(f"处理PictureItem失败: {e}")
            return {
                "type": "picture",
                "content": f"[处理失败: {e}]",
                "sheet": sheet_name,
                "error": str(e)
            }

    # ===== 主要处理方法 =====
    def _build_reference_index(self, doc) -> Dict[str, Any]:
        """
        构建引用索引：从self_ref到实际对象的映射
        
        Args:
            doc: DoclingDocument对象
            
        Returns:
            引用索引字典 {self_ref: actual_item}
        """
        ref_index = {}
        item_types = {}
        
        try:
            # 遍历所有项目建立引用索引
            for item, level in doc.iterate_items():
                self_ref = getattr(item, "self_ref", None)
                if self_ref:
                    ref_index[self_ref] = item
                    item_type = type(item).__name__
                    item_types[item_type] = item_types.get(item_type, 0) + 1
                    
            logger.info(f"构建引用索引完成，包含 {len(ref_index)} 个引用")
            logger.info(f"引用对象类型分布: {item_types}")
            
            # 调试：打印前几个引用
            ref_keys = list(ref_index.keys())[:3]
            for ref_key in ref_keys:
                item = ref_index[ref_key]
                logger.debug(f"引用示例: {ref_key} -> {type(item).__name__}")
            
        except Exception as e:
            logger.warning(f"构建引用索引失败: {e}")
            
        return ref_index

    def _resolve_ref_item(self, child_ref, ref_index: Dict[str, Any]):
        """
        解析RefItem引用，获取实际的数据对象
        
        Args:
            child_ref: RefItem对象或其他引用
            ref_index: 引用索引字典
            
        Returns:
            实际的数据对象或None
        """
        try:
            # 如果child_ref有cref属性，说明是RefItem
            if hasattr(child_ref, "cref"):
                cref = child_ref.cref
                # 从引用索引中查找实际对象
                actual_item = ref_index.get(cref)
                if actual_item:
                    logger.debug(f"成功解析引用: {cref} -> {type(actual_item).__name__}")
                    return actual_item
                else:
                    logger.debug(f"未找到引用对应的对象: {cref}")
                    return None
            else:
                # 如果不是RefItem，直接返回
                return child_ref
                
        except Exception as e:
            logger.debug(f"解析引用失败: {e}")
            return None

    def _process_groups_based(self, doc, need_image: bool) -> List[Dict[str, Any]]:
        """
        基于doc.groups按顺序处理Excel的多个Sheet
        
        Args:
            doc: DoclingDocument对象
            need_image: 是否需要处理图像
            
        Returns:
            按Sheet分组的处理结果列表
        """
        sheets_output = []
        groups = getattr(doc, "groups", None)
        
        if not groups:
            logger.warning("文档中未找到groups，无法按Sheet分组")
            return []
        
        # 第1步：构建引用索引
        ref_index = self._build_reference_index(doc)
        if not ref_index:
            logger.warning("引用索引为空，回退到iterate_items模式")
            return []
        
        logger.info(f"发现 {len(groups)} 个groups（Sheet），引用索引包含 {len(ref_index)} 个对象")
        
        for group_index, group in enumerate(groups):
            try:
                # 提取Sheet名称
                raw_name = getattr(group, "name", None) or getattr(group, "label", None)
                if isinstance(raw_name, str) and raw_name.startswith("sheet:"):
                    sheet_name = raw_name.split("sheet:", 1)[1].strip()
                elif raw_name:
                    sheet_name = str(raw_name)
                else:
                    sheet_name = f"Sheet{group_index + 1}"
                
                logger.info(f"处理Sheet: {sheet_name}")
                
                # 处理group中的children
                children = getattr(group, "children", []) or []
                sheet_items = []

                logger.debug(f"Sheet '{sheet_name}' 包含 {len(children)} 个child引用")
                
                for child_index, child_ref in enumerate(children):
                    try:
                        # 第2步：解析RefItem引用获取实际对象
                        actual_item = self._resolve_ref_item(child_ref, ref_index)
                        
                        if actual_item is None:
                            logger.debug(f"无法解析child_ref (sheet={sheet_name}, index={child_index})")
                            continue
                        
                        # 第3步：按实际对象类型处理
                        if isinstance(actual_item, TextItem) or (hasattr(actual_item, "get_text") or hasattr(actual_item, "text")) and not hasattr(actual_item, "export_to_dataframe"):
                            result = self._process_text_item(actual_item, doc, sheet_name)
                            sheet_items.append((child_index, result))
                            logger.debug(f"处理TextItem: {child_index}")
                            
                        elif isinstance(actual_item, TableItem) or hasattr(actual_item, "export_to_dataframe"):
                            result = self._process_table_item(actual_item, doc, sheet_name)
                            sheet_items.append((child_index, result))
                            logger.debug(f"处理TableItem: {child_index}")
                            
                        elif isinstance(actual_item, PictureItem) or hasattr(actual_item, "get_image") or hasattr(actual_item, "image"):
                            if need_image and self.enable_vlm:  # 限制并行图像数量
                                result = self._process_picture_item(actual_item, doc, sheet_name, need_image)
                                sheet_items.append((child_index, result))
                                logger.debug(f"串行处理PictureItem: {child_index}")
                        else:
                            # 其他类型元素
                            sheet_items.append((child_index, {
                                "type": "other",
                                "content": f"{type(actual_item).__name__}: {str(actual_item)[:100]}...",
                                "sheet": sheet_name
                            }))
                            logger.debug(f"处理其他类型: {child_index} - {type(actual_item).__name__}")
                            
                    except Exception as e:
                        logger.exception(f"处理child失败 (sheet={sheet_name}, index={child_index}): {e}")
                        sheet_items.append((child_index, {
                            "type": "error",
                            "content": f"处理失败: {e}",
                            "sheet": sheet_name
                        }))
                
                # 第5步：按原始顺序排序并输出
                sheet_items.sort(key=lambda x: x[0])
                final_items = [item[1] for item in sheet_items]
                
                # 统计元素类型
                type_counts = {}
                for item in final_items:
                    item_type = item.get("type", "unknown")
                    type_counts[item_type] = type_counts.get(item_type, 0) + 1
                
                sheets_output.append({
                    "sheet_name": sheet_name,
                    "items": final_items,
                    "item_count": len(final_items),
                    "type_counts": type_counts
                })
                
                logger.info(f"Sheet '{sheet_name}' 处理完成，包含 {len(final_items)} 个元素: {type_counts}")
                
            except Exception as e:
                logger.exception(f"处理group失败 (index={group_index}): {e}")
                
        return sheets_output

    def __call__(self, filename_or_bytes, need_image: bool = True, is_xls: bool = False) -> List[Dict[str, Any]]:
        """
        解析Excel文件的主入口方法
        
        Args:
            filename_or_bytes: 文件路径字符串或字节数据
            need_image: 是否需要处理图像（启用VLM分析）
            is_xls: 是否为XLS格式（需要转换）
            
        Returns:
            按Sheet分组的解析结果列表
        """
        start_time = time.time()
        temp_files = []  # 记录临时文件用于清理
        
        try:
            logger.info(f"开始解析Excel文件，need_image={need_image}, is_xls={is_xls}")
            
            # 动态调整VLM设置
            if need_image != self.enable_vlm:
                logger.info(f"动态调整VLM设置: {self.enable_vlm} -> {need_image}")
                self.enable_vlm = need_image
            
            # 第1步：处理输入源
            if is_xls:
                # XLS需要先转换为XLSX
                if isinstance(filename_or_bytes, (bytes, bytearray)):
                    # 字节数据：先写入临时XLS文件
                    with tempfile.NamedTemporaryFile(suffix=".xls", delete=False) as tmp_xls:
                        tmp_xls.write(filename_or_bytes)
                        tmp_xls.flush()
                        temp_files.append(tmp_xls.name)
                        xls_path = tmp_xls.name
                else:
                    # 文件路径
                    xls_path = str(filename_or_bytes)
                
                # 转换XLS到XLSX
                xlsx_path = self._convert_xls_to_xlsx(xls_path)
                temp_files.append(xlsx_path)
                source = str(xlsx_path)
            else:
                # 直接处理XLSX
                if isinstance(filename_or_bytes, (bytes, bytearray)):
                    source = DocumentStream(name="document.xlsx", stream=BytesIO(filename_or_bytes))
                else:
                    source = str(filename_or_bytes)
            
            # 第2步：使用Docling转换
            logger.info("开始Docling文档转换...")
            result = self.converter.convert(source)
            doc = result.document
            
            if doc:
                logger.info(f"Docling转换成功，耗时: {time.time() - start_time:.2f}s")
            else:
                logger.warning(f"Docling转换可能有问题: {result.status}")
            
            # 第3步：分析文档结构
            groups = getattr(doc, "groups", None)
            items_count = len(list(doc.iterate_items())) if hasattr(doc, 'iterate_items') else 0
            
            logger.info(f"文档结构分析: groups={len(groups) if groups else 0}, total_items={items_count}")
            
            # 第4步：按groups处理或回退到iterate_items
            if groups and len(groups) > 0:
                logger.info("使用groups模式处理Excel")
                sheets_result = self._process_groups_based(doc, need_image)
            else:
                logger.info("使用iterate_items回退模式处理Excel")
                print("docling faild")

            # 第5步：后处理和验证
            total_items = sum(sheet.get("item_count", 0) for sheet in sheets_result)
            logger.info(f"Excel解析完成: {len(sheets_result)} 个Sheet, 总计 {total_items} 个元素, 耗时: {time.time() - start_time:.2f}s")
            
            return sheets_result
            
        except Exception as e:
            logger.exception(f"Excel解析失败: {e}")
            raise
            
        finally:
            # 清理临时文件
            for temp_file in temp_files:
                try:
                    if Path(temp_file).exists():
                        Path(temp_file).unlink()
                        logger.debug(f"已清理临时文件: {temp_file}")
                except Exception as e:
                    logger.debug(f"清理临时文件失败: {temp_file}, {e}")
            
            # 强制垃圾回收
            if self.enable_vlm:
                gc.collect()


# ===== Ollama客户端示例 =====
class SimpleOllamaClient:
    """
    简单的Ollama VLM客户端
    支持本地ollama包和HTTP API两种方式
    """
    def __init__(self, model: str = "llama3.2-vision", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def describe(self, pil_image: Image.Image, prompt: str = "请描述这张图片的内容，并提取其中的文字：") -> str:
        """
        使用VLM分析图像
        
        Args:
            pil_image: PIL图像对象
            prompt: 分析提示词
            
        Returns:
            图像描述文本
        """
        try:
            # 优先使用ollama包
            if ollama:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpf:
                    pil_image.save(tmpf.name)
                    tmpf.flush()
                    resp = ollama.chat(
                        model=self.model, 
                        messages=[{
                            "role": "user", 
                            "content": prompt, 
                            "images": [tmpf.name]
                        }]
                    )
                    # 清理临时文件
                    try:
                        os.unlink(tmpf.name)
                    except:
                        pass
                    
                    # 提取响应内容
                return resp["message"].get("content", str(resp))

        except Exception as e:
            logger.debug(f"ollama包调用失败，尝试HTTP API: {e}")

        # 回退到HTTP API
        try:
            import requests
            with tempfile.NamedTemporaryFile(suffix=".png") as tmpf:
                pil_image.save(tmpf.name)
                
                # 准备请求数据
                with open(tmpf.name, "rb") as img_file:
                    import base64
                    img_b64 = base64.b64encode(img_file.read()).decode()
                
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "images": [img_b64]
                }
                
                response = requests.post(
                    f"{self.base_url}/api/chat", 
                    json=payload, 
                    timeout=120
                )
                response.raise_for_status()
                
                # 解析响应
                resp_data = response.json()
                if "message" in resp_data:
                    return resp_data["message"].get("content", str(resp_data))
                return str(resp_data)
                
        except Exception as e:
            logger.error(f"Ollama HTTP请求失败: {e}")
            return f"[Ollama分析失败: {e}]"


# ===== 使用示例 =====
if __name__ == "__main__":
    # 初始化VLM客户端
    try:
        vlm_client = SimpleOllamaClient(
            model="blaifa/InternVL3_5:8b",  # 或其他支持的VLM模型
            base_url="http://localhost:11434"
        )
        print("✅ VLM客户端初始化成功")
    except Exception as e:
        print(f"❌ VLM客户端初始化失败: {e}")
        vlm_client = None

    # 初始化Excel解析器
    parser = ExcelParserVLM(
        vlm_mdl=vlm_client,
        enable_vlm=bool(vlm_client),
        max_workers=4,
        min_image_area=30000,  # 200x150像素
        temp_image_dir="/tmp/docling_excel_images"
    )
    print("✅ Excel解析器初始化完成")

    # 解析示例文件
    try:
        excel_file = "/home/zzg/商业项目/upwork/RAG/docs/Swire_Excel Financial Model.xls"
        print(f"📊 开始解析Excel文件: {excel_file}")
        
        # 解析Excel（XLS格式需要转换）
        result = parser(excel_file, need_image=True, is_xls=True)
        
        # 输出结果摘要
        print(f"\n🎉 解析完成！发现 {len(result)} 个Sheet:")
        
        for i, sheet in enumerate(result):
            sheet_name = sheet["sheet_name"]
            items = sheet["items"]
            item_count = len(items)
            
            print(f"\n📋 Sheet {i+1}: '{sheet_name}' ({item_count} 个元素)")
            
            # 统计各类型元素数量
            type_counts = {}
            for item in items:
                item_type = item.get("type", "unknown")
                type_counts[item_type] = type_counts.get(item_type, 0) + 1
            
            for item_type, count in type_counts.items():
                print(f"   - {item_type}: {count} 个")
            
            # 显示前几个元素的内容预览
            print(f"   📝 内容预览:")
            for j, item in enumerate(items[:3]):  # 只显示前3个
                content = item.get("content", "")
                if isinstance(content, str) and len(content) > 50:
                    content = content[:50] + "..."
                print(f"     {j+1}. [{item.get('type', '未知')}] {content}")
            
            if len(items) > 3:
                print(f"     ... (还有 {len(items) - 3} 个元素)")
        
    except Exception as e:
        print(f"❌ Excel解析失败: {e}")
        import traceback
        traceback.print_exc()
