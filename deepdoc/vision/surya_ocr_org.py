# coding=utf-8
# @Time : 2024/9/6 16:52
# @File : surya_ocr.py
from collections import defaultdict, OrderedDict
from typing import List

from PIL import Image

from surya.models import load_predictors
from surya.detection.schema import TextDetectionResult
from surya.recognition.schema import OCRResult
from surya.layout.schema import LayoutResult
from surya.input.processing import (
    convert_if_not_rgb,
)
from unstructured.partition.pdf_image.inference_utils import build_text_region_from_coords
from unstructured_inference.inference.elements import TextRegion
import re


class SuryaOCR:
    def __init__(self):
        print("loading surya model...")
        self.predictors = load_predictors()
        self.det_predictor = self.predictors["detection"]
        self.rec_predictor = self.predictors["recognition"]
        self.layout_predictor = self.predictors["layout"]
        # Note: ordering is no longer available in new version
        self.order_predictor = None

    def det_target(self, img, target):
        layout_pred = self.layout_detection(img)

        return next((i for i in layout_pred.bboxes if i.label == target), None)

    def calculate_intersection_area(self, box1, box2):
        # 计算两个矩形框的交集面积
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        # 计算相交矩形的宽度和高度
        inter_width = max(0, x_right - x_left)
        inter_height = max(0, y_bottom - y_top)

        return inter_width * inter_height
        # return intersection_area

    def sort_ocr_results_by_reading_order(self, det_results, text_results, reading_order_boxes):
        sorted_ocr = defaultdict(list)

        for order_index, reading_box in enumerate(reading_order_boxes):
            # 将当前阅读顺序框内的 OCR 结果分配到对应的段落中
            ocr_in_this_section = []

            for i, det_box in enumerate(det_results):
                intersection_area = self.calculate_intersection_area(det_box, reading_box)
                if intersection_area > 0:
                    # 去重
                    duplicated_flag = False
                    for box_idx, box_ocr in sorted_ocr.items():
                        for text_idx, text_box in enumerate(box_ocr):
                            if text_box[1] == text_results[i]:
                                intersection_area_tmp = self.calculate_intersection_area(
                                    text_box[0], reading_order_boxes[box_idx]
                                )
                                if intersection_area_tmp > intersection_area:
                                    del sorted_ocr[box_idx][text_idx]
                                    # duplicated_flag = False
                                else:
                                    duplicated_flag = True
                                break
                    if not duplicated_flag:
                        ocr_in_this_section.append((det_box, text_results[i]))

            # 如果有多个 OCR 结果在同一个段落中，需要进一步排序
            ocr_in_this_section.sort(
                key=lambda ocr: (ocr[0][1], ocr[0][0])
            )  # 按照Y坐标排序，次要按X

            sorted_ocr[order_index].extend(ocr_in_this_section)

        sorted_text_results = []
        for rbidx in range(len(reading_order_boxes)):
            for item in sorted_ocr[rbidx]:
                sorted_text_results.append(item)
        return sorted_text_results

    def run_ocr(
            self,
            images: List[Image.Image],
            batch_size=4,
    ) -> List[OCRResult]:
        # Convert images to RGB
        images = convert_if_not_rgb(images)

        # Use the new recognition predictor with detection
        results = self.rec_predictor(
            images,
            det_predictor=self.det_predictor,
            recognition_batch_size=batch_size,
            sort_lines=True
        )

        return results

    def layout_detection(self, img) -> LayoutResult:
        # Convert PIL Image to RGB if needed
        if not isinstance(img, Image.Image):
            raise ValueError("Input must be a PIL Image")
        img = img.convert("RGB")
        pred = self.layout_predictor([img])[0]
        return pred

    def text_detection(self, img) -> TextDetectionResult:
        # Convert PIL Image to RGB if needed
        if not isinstance(img, Image.Image):
            raise ValueError("Input must be a PIL Image")
        img = img.convert("RGB")
        pred = self.det_predictor([img])[0]
        return pred

    def order_detection(self, img):
        # Note: Ordering functionality is not available in surya-ocr 0.16.7+
        # Fallback to layout detection only
        layout_pred = self.layout_detection(img)
        # Return a simplified ordered dict based on reading order (top-to-bottom, left-to-right)
        sorted_boxes = sorted(enumerate(layout_pred.bboxes), key=lambda x: (x[1].bbox[1], x[1].bbox[0]))
        return OrderedDict({
            i: {
                "bbox": box.bbox,
                'conf': box.confidence,
                "label": box.label
            } for i, (_, box) in enumerate(sorted_boxes)
        })

    def predict(self, image, chat_mdl):
        # ocr
        # replace_lang_with_code(langs)
        img_pred = self.run_ocr(
            [image], batch_size=4
        )[0]
        for line in img_pred.text_lines:
            line.text = line.text.replace("\t", "")
            line.text = line.text.replace("\n", "")
        if len(img_pred.text_lines) == 0:
            return ''

        # 定义你想删除的特殊字符
        pattern = r'[。？！；!?;+.]+'
        cleaned_text = re.sub(pattern, '', img_pred.text_lines[0].text)
        if len(cleaned_text) < 2:
            return ''
        text = [l.text for l in img_pred.text_lines]

        # Sort OCR results by reading order
        prompt_user = f"""
                OCR Results:
                {text}
                The above OCR results are text data extracted from images using OCR.
                Based on context, please correct any errors in the OCR results to the correct text.
                Please organize the information in an easy-to-understand manner.
                Output only the organized OCR information. Do not include any explanations or supplementary information.
                Do not output unnecessary preambles or explanations.
                Remove all Unicode symbols (such as emoticons or special symbols) and irrelevant content from the OCR results. Keep only necessary information.
                Do not translate the OCR results.
                """
        # llm
        prompt_system = "You are an AI assistant who specializes in organizing fragmented information and extracting useful information from it."
        res = chat_mdl.chat(prompt_system,
                            [{"role": "user", "content": prompt_user}],
                            {"temperature": 0.0}
                            )
        return res


class OCRAgentSurya():
    """OCR service implementation for Tesseract."""

    def __init__(self, language: str = "eng"):
        self.language = language

    def is_text_sorted(self):
        return True

    def get_layout_from_image(self, image: Image.Image) -> List[TextRegion]:
        """Get the OCR regions from image as a list of text regions with tesseract."""
        ocr_regions: list[TextRegion] = []
        img_pred = surya_model.run_ocr(
            images=[image], batch_size=8
        )[0]

        for l in img_pred.text_lines:
            cleaned_text = str(l.text) if not isinstance(l.text, str) else l.text.strip()
            if cleaned_text:
                x1, y1, x2, y2 = l.bbox
                text_region = build_text_region_from_coords(
                    x1, y1, x2, y2, text=cleaned_text, source="surya"
                )
                ocr_regions.append(text_region)
        return ocr_regions


surya_model = None
