import logging
import json
import fitz  # PyMuPDF
from openai import OpenAI
from rapidocr_onnxruntime import RapidOCR
import cv2
import numpy as np
import re
import os
import tkinter as tk
from tkinter import filedialog, ttk
import threading
import sys
import subprocess
import time

__version__ = "4.0"

# --- 配置日志 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(SCRIPT_DIR, "pdf_whiteout.log")
env_file = os.path.join(SCRIPT_DIR, ".env")

def load_local_env(path):
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

load_local_env(env_file)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- 配置类 ---
class Config:
    # OpenAI-compatible local LLM API configuration.
    # Set these in your shell instead of hard-coding private local settings:
    #   export TOPICKEEPER_LLM_BASE_URL="http://127.0.0.1:8000/v1"
    #   export TOPICKEEPER_LLM_API_KEY="your-local-api-key"
    #   export TOPICKEEPER_LLM_MODEL="your-model-name"
    LLM_BASE_URL = os.getenv("TOPICKEEPER_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    LLM_API_KEY = os.getenv("TOPICKEEPER_LLM_API_KEY", "local-api-key")
    LLM_MODEL = os.getenv("TOPICKEEPER_LLM_MODEL", "local-model-name")

    # 涂白参数
    WHITEOUT_COLOR = (1, 1, 1)  # RGB 白色
    PADDING_Y = 7  # 垂直方向涂白扩充像素
    PADDING_X = 8  # 水平方向扩充
    MERGE_RECT_GAP = 6  # 同一行相邻涂白框的合并容差
    BOUNDARY_SAFE_GAP = 18  # 整段涂白距离保留标题/正文的安全间距
    DELETE_UNMARKED_PAGES = True  # 扫描件整页无保留内容时，删除该页而不是因为有底图而保留
    
    # OCR 配置
    USE_RAPIDOCR = True
    OCR_DPI = 320
    OCR_FALLBACK_DPI = 420
    OCR_MIN_BLOCKS_PER_PAGE = 3
    OCR_MIN_CONFIDENCE = 0.35
    TITLE_MATCH_WINDOW = 4

    # 生成后结构检查：再 OCR 一遍输出文件，用规则检查标题、纪要头、目标议题和人员/footer 是否保留。
    POST_GENERATION_STRUCTURE_CHECK = True
    POST_CHECK_DPI = 220
    PREPROCESS_AFTER_FILE_SELECT = True

class SemanticWhiteoutTool:
    def __init__(self):
        self.client = OpenAI(
            base_url=Config.LLM_BASE_URL,
            api_key=Config.LLM_API_KEY
        )
        if Config.USE_RAPIDOCR:
            self.ocr_engine = RapidOCR()
            logger.info("RapidOCR engine initialized.")
        else:
            raise NotImplementedError("Currently only RapidOCR is supported.")

    def _render_page_pixmap(self, page, dpi):
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        return page.get_pixmap(matrix=matrix, alpha=False)

    def _pixmap_to_cv_image(self, pix):
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        if pix.n == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def _encode_png(self, image):
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise ValueError("OpenCV failed to encode image for OCR")
        return encoded.tobytes()

    def _build_ocr_variants(self, pix):
        image = self._pixmap_to_cv_image(pix)
        variants = [("原图", pix.tobytes("png"), pix.width, pix.height)]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)

        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(denoised, -1, sharpen_kernel)
        variants.append(("增强灰度", self._encode_png(sharpened), pix.width, pix.height))

        binary = cv2.adaptiveThreshold(
            sharpened,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11
        )
        variants.append(("自适应二值化", self._encode_png(binary), pix.width, pix.height))
        return variants

    def _parse_ocr_result(self, result, page_index, pdf_w, pdf_h, img_w, img_h, source):
        blocks = []
        if not result:
            return blocks

        scale_x = pdf_w / img_w
        scale_y = pdf_h / img_h
        for line in result:
            try:
                confidence = 1.0
                if len(line) == 3:
                    points = line[0]
                    text = str(line[1]).strip()
                    confidence = float(line[2]) if isinstance(line[2], (int, float)) else 1.0
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    x0, y0 = min(xs), min(ys)
                    x2, y2 = max(xs), max(ys)
                elif len(line) == 5:
                    x0, y0, x2, y2, text = line
                    text = str(text).strip()
                elif len(line) == 6:
                    x0, y0, x2, y2, text, confidence = line
                    text = str(text).strip()
                    confidence = float(confidence) if isinstance(confidence, (int, float)) else 1.0
                else:
                    continue

                if not text or confidence < Config.OCR_MIN_CONFIDENCE:
                    continue

                pdf_bbox = (x0 * scale_x, y0 * scale_y, x2 * scale_x, y2 * scale_y)
                blocks.append({
                    'page': page_index,
                    'text': text,
                    'bbox': pdf_bbox,
                    'page_width': pdf_w,
                    'page_height': pdf_h,
                    'confidence': confidence,
                    'source': source
                })
            except Exception as e:
                logger.debug(f"Skip malformed OCR line on page {page_index + 1}: {e}")
        return blocks

    def _dedupe_blocks(self, blocks):
        seen = set()
        deduped = []
        for block in blocks:
            rect = fitz.Rect(block['bbox'])
            key = (
                block['page'],
                self._normalize_text(block['text'])[:30],
                round(rect.x0),
                round(rect.y0),
                round(rect.x1),
                round(rect.y1)
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(block)
        return deduped

    def _sort_ocr_blocks(self, blocks):
        def sort_key(block):
            rect = fitz.Rect(block['bbox'])
            line_bucket = round(rect.y0 / 8)
            return (block['page'], line_bucket, rect.x0)
        return sorted(blocks, key=sort_key)

    def _ocr_page(self, page, page_index, dpi, use_all_variants=False):
        pix = self._render_page_pixmap(page, dpi)
        variants = self._build_ocr_variants(pix) if use_all_variants else [("原图", pix.tobytes("png"), pix.width, pix.height)]
        best_blocks = []
        best_source = ""

        for source, img_bytes, img_w, img_h in variants:
            result, _ = self.ocr_engine(img_bytes)
            blocks = self._parse_ocr_result(result, page_index, page.rect.width, page.rect.height, img_w, img_h, source)
            logger.info(f"Page {page_index + 1} OCR [{source}, {dpi}dpi]: {len(blocks)} blocks")
            if len(blocks) > len(best_blocks):
                best_blocks = blocks
                best_source = source

        if best_source:
            logger.info(f"Page {page_index + 1} selected OCR variant: {best_source}, {len(best_blocks)} blocks")
        return best_blocks

    def _extract_layout_ocr(self, pdf_path):
        logger.info(f"--- Starting OCR for {pdf_path} ---")
        if not os.path.exists(pdf_path):
            logger.error(f"File not found: {pdf_path}")
            return []

        doc = fitz.open(pdf_path)
        all_blocks = []
        try:
            for page_index in range(len(doc)):
                page = doc[page_index]
                page_blocks = self._ocr_page(page, page_index, Config.OCR_DPI, use_all_variants=False)

                if len(page_blocks) < Config.OCR_MIN_BLOCKS_PER_PAGE:
                    logger.info(f"Page {page_index + 1} has few OCR blocks, trying enhanced scan pipeline...")
                    enhanced_blocks = self._ocr_page(page, page_index, Config.OCR_DPI, use_all_variants=True)
                    if len(enhanced_blocks) > len(page_blocks):
                        page_blocks = enhanced_blocks

                if len(page_blocks) < Config.OCR_MIN_BLOCKS_PER_PAGE:
                    logger.info(f"Page {page_index + 1} still weak, retrying at {Config.OCR_FALLBACK_DPI}dpi...")
                    high_dpi_blocks = self._ocr_page(page, page_index, Config.OCR_FALLBACK_DPI, use_all_variants=True)
                    if len(high_dpi_blocks) > len(page_blocks):
                        page_blocks = high_dpi_blocks

                all_blocks.extend(page_blocks)
        finally:
            doc.close()

        all_blocks = self._sort_ocr_blocks(self._dedupe_blocks(all_blocks))
        logger.info(f"--- OCR Completed. Extracted {len(all_blocks)} text blocks. ---")
        return all_blocks

    def _extract_validation_ocr(self, pdf_path):
        logger.info(f"--- Starting post-generation OCR check for {pdf_path} ---")
        if not os.path.exists(pdf_path):
            logger.error(f"File not found: {pdf_path}")
            return []

        doc = fitz.open(pdf_path)
        all_blocks = []
        try:
            for page_index in range(len(doc)):
                page_blocks = self._ocr_page(doc[page_index], page_index, Config.POST_CHECK_DPI, use_all_variants=False)
                all_blocks.extend(page_blocks)
        finally:
            doc.close()

        all_blocks = self._sort_ocr_blocks(self._dedupe_blocks(all_blocks))
        logger.info(f"--- Post-generation OCR check completed. Extracted {len(all_blocks)} text blocks. ---")
        return all_blocks

    def _analyze_structure_with_llm(self, ocr_blocks, user_keyword):
        logger.info("--- Sending data to LLM for semantic analysis ---")
        
        text_stream = ""
        for i, block in enumerate(ocr_blocks):
            if i > 3000: break
            text_stream += f"[{i}] {block['text']}\n"
        
        prompt = f"""
你是一个专业的会议纪要分析助手。
**用户输入的关键词**: "{user_keyword}"

**文本内容** (按OCR识别顺序排列，每行前面的 [] 是 OCR 块索引):
{text_stream}

**任务定义**:
1. **目标议题**: 找出用户关键词对应的议题。提取该议题的**完整标题**。
2. **下一个议题**: 提取紧跟在目标议题后面的那个议题的**完整标题**。如果目标议题是最后一个，则返回 "NULL"。
3. **纪要结尾**: 提取文档最后一部分（通常是“出席”或“参加会议人员”那一行）的一小段**独特且不常见**的文字。

**关键要求**:
- 标题跨多行时，请合并成一个完整标题，不要只返回最后一行。
- footer_anchor 必须尽量选择 OCR 文本中真实出现的一小段连续文字。
- 不要输出解释、Markdown、代码块之外的文字。

**输出要求**:
仅返回 JSON 格式：
{{
    "target_title": "目标议题的完整标题",
    "next_topic_title": "下一个议题的完整标题 (如果没有则为 NULL)",
    "footer_anchor": "纪要结尾中的一段独特文字"
}}
"""
        try:
            response = self.client.chat.completions.create(
                model=Config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个严谨的文档结构分析助手。只输出JSON。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            content = response.choices[0].message.content
            logger.info(f"LLM Raw Response: {content}")
            json_str = self._extract_json_object(content)
            analysis_result = json.loads(json_str)
            
            if not analysis_result.get("target_title"):
                raise ValueError("LLM failed to identify target_title")
            
            logger.info(f"--- Analysis Result ---")
            logger.info(f"Target: {analysis_result.get('target_title')}")
            logger.info(f"Next: {analysis_result.get('next_topic_title')}")
            logger.info(f"Footer: {analysis_result.get('footer_anchor')}")
            return analysis_result
        except Exception as e:
            logger.error(f"LLM API Error or Parsing Error: {e}")
            raise

    def _extract_json_object(self, content):
        cleaned = re.sub(r'```json|```', '', str(content or ""), flags=re.IGNORECASE).strip()
        match = re.search(r'\{.*\}', cleaned, flags=re.DOTALL)
        if match:
            return match.group(0)
        return cleaned

    def _normalize_text(self, text):
        text = str(text or "")
        text = re.sub(r'\s+', '', text)
        text = re.sub(r'[^\w\u4e00-\u9fff]', '', text)
        return text.lower()

    def _find_multiline_title_start(self, ocr_blocks, normalized_snippet, start_idx):
        if len(normalized_snippet) < 8:
            return -1

        snippet_prefix = normalized_snippet[:6]
        for i in range(start_idx, len(ocr_blocks)):
            page_idx = ocr_blocks[i]['page']
            candidate_text = self._normalize_text(ocr_blocks[i]['text'])
            if snippet_prefix not in candidate_text:
                continue

            window_text = ""
            for j in range(i, min(len(ocr_blocks), i + Config.TITLE_MATCH_WINDOW)):
                if ocr_blocks[j]['page'] != page_idx:
                    break
                window_text += self._normalize_text(ocr_blocks[j]['text'])
                if len(window_text) >= 8 and (
                    normalized_snippet in window_text or
                    normalized_snippet[:12] in window_text
                ):
                    logger.info(f"Found multiline title start at index {i}: '{ocr_blocks[i]['text']}'")
                    return i
        return -1

    def _find_block_index_flexible(self, ocr_blocks, text_snippet, start_idx=0):
        if not text_snippet or text_snippet == "NULL":
            return -1

        normalized_snippet = self._normalize_text(text_snippet)
        if not normalized_snippet:
            return -1
        
        for i in range(start_idx, len(ocr_blocks)):
            block = ocr_blocks[i]
            if text_snippet in block['text']:
                logger.info(f"Found '{text_snippet}' at index {i}: '{block['text']}'")
                return i

        for i in range(start_idx, len(ocr_blocks)):
            block = ocr_blocks[i]
            normalized_block = self._normalize_text(block['text'])
            if normalized_snippet in normalized_block:
                logger.info(f"Found normalized '{text_snippet}' at index {i}: '{block['text']}'")
                return i

        multiline_idx = self._find_multiline_title_start(ocr_blocks, normalized_snippet, start_idx)
        if multiline_idx != -1:
            return multiline_idx
        
        if len(text_snippet) > 4:
            short_snippet = normalized_snippet[:12]
            for i in range(start_idx, len(ocr_blocks)):
                block = ocr_blocks[i]
                if short_snippet and short_snippet in self._normalize_text(block['text']):
                    logger.info(f"Found partial '{short_snippet}' at index {i}")
                    return i

        for i in range(start_idx, len(ocr_blocks)):
            block = ocr_blocks[i]
            normalized_block = self._normalize_text(block['text'])
            if len(normalized_block) >= 12 and normalized_block in normalized_snippet:
                logger.info(f"Found trailing fragment '{block['text']}' at index {i}")
                return i
        return -1

    def _predict_next_topic_patterns(self, target_title):
        patterns = []
        cn_numerals = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十']
        for char in target_title:
            if char in cn_numerals:
                idx = cn_numerals.index(char)
                if idx + 1 < len(cn_numerals):
                    next_cn = cn_numerals[idx + 1]
                    patterns.extend([f"议题{next_cn}", f"{next_cn}、"])
                break
        
        match = re.search(r'\d+', target_title)
        if match:
            num = int(match.group())
            next_num = num + 1
            patterns.extend([f"议题{next_num}", f"{next_num}、", f"{next_num}."])
            
        if patterns:
            logger.info(f"Predicted next patterns: {patterns}")
        return patterns

    def _search_for_subsequent_topic_by_regex(self, ocr_blocks, start_idx, target_title):
        logger.info("--- LLM missed next topic. Trying Regex fallback ---")
        patterns = self._predict_next_topic_patterns(target_title)
        if not patterns:
            return -1
            
        for i in range(start_idx, len(ocr_blocks)):
            text = ocr_blocks[i]['text']
            for pattern in patterns:
                if re.search(pattern, text):
                    logger.info(f"Regex FOUND next topic pattern '{pattern}' at index {i}")
                    return i
        logger.info("Regex fallback failed to find next topic.")
        return -1

    def _find_first_matching_block(self, ocr_blocks, candidates, start_idx=0, end_idx=None):
        if not ocr_blocks:
            return -1
        end_idx = len(ocr_blocks) if end_idx is None else min(end_idx, len(ocr_blocks))
        normalized_candidates = [self._normalize_text(candidate) for candidate in candidates if self._normalize_text(candidate)]

        for i in range(max(0, start_idx), end_idx):
            normalized_block = self._normalize_text(ocr_blocks[i].get('text', ''))
            if not normalized_block:
                continue
            for candidate in normalized_candidates:
                if candidate in normalized_block:
                    return i
        return -1

    def _find_multiline_phrase_index(self, ocr_blocks, phrase, start_idx=0, end_idx=None, window_size=None):
        normalized_phrase = self._normalize_text(phrase)
        if not normalized_phrase:
            return -1

        end_idx = len(ocr_blocks) if end_idx is None else min(end_idx, len(ocr_blocks))
        window_size = window_size or Config.TITLE_MATCH_WINDOW

        for i in range(max(0, start_idx), end_idx):
            page_idx = ocr_blocks[i]['page']
            window_text = ""
            for j in range(i, min(end_idx, i + window_size)):
                if ocr_blocks[j]['page'] != page_idx:
                    break
                window_text += self._normalize_text(ocr_blocks[j].get('text', ''))
                if normalized_phrase in window_text:
                    return i

        if len(normalized_phrase) >= 10:
            return self._find_first_matching_block(ocr_blocks, [normalized_phrase[:10]], start_idx, end_idx)
        return self._find_first_matching_block(ocr_blocks, [normalized_phrase], start_idx, end_idx)

    def _get_pdf_page_count(self, pdf_path):
        doc = fitz.open(pdf_path)
        try:
            return doc.page_count
        finally:
            doc.close()

    def _validate_output_structure(self, input_pdf, output_pdf, original_ocr_blocks, analysis_result, user_keyword):
        started_at = time.perf_counter()
        report = {
            "passed": True,
            "warnings": [],
            "details": []
        }

        try:
            input_pages = self._get_pdf_page_count(input_pdf)
            output_pages = self._get_pdf_page_count(output_pdf)
        except Exception as e:
            report["passed"] = False
            report["warnings"].append(f"PDF 文件打开失败: {e}")
            return report

        if output_pages <= 0:
            report["passed"] = False
            report["warnings"].append("输出 PDF 页数为 0")
            return report
        if output_pages > input_pages:
            report["passed"] = False
            report["warnings"].append(f"输出页数异常增加: {input_pages} -> {output_pages}")
        else:
            report["details"].append(f"页数检查: {input_pages} -> {output_pages}")

        output_blocks = self._extract_validation_ocr(output_pdf)
        if not output_blocks:
            report["passed"] = False
            report["warnings"].append("输出 PDF 没有识别到可用文字，可能被过度涂白")
            return report

        first_section_end = max(12, min(len(output_blocks), int(len(output_blocks) * 0.25)))
        last_section_start = max(0, int(len(output_blocks) * 0.60))

        title_idx = self._find_first_matching_block(output_blocks, ["会议纪要", "纪要"], 0, first_section_end)
        if title_idx == -1:
            report["warnings"].append("未在输出文件前部识别到“会议纪要/纪要”等标题锚点")
        else:
            report["details"].append(f"标题锚点: 第 {output_blocks[title_idx]['page'] + 1} 页")

        header_terms = ["时间", "地点", "主持", "会议", "议题", "审议", "听取", "研究"]
        header_idx = self._find_first_matching_block(output_blocks, header_terms, 0, first_section_end)
        if header_idx == -1:
            report["warnings"].append("未在输出文件前部识别到明显的纪要头/会议头信息")
        else:
            report["details"].append(f"纪要头锚点: 第 {output_blocks[header_idx]['page'] + 1} 页")

        target_title = analysis_result.get("target_title", "")
        target_idx = self._find_multiline_phrase_index(output_blocks, target_title)
        if target_idx == -1 and user_keyword:
            target_idx = self._find_multiline_phrase_index(output_blocks, user_keyword)
        if target_idx == -1:
            report["passed"] = False
            report["warnings"].append("未在输出文件中识别到目标议题标题或关键词")
        else:
            report["details"].append(f"目标议题锚点: 第 {output_blocks[target_idx]['page'] + 1} 页")

        footer_terms = ["出席", "列席", "请假", "记录", "分送", "参加会议人员", "参会人员"]
        footer_idx = self._find_first_matching_block(output_blocks, footer_terms, last_section_start)
        if footer_idx == -1:
            report["passed"] = False
            report["warnings"].append("未在输出文件后部识别到出席/列席/记录/分送等人员或结尾信息")
        else:
            report["details"].append(f"人员/footer 锚点: 第 {output_blocks[footer_idx]['page'] + 1} 页")

        order_points = [idx for idx in [title_idx if title_idx != -1 else header_idx, target_idx, footer_idx] if idx != -1]
        if len(order_points) >= 2 and order_points != sorted(order_points):
            report["passed"] = False
            report["warnings"].append("输出结构顺序异常：头部、目标议题、人员/footer 的位置不符合常规顺序")

        next_title = analysis_result.get("next_topic_title")
        if next_title and next_title != "NULL":
            residue_idx = self._find_multiline_phrase_index(output_blocks, next_title)
            if residue_idx != -1:
                report["warnings"].append(f"可能存在下一个议题标题残留: {output_blocks[residue_idx]['text'][:30]}")

        if not report["warnings"]:
            report["details"].append("结构完整性检查通过")

        elapsed = time.perf_counter() - started_at
        report["details"].append(f"结构检查耗时: {elapsed:.2f}s")
        logger.info(f"Structure check passed: {report['passed']}")
        for detail in report["details"]:
            logger.info(f"Structure check detail: {detail}")
        for warning in report["warnings"]:
            logger.warning(f"Structure check warning: {warning}")
        return report

    def _calculate_erase_bbox(self, ocr_blocks, analysis_result):
        logger.info("--- Calculating Coordinates ---")
        
        # 1. Header
        h_idx = -1
        for i, block in enumerate(ocr_blocks):
            if i < 20 and "：" in block['text']:
                h_idx = i
                logger.info(f"Header Boundary at {i}")
                break
        if h_idx == -1: h_idx = 0

        target_title = analysis_result.get("target_title")
        next_title_str = analysis_result.get("next_topic_title")
        footer_text = analysis_result.get("footer_anchor")

        t_idx = self._find_block_index_flexible(ocr_blocks, target_title)
        if t_idx == -1:
            raise ValueError("无法定位目标议题")

        # 2. Next Topic Boundary
        end_boundary_idx = -1
        if next_title_str and next_title_str != "NULL":
            end_boundary_idx = self._find_block_index_flexible(ocr_blocks, next_title_str, start_idx=t_idx)
        
        if end_boundary_idx != -1:
            if end_boundary_idx < t_idx:
                logger.warning(f"Next topic index {end_boundary_idx} < target index {t_idx}. Ignoring.")
                end_boundary_idx = -1
        else:
            logger.info("LLM next topic not found or NULL. Trying Regex.")
            end_boundary_idx = self._search_for_subsequent_topic_by_regex(ocr_blocks, t_idx, target_title)

        # 3. Footer Boundary
        f_idx = -1
        search_start_idx = int(len(ocr_blocks) * 0.8)
        logger.info(f"Searching footer in last 20% (from index {search_start_idx})")
        
        if footer_text and footer_text != "NULL":
            f_idx = self._find_block_index_flexible(ocr_blocks, footer_text, start_idx=search_start_idx)
            
        if f_idx == -1:
            logger.warning("Footer anchor not found. Defaulting to end of doc.")
            f_idx = len(ocr_blocks)
        else:
            f_idx = self._adjust_footer_start_index(ocr_blocks, f_idx)

        # Final Decision
        if end_boundary_idx == -1:
            end_boundary_idx = f_idx
            logger.info("Final Boundary Strategy: Use Footer")
        else:
            logger.info("Final Boundary Strategy: Use Next Topic")

        logger.info(f"Final Indices -> Header: {h_idx}, Target: {t_idx}, End: {end_boundary_idx}, Footer: {f_idx}")
        
        erase_rects = []
        pages_to_keep = set()

        for i, block in enumerate(ocr_blocks):
            page_idx = block['page']
            should_erase = False
            reason = ""

            if i <= h_idx:
                should_erase = False
                reason = "Header Area"
            elif h_idx < i < t_idx:
                should_erase = True
                reason = "Before Target"
            elif t_idx <= i < end_boundary_idx:
                should_erase = False
                reason = "Target Content"
            else:
                # After End Boundary
                if i >= f_idx:
                    should_erase = False
                    reason = "Footer Area"
                else:
                    should_erase = True
                    reason = "After Target (Before Footer)"

            if i in [h_idx, t_idx, end_boundary_idx, f_idx] or (i % 500 == 0):
                 logger.info(f"Block {i} (Page {page_idx}): {reason}. Text: {block['text'][:20]}...")

            if should_erase:
                bbox = fitz.Rect(block['bbox'])
                bbox = self._expand_rect(bbox)
                erase_rects.append((page_idx, bbox))
            else:
                pages_to_keep.add(page_idx)

        self._mark_content_pages_to_keep(pages_to_keep, ocr_blocks, t_idx, end_boundary_idx, f_idx)
        self._add_boundary_redactions(erase_rects, ocr_blocks, h_idx, t_idx, end_boundary_idx, f_idx)
        logger.info(f"--- Calculation Done ---")
        logger.info(f"Erase Rects: {len(erase_rects)}")
        logger.info(f"Pages to Keep: {sorted(list(pages_to_keep))}")
        return erase_rects, pages_to_keep

    def _adjust_footer_start_index(self, ocr_blocks, anchor_idx):
        footer_labels = [
            "出席",
            "参会人员",
            "参加会议人员",
            "列席",
            "请假",
            "记录",
            "分送",
        ]
        anchor_page = ocr_blocks[anchor_idx]['page']
        search_start = max(0, anchor_idx - 8)

        for i in range(anchor_idx, search_start - 1, -1):
            block = ocr_blocks[i]
            if block['page'] != anchor_page:
                break
            normalized = self._normalize_text(block['text'])
            for label in footer_labels:
                if self._normalize_text(label) in normalized:
                    if i != anchor_idx:
                        logger.info(f"Adjusted footer start from index {anchor_idx} to label index {i}: '{block['text']}'")
                    return i
        return anchor_idx

    def _mark_content_pages_to_keep(self, pages_to_keep, ocr_blocks, t_idx, end_boundary_idx, f_idx):
        if not ocr_blocks or t_idx < 0:
            return

        target_page = ocr_blocks[t_idx]['page']
        if 0 <= end_boundary_idx < len(ocr_blocks):
            content_end_page = ocr_blocks[end_boundary_idx]['page']
        elif 0 <= f_idx < len(ocr_blocks):
            content_end_page = ocr_blocks[f_idx]['page']
        else:
            content_end_page = target_page

        for page_idx in range(target_page, content_end_page + 1):
            pages_to_keep.add(page_idx)

        if 0 <= f_idx < len(ocr_blocks):
            last_ocr_page = ocr_blocks[-1]['page']
            for page_idx in range(ocr_blocks[f_idx]['page'], last_ocr_page + 1):
                pages_to_keep.add(page_idx)

    def _page_rect_from_block(self, block):
        width = block.get('page_width')
        height = block.get('page_height')
        if not width or not height:
            return None
        return fitz.Rect(0, 0, width, height)

    def _full_width_rect(self, block, y0, y1):
        page_rect = self._page_rect_from_block(block)
        if not page_rect:
            return None
        y0 = max(page_rect.y0, y0)
        y1 = min(page_rect.y1, y1)
        if y1 <= y0:
            return None
        return fitz.Rect(page_rect.x0, y0, page_rect.x1, y1)

    def _add_boundary_redactions(self, erase_rects, ocr_blocks, h_idx, t_idx, end_boundary_idx, f_idx):
        """补充整段边界涂白，处理扫描件中 OCR 漏块或跨行标题残留。"""
        if not ocr_blocks or t_idx < 0 or end_boundary_idx < 0 or end_boundary_idx >= len(ocr_blocks):
            return

        header_block = ocr_blocks[h_idx] if 0 <= h_idx < len(ocr_blocks) else None
        target_block = ocr_blocks[t_idx]
        target_rect = fitz.Rect(target_block['bbox'])
        if not header_block or target_block['page'] != header_block['page']:
            top_rect = self._full_width_rect(target_block, 0, target_rect.y0 - Config.BOUNDARY_SAFE_GAP)
            if top_rect:
                erase_rects.append((target_block['page'], top_rect))
                logger.info(f"Added top-of-target-page redaction on page {target_block['page']}: y={top_rect.y0:.1f}-{top_rect.y1:.1f}")
        else:
            header_rect = fitz.Rect(header_block['bbox'])
            top_rect = self._full_width_rect(target_block, header_rect.y1 + Config.PADDING_Y, target_rect.y0 - Config.BOUNDARY_SAFE_GAP)
            if top_rect:
                erase_rects.append((target_block['page'], top_rect))
                logger.info(f"Added before-target redaction on header page {target_block['page']}: y={top_rect.y0:.1f}-{top_rect.y1:.1f}")

        end_block = ocr_blocks[end_boundary_idx]
        end_rect = fitz.Rect(end_block['bbox'])
        footer_block = ocr_blocks[f_idx] if 0 <= f_idx < len(ocr_blocks) else None

        end_y0 = end_rect.y0 - Config.PADDING_Y
        erase_to_y = end_block.get('page_height', end_rect.y1)
        if footer_block and footer_block['page'] == end_block['page']:
            erase_to_y = fitz.Rect(footer_block['bbox']).y0 - Config.PADDING_Y

        boundary_rect = self._full_width_rect(end_block, end_y0, erase_to_y)
        if boundary_rect:
            erase_rects.append((end_block['page'], boundary_rect))
            logger.info(f"Added boundary redaction on page {end_block['page']}: y={boundary_rect.y0:.1f}-{boundary_rect.y1:.1f}")

        if not footer_block:
            return

        redacted_pages = set()
        for block in ocr_blocks:
            page_idx = block['page']
            if end_block['page'] < page_idx < footer_block['page'] and page_idx not in redacted_pages:
                page_rect = self._page_rect_from_block(block)
                if page_rect:
                    erase_rects.append((page_idx, page_rect))
                    redacted_pages.add(page_idx)
                    logger.info(f"Added full-page redaction on page {page_idx} between target and footer")

    def _clean_text_for_blank_check(self, text):
        """极致的文本清洗：去掉所有数字（页码）、空格、换行、标点。"""
        text = re.sub(r'\s+', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[\-—\.·•,，。、;；]', '', text)
        return text

    def _expand_rect(self, rect):
        rect = fitz.Rect(rect)
        rect.x0 -= Config.PADDING_X
        rect.y0 -= Config.PADDING_Y
        rect.x1 += Config.PADDING_X
        rect.y1 += Config.PADDING_Y
        return rect

    def _clip_rect_to_page(self, rect, page):
        clipped = fitz.Rect(rect) * page.derotation_matrix
        page_box = page.cropbox
        clipped.x0 = max(page_box.x0, clipped.x0)
        clipped.y0 = max(page_box.y0, clipped.y0)
        clipped.x1 = min(page_box.x1, clipped.x1)
        clipped.y1 = min(page_box.y1, clipped.y1)
        return clipped if not clipped.is_empty and clipped.get_area() > 0 else None

    def _can_merge_rects(self, left, right):
        vertical_overlap = min(left.y1, right.y1) - max(left.y0, right.y0)
        min_height = max(1, min(left.height, right.height))
        same_line = vertical_overlap >= min_height * 0.45 or (
            abs(left.y0 - right.y0) <= Config.MERGE_RECT_GAP and
            abs(left.y1 - right.y1) <= Config.MERGE_RECT_GAP
        )
        close_enough = right.x0 <= left.x1 + Config.MERGE_RECT_GAP
        return same_line and close_enough

    def _merge_rects_for_page(self, rects):
        if not rects:
            return []

        rects = sorted((fitz.Rect(rect) for rect in rects), key=lambda r: (round(r.y0 / 6), r.x0))
        merged = []
        current = rects[0]
        for rect in rects[1:]:
            if self._can_merge_rects(current, rect):
                current.include_rect(rect)
            else:
                merged.append(current)
                current = rect
        merged.append(current)
        return merged

    def _group_and_merge_rects(self, doc, erase_rects):
        grouped = {}
        for page_idx, rect in erase_rects:
            if page_idx < 0 or page_idx >= doc.page_count:
                continue
            clipped = self._clip_rect_to_page(rect, doc[page_idx])
            if clipped:
                grouped.setdefault(page_idx, []).append(clipped)

        merged = {}
        for page_idx, rects in grouped.items():
            merged[page_idx] = self._merge_rects_for_page(rects)
        return merged

    def _apply_redactions_to_page(self, page, rects):
        if not rects:
            return 0

        for rect in rects:
            page.add_redact_annot(rect, fill=Config.WHITEOUT_COLOR)

        try:
            page.apply_redactions(
                images=getattr(fitz, "PDF_REDACT_IMAGE_PIXELS", 2),
                graphics=getattr(fitz, "PDF_REDACT_LINE_ART_REMOVE_IF_COVERED", 2),
                text=getattr(fitz, "PDF_REDACT_TEXT_REMOVE", 0)
            )
        except TypeError:
            try:
                page.apply_redactions(images=getattr(fitz, "PDF_REDACT_IMAGE_PIXELS", 2))
            except TypeError:
                page.apply_redactions()

        for rect in rects:
            page.draw_rect(rect, color=Config.WHITEOUT_COLOR, fill=Config.WHITEOUT_COLOR, overlay=True)
        return len(rects)

    def _apply_whiteout_and_save(self, input_pdf, output_pdf, erase_rects, pages_to_keep):
        logger.info("--- Applying redactions and deleting blank / unrelated pages ---")
        doc = fitz.open(input_pdf)
        pages_modified_count = 0

        merged_rects = self._group_and_merge_rects(doc, erase_rects)
        original_rect_count = len(erase_rects)
        merged_rect_count = sum(len(rects) for rects in merged_rects.values())
        logger.info(f"Redaction rects merged: {original_rect_count} -> {merged_rect_count}")

        for page_idx, rects in merged_rects.items():
            pages_modified_count += self._apply_redactions_to_page(doc[page_idx], rects)
            
        total_pages = doc.page_count
        deleted_pages_count = 0
        
        for i in range(total_pages - 1, -1, -1):
            is_marked_keep = (i in pages_to_keep)
            has_images = bool(doc[i].get_images())
            raw_text = doc[i].get_text("text").strip()
            
            clean_text = self._clean_text_for_blank_check(raw_text)
            clean_len = len(clean_text)
            
            is_unrelated_page = Config.DELETE_UNMARKED_PAGES and not is_marked_keep
            is_blank_candidate = (not is_marked_keep) and (not has_images) and (clean_len < 3)
            
            logger.info(f"[Page {i} Check] MarkedKeep: {is_marked_keep}, HasImg: {has_images}, CleanTextLen: {clean_len}, RawTextPreview: {raw_text[:30].replace(chr(10), ' ')}")
            
            if is_unrelated_page or is_blank_candidate:
                doc.delete_page(i)
                deleted_pages_count += 1
                reason = "No kept OCR content" if is_unrelated_page else "Blank check passed"
                logger.info(f"  >>> DELETED Page {i} (Reason: {reason})")
            else:
                if is_marked_keep:
                    logger.info(f"  >>> KEPT Page {i} (Reason: Marked as Content)")
                elif has_images:
                    logger.info(f"  >>> KEPT Page {i} (Reason: Contains Images)")
                elif clean_len >= 3:
                    logger.info(f"  >>> KEPT Page {i} (Reason: Contains enough valid text: '{clean_text[:10]}...')")
        
        doc.save(output_pdf, garbage=4, deflate=True)
        doc.close()
        logger.info(f"--- Process Complete ---")
        return pages_modified_count, deleted_pages_count

# --- GUI 类 ---
class PDFCleanerApp:
    BG = "#f5f7fb"
    CARD = "#ffffff"
    TEXT = "#172033"
    MUTED = "#6b7280"
    ACCENT = "#2563eb"
    ACCENT_DARK = "#1d4ed8"
    BORDER = "#dbe3ef"

    def __init__(self, root):
        self.root = root
        self.root.title(f"TopicKeeper v{__version__}")
        self.root.geometry("820x620")
        self.root.minsize(760, 560)
        self.root.configure(bg=self.BG)
        self.main_thread = threading.current_thread()

        self.tool = None
        self.ocr_blocks = None
        self.analysis_result = None
        self.erase_rects = None
        self.pages_to_keep = None
        self.input_pdf_path = None
        self.output_pdf_path = None
        self.user_keyword = ""
        self.preprocess_path = None
        self.preprocess_event = None
        self.preprocess_error = None
        self.preprocess_running = False
        self.is_processing = False

        self.file_var = tk.StringVar()
        self.keyword_var = tk.StringVar()
        self.status_var = tk.StringVar(value="就绪")
        self.detail_var = tk.StringVar(value="选择 PDF，输入想保留的议题关键词，然后开始处理。")
        self.confirm_summary_var = tk.StringVar(value="")

        self._build_styles()
        self._build_ui()

    def _build_styles(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", font=("Helvetica Neue", 12), background=self.BG, foreground=self.TEXT)
        style.configure("Card.TFrame", background=self.CARD, relief="flat")
        style.configure("Hero.TLabel", background=self.BG, foreground=self.TEXT, font=("Helvetica Neue", 24, "bold"))
        style.configure("Subtitle.TLabel", background=self.BG, foreground=self.MUTED, font=("Helvetica Neue", 12))
        style.configure("Section.TLabel", background=self.CARD, foreground=self.TEXT, font=("Helvetica Neue", 13, "bold"))
        style.configure("Hint.TLabel", background=self.CARD, foreground=self.MUTED, font=("Helvetica Neue", 10))
        style.configure("Status.TLabel", background=self.CARD, foreground=self.ACCENT, font=("Helvetica Neue", 12, "bold"))
        style.configure("Muted.TLabel", background=self.CARD, foreground=self.MUTED, font=("Helvetica Neue", 10))
        style.configure("Primary.TButton", background=self.ACCENT, foreground="white", borderwidth=0, focusthickness=0, padding=(18, 12))
        style.map("Primary.TButton", background=[("active", self.ACCENT_DARK), ("disabled", "#9bb7ed")])
        style.configure("Ghost.TButton", background=self.CARD, foreground=self.TEXT, borderwidth=1, relief="solid", padding=(12, 9))
        style.map("Ghost.TButton", background=[("active", "#edf3ff")])
        style.configure("Danger.TButton", background="#fee2e2", foreground="#991b1b", borderwidth=1, relief="solid", padding=(12, 9))
        style.map("Danger.TButton", background=[("active", "#fecaca")])
        style.configure("TEntry", fieldbackground="#fbfdff", bordercolor=self.BORDER, lightcolor=self.BORDER, darkcolor=self.BORDER, padding=8)
        style.configure("Horizontal.TProgressbar", troughcolor="#edf1f7", background=self.ACCENT, bordercolor="#edf1f7", lightcolor=self.ACCENT, darkcolor=self.ACCENT)

    def _build_ui(self):
        shell = ttk.Frame(self.root, padding=(28, 24, 28, 20), style="TFrame")
        shell.pack(fill=tk.BOTH, expand=True)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(2, weight=1)

        header = ttk.Frame(shell, style="TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 18))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="TopicKeeper", style="Hero.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="保留指定会议议题，自动涂白无关内容，并在生成后检查结构完整性。",
            style="Subtitle.TLabel"
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Label(header, text=f"v{__version__}", style="Subtitle.TLabel").grid(row=0, column=1, sticky="e")

        work_card = self._card(shell)
        work_card.grid(row=1, column=0, sticky="ew", pady=(0, 16))
        work_card.columnconfigure(1, weight=1)

        ttk.Label(work_card, text="选择文件", style="Section.TLabel").grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(work_card, text="支持扫描版 PDF。输出文件会保存在原文件旁边，后缀为 _cleaned。", style="Hint.TLabel").grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 12))
        self.file_entry = ttk.Entry(work_card, textvariable=self.file_var)
        self.file_entry.grid(row=2, column=0, columnspan=2, sticky="ew", padx=(0, 10))
        ttk.Button(work_card, text="浏览 PDF", command=self.browse_file, style="Ghost.TButton").grid(row=2, column=2, sticky="ew")

        ttk.Label(work_card, text="关键词", style="Section.TLabel").grid(row=3, column=0, columnspan=3, sticky="w", pady=(20, 0))
        ttk.Label(work_card, text="输入想保留的议题关键词，例如：东升变更、预留上盖费用。", style="Hint.TLabel").grid(row=4, column=0, columnspan=3, sticky="w", pady=(4, 12))
        self.keyword_entry = ttk.Entry(work_card, textvariable=self.keyword_var)
        self.keyword_entry.grid(row=5, column=0, columnspan=3, sticky="ew")
        self.keyword_entry.bind("<Return>", lambda _event: self.start_processing())

        action_row = ttk.Frame(work_card, style="Card.TFrame")
        action_row.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(22, 0))
        action_row.columnconfigure(0, weight=1)
        self.start_btn = ttk.Button(action_row, text="开始处理", command=self.start_processing, style="Primary.TButton")
        self.start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.open_output_btn = ttk.Button(action_row, text="打开输出文件", command=self.open_output_file, style="Ghost.TButton", state=tk.DISABLED)
        self.open_output_btn.grid(row=0, column=1, sticky="ew", padx=(0, 10))
        ttk.Button(action_row, text="查看日志", command=self.open_log, style="Ghost.TButton").grid(row=0, column=2, sticky="ew")

        bottom = ttk.Frame(shell, style="TFrame")
        bottom.grid(row=2, column=0, sticky="nsew")
        bottom.columnconfigure(0, weight=2)
        bottom.columnconfigure(1, weight=1)
        bottom.rowconfigure(0, weight=1)

        status_card = self._card(bottom)
        status_card.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        status_card.columnconfigure(0, weight=1)
        ttk.Label(status_card, text="处理状态", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(status_card, textvariable=self.status_var, style="Status.TLabel").grid(row=1, column=0, sticky="w", pady=(12, 4))
        ttk.Label(status_card, textvariable=self.detail_var, style="Muted.TLabel", wraplength=430, justify=tk.LEFT).grid(row=2, column=0, sticky="ew")
        self.progress = ttk.Progressbar(status_card, mode="indeterminate")
        self.progress.grid(row=3, column=0, sticky="ew", pady=(18, 12))
        self.result_text = tk.Text(
            status_card,
            height=8,
            wrap=tk.WORD,
            relief=tk.FLAT,
            bg="#fbfdff",
            fg=self.TEXT,
            insertbackground=self.TEXT,
            padx=12,
            pady=10,
            font=("Helvetica Neue", 11)
        )
        self.result_text.grid(row=4, column=0, sticky="nsew")
        self.confirm_frame = ttk.Frame(status_card, style="Card.TFrame")
        self.confirm_frame.grid(row=5, column=0, sticky="ew", pady=(12, 0))
        self.confirm_frame.columnconfigure(0, weight=1)
        self.confirm_summary = ttk.Label(
            self.confirm_frame,
            textvariable=self.confirm_summary_var,
            style="Section.TLabel",
            wraplength=650,
            justify=tk.LEFT
        )
        self.confirm_summary.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        self.confirm_btn = ttk.Button(self.confirm_frame, text="确认生成 PDF", command=self.confirm_generation, style="Primary.TButton")
        self.confirm_btn.grid(row=1, column=0, sticky="ew", padx=(0, 10))
        self.cancel_btn = ttk.Button(self.confirm_frame, text="重新选择", command=self.cancel_confirmation, style="Danger.TButton")
        self.cancel_btn.grid(row=1, column=1, sticky="ew")
        self.confirm_frame.grid_remove()
        status_card.rowconfigure(4, weight=1)
        self._set_result("暂无输出。")

        info_card = self._card(bottom)
        info_card.grid(row=0, column=1, sticky="nsew")
        ttk.Label(info_card, text="流程", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        steps = [
            ("1", "OCR 识别文本和位置"),
            ("2", "本地模型定位议题边界"),
            ("3", "涂白无关内容并删除空页"),
            ("4", "生成后结构完整性检查")
        ]
        for row, (num, text) in enumerate(steps, start=1):
            ttk.Label(info_card, text=num, style="Status.TLabel", width=2).grid(row=row, column=0, sticky="nw", pady=(14, 0))
            ttk.Label(info_card, text=text, style="Muted.TLabel", wraplength=220).grid(row=row, column=1, sticky="w", pady=(14, 0))
        ttk.Label(info_card, text="本地 API", style="Section.TLabel").grid(row=6, column=0, columnspan=2, sticky="w", pady=(28, 0))
        api_text = f"{Config.LLM_BASE_URL}\n{Config.LLM_MODEL}"
        ttk.Label(info_card, text=api_text, style="Muted.TLabel", wraplength=240).grid(row=7, column=0, columnspan=2, sticky="w", pady=(8, 0))

    def _card(self, parent):
        frame = ttk.Frame(parent, padding=20, style="Card.TFrame")
        return frame

    def _open_path(self, path):
        try:
            if os.name == 'nt':
                os.startfile(path)
            elif os.name == 'posix':
                subprocess.call(['open', path] if sys.platform == 'darwin' else ['xdg-open', path])
        except Exception as e:
            self.show_inline_error("无法打开文件", f"{e}\n路径: {path}")

    def browse_file(self):
        filename = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if filename:
            self.file_var.set(filename)
            self.input_pdf_path = filename
            self.output_pdf_path = None
            self.ocr_blocks = None
            self.analysis_result = None
            self.open_output_btn.config(state=tk.DISABLED)
            self._hide_confirmation()
            if Config.PREPROCESS_AFTER_FILE_SELECT:
                self._set_result("已选择文件，正在后台预处理 OCR：\n" + filename)
                self.start_preprocessing(filename)
            else:
                self.update_status("已选择文件", os.path.basename(filename))
                self._set_result("等待处理：\n" + filename)

    def open_log(self):
        if not os.path.exists(log_file):
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("Log file created.\n")
        self._open_path(log_file)

    def open_output_file(self):
        if self.output_pdf_path and os.path.exists(self.output_pdf_path):
            self._open_path(self.output_pdf_path)
        else:
            self.show_inline_error("还没有输出文件", "处理完成后，这里会启用“打开输出文件”。")

    def start_preprocessing(self, pdf_path):
        self.preprocess_path = pdf_path
        self.preprocess_error = None
        self.preprocess_running = True
        self.preprocess_event = threading.Event()
        self.progress.start(12)
        self.update_status("正在预处理 OCR", "你可以继续输入关键词；预处理完成后，点击开始会更快进入模型分析。")
        thread = threading.Thread(target=self.run_preprocessing_thread, args=(pdf_path, self.preprocess_event), daemon=True)
        thread.start()

    def run_preprocessing_thread(self, pdf_path, done_event):
        try:
            logger.info(f"--- Background preprocessing started: {pdf_path} ---")
            tool = SemanticWhiteoutTool()
            ocr_blocks = tool._extract_layout_ocr(pdf_path)
            if not ocr_blocks:
                raise ValueError("OCR 没有识别到可用文字")

            if self.input_pdf_path == pdf_path and self.preprocess_event is done_event:
                self.tool = tool
                self.ocr_blocks = ocr_blocks
                self.preprocess_error = None
                logger.info(f"--- Background preprocessing completed: {len(ocr_blocks)} blocks ---")
                self.root.after(0, lambda: self.on_preprocessing_finished(pdf_path, len(ocr_blocks), None))
            else:
                logger.info("Background preprocessing result ignored because selected file changed.")
        except Exception as e:
            logger.exception("Background preprocessing failed")
            if self.input_pdf_path == pdf_path and self.preprocess_event is done_event:
                self.preprocess_error = e
                self.root.after(0, lambda err=str(e): self.on_preprocessing_finished(pdf_path, 0, err))
        finally:
            if self.input_pdf_path == pdf_path and self.preprocess_event is done_event:
                self.preprocess_running = False
                done_event.set()

    def on_preprocessing_finished(self, pdf_path, block_count, error):
        if self.input_pdf_path != pdf_path:
            return
        if not self.is_processing:
            self.progress.stop()
        if error:
            self.update_status("预处理失败", error)
            self._set_result(f"预处理失败\n\n{error}\n\n你可以换一个 PDF，或查看日志定位原因。")
            return
        self.update_status("预处理完成", f"OCR 已提前完成，识别到 {block_count} 个文本块。输入关键词后可直接开始分析。")
        self._set_result("预处理完成，可以输入关键词并开始处理：\n" + pdf_path)

    def wait_for_preprocessing_if_available(self, pdf_path):
        event = self.preprocess_event
        if self.preprocess_path != pdf_path or event is None:
            return False

        if not event.is_set():
            self.update_status("等待预处理完成", "OCR 正在后台收尾，完成后会直接进入模型分析。")
            event.wait()

        if self.preprocess_path != pdf_path:
            return False
        if self.preprocess_error:
            raise self.preprocess_error
        return bool(self.ocr_blocks)

    def _set_result(self, text):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.config(state=tk.DISABLED)

    def show_inline_error(self, title, detail):
        self.update_status(title, detail)
        self._set_result(f"{title}\n\n{detail}")

    def _hide_confirmation(self):
        self.confirm_summary_var.set("")
        self.confirm_frame.grid_remove()

    def _show_confirmation(self, preview, summary):
        self.progress.stop()
        self.confirm_summary_var.set(summary)
        self.confirm_frame.grid()
        self.start_btn.config(state=tk.DISABLED)
        self.open_output_btn.config(state=tk.DISABLED)
        self._set_result(preview)

    def _set_busy(self, busy):
        self.is_processing = busy
        if busy:
            self._hide_confirmation()
            self.progress.start(12)
            self.start_btn.config(state=tk.DISABLED)
            self.open_output_btn.config(state=tk.DISABLED)
        else:
            if not self.preprocess_running:
                self.progress.stop()
            self.start_btn.config(state=tk.NORMAL)
            if self.output_pdf_path and os.path.exists(self.output_pdf_path):
                self.open_output_btn.config(state=tk.NORMAL)

    def _set_status_now(self, message, detail=None):
        self.status_var.set(message)
        if detail is not None:
            self.detail_var.set(detail)

    def update_status(self, message, detail=None):
        if threading.current_thread() is self.main_thread:
            self._set_status_now(message, detail)
            self.root.update_idletasks()
        else:
            self.root.after(0, lambda: self._set_status_now(message, detail))

    def start_processing(self):
        self.input_pdf_path = self.file_var.get().strip()
        if not self.input_pdf_path or not os.path.exists(self.input_pdf_path):
            self.show_inline_error("请先选择 PDF 文件", "点击“浏览 PDF”，选择需要处理的会议纪要。")
            return
        keyword = self.keyword_var.get().strip()
        if not keyword:
            self.show_inline_error("请输入关键词", "例如：东升变更、预留上盖费用。")
            return
        self.user_keyword = keyword
        self.output_pdf_path = None
        self._set_busy(True)
        if self.preprocess_path == self.input_pdf_path and self.preprocess_event is not None:
            self._set_result("正在复用后台预处理结果，请稍候。")
        else:
            self._set_result("正在准备处理，请稍候。")
        thread = threading.Thread(target=self.run_analysis_thread, args=(keyword,), daemon=True)
        thread.start()

    def run_analysis_thread(self, keyword):
        try:
            pdf_path = self.input_pdf_path
            used_preprocess = self.wait_for_preprocessing_if_available(pdf_path)
            if used_preprocess:
                self.update_status("正在分析议题", "已复用预处理 OCR 结果，本地大模型正在定位议题边界。")
            else:
                self.update_status("正在初始化", "加载 OCR 引擎和本地模型配置。")
                self.tool = SemanticWhiteoutTool()
                self.update_status("正在 OCR 识别", "扫描件会稍慢一些，正在提取文字和坐标。")
                self.ocr_blocks = self.tool._extract_layout_ocr(pdf_path)
            if not self.ocr_blocks:
                self.root.after(0, lambda: self.show_inline_error("OCR 失败", "没有识别到可用文字，请检查 PDF 是否能正常打开。"))
                self.root.after(0, self.reset_ui)
                return

            if not used_preprocess:
                self.update_status("正在分析议题", "本地大模型正在定位目标议题、下一议题和结尾人员区。")
            self.analysis_result = self.tool._analyze_structure_with_llm(self.ocr_blocks, keyword)
            self.root.after(0, self.ask_confirmation)

        except Exception as e:
            logger.exception("Error")
            err = str(e)
            self.root.after(0, lambda: self.show_inline_error("处理失败", err))
            self.root.after(0, self.reset_ui)

    def ask_confirmation(self):
        target_title = self.analysis_result.get("target_title", "")
        next_title = self.analysis_result.get("next_topic_title", "")

        summary = f"确认保留：{target_title or '未识别到标题'}"
        preview = "请确认下面的识别结果。\n\n"
        preview += f"将保留的目标议题：\n{target_title or '未识别到标题'}\n\n"
        if next_title and next_title != "NULL":
            preview += f"从这个后接议题开始涂白：\n{next_title}\n\n"
            summary += f"\n涂白从：{next_title}"
        else:
            preview += "后接议题：未明确识别，程序会用规则辅助判断。\n\n"
            summary += "\n后接议题：未明确识别"
        preview += "请在下方确认是否生成涂白后的 PDF。"
        self.update_status("请确认识别结果", summary.replace("\n", "；"))
        self._show_confirmation(preview, summary)

    def confirm_generation(self):
        self.update_status("正在生成 PDF", "计算涂白区域并写入新文件。")
        self._set_busy(True)
        thread = threading.Thread(target=self.run_final_process_thread, daemon=True)
        thread.start()

    def cancel_confirmation(self):
        self._hide_confirmation()
        self.update_status("已取消", "可以修改关键词后重新开始。")
        self.reset_ui()

    def run_final_process_thread(self):
        global output_path
        try:
            self.erase_rects, self.pages_to_keep = self.tool._calculate_erase_bbox(self.ocr_blocks, self.analysis_result)

            base, ext = os.path.splitext(self.input_pdf_path)
            output_path = base + "_cleaned" + ext

            mod_count, del_count = self.tool._apply_whiteout_and_save(
                self.input_pdf_path,
                output_path,
                self.erase_rects,
                self.pages_to_keep
            )

            check_report = None
            if Config.POST_GENERATION_STRUCTURE_CHECK:
                self.update_status("正在检查输出结构", "复查标题、纪要头、目标议题和人员/footer 是否完整。")
                check_report = self.tool._validate_output_structure(
                    self.input_pdf_path,
                    output_path,
                    self.ocr_blocks,
                    self.analysis_result,
                    self.user_keyword
                )

            self.output_pdf_path = output_path
            result = [
                "处理完成",
                "",
                f"输出文件：{output_path}",
                f"涂白区域：{mod_count} 处",
                f"删除空白/无关页：{del_count} 页"
            ]
            if check_report:
                if check_report["passed"] and not check_report["warnings"]:
                    result.append("结构检查：通过")
                else:
                    warnings = check_report["warnings"][:4]
                    result.append("结构检查：有警告")
                    result.extend(f"- {warning}" for warning in warnings)

            self.root.after(0, lambda: self._set_result("\n".join(result)))
            self.update_status("完成", "可以打开输出文件检查效果。")

        except Exception as e:
            logger.exception("Final error")
            err = str(e)
            self.root.after(0, lambda: self.show_inline_error("生成失败", err))
        finally:
            self.root.after(0, self.reset_ui)

    def reset_ui(self):
        self._hide_confirmation()
        self._set_busy(False)

if __name__ == "__main__":
    root = tk.Tk()
    app = PDFCleanerApp(root)
    root.mainloop()
