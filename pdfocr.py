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
from tkinter import filedialog, messagebox
import threading
import sys
import subprocess

__version__ = "3.0"

# --- 配置日志 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(SCRIPT_DIR, "pdf_whiteout.log")

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
    def __init__(self, root):
        self.root = root
        self.root.title(f"TopicKeeper v{__version__}")
        self.root.geometry("600x420")
        
        self.tool = None
        self.ocr_blocks = None
        self.analysis_result = None
        self.erase_rects = None
        self.pages_to_keep = None
        self.input_pdf_path = None

        tk.Label(root, text="1. 选择 PDF 文件:", font=("Arial", 10, "bold")).pack(pady=(15, 5))
        file_frame = tk.Frame(root)
        file_frame.pack(pady=5)
        self.file_entry = tk.Entry(file_frame, width=50)
        self.file_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(file_frame, text="浏览...", command=self.browse_file).pack(side=tk.LEFT)
        
        tk.Label(root, text="2. 输入关键词 (如: 议题二):", font=("Arial", 10, "bold")).pack(pady=(15, 5))
        self.keyword_entry = tk.Entry(root, width=30, font=("Arial", 10))
        self.keyword_entry.pack(pady=5)
        
        self.start_btn = tk.Button(root, text="3. 开始处理", command=self.start_processing, bg="#e1f5fe", height=2)
        self.start_btn.pack(pady=20, fill=tk.X, padx=50)
        
        self.status_label = tk.Label(root, text="就绪", fg="blue", font=("Arial", 9))
        self.status_label.pack(side=tk.BOTTOM, pady=10)
        
        tk.Button(root, text="打开详细日志", command=self.open_log, bg="#ffccbc").pack(side=tk.BOTTOM, pady=5)

    def browse_file(self):
        filename = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if filename:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, filename)
            self.input_pdf_path = filename

    def open_log(self):
        try:
            if not os.path.exists(log_file):
                with open(log_file, 'w', encoding='utf-8') as f:
                    f.write("Log file created.\n")
            if os.name == 'nt': os.startfile(log_file)
            elif os.name == 'posix': subprocess.call(['open', log_file] if sys.platform == 'darwin' else ['xdg-open', log_file])
        except Exception as e:
            messagebox.showerror("错误", f"无法打开日志文件: {e}\n路径: {log_file}")

    def update_status(self, message):
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def start_processing(self):
        if not self.input_pdf_path or not os.path.exists(self.input_pdf_path):
            messagebox.showerror("错误", "请先选择 PDF 文件")
            return
        keyword = self.keyword_entry.get().strip()
        if not keyword:
            messagebox.showerror("错误", "请输入关键词")
            return
        self.start_btn.config(state=tk.DISABLED)
        thread = threading.Thread(target=self.run_analysis_thread, args=(keyword,))
        thread.start()

    def run_analysis_thread(self, keyword):
        try:
            self.update_status("正在初始化...")
            self.tool = SemanticWhiteoutTool()
            self.update_status("正在进行 OCR (较慢)...")
            self.ocr_blocks = self.tool._extract_layout_ocr(self.input_pdf_path)
            if not self.ocr_blocks:
                self.root.after(0, lambda: messagebox.showerror("错误", "OCR失败"))
                self.reset_ui()
                return

            self.update_status("正在 LLM 分析...")
            self.analysis_result = self.tool._analyze_structure_with_llm(self.ocr_blocks, keyword)
            
            self.root.after(0, self.ask_confirmation)
            
        except Exception as e:
            logger.exception("Error")
            self.root.after(0, lambda: messagebox.showerror("处理失败", str(e)))
            self.reset_ui()

    def ask_confirmation(self):
        target_title = self.analysis_result.get("target_title", "")
        next_title = self.analysis_result.get("next_topic_title", "")
        
        msg = f"识别议题：『 {target_title} 』\n"
        if next_title and next_title != "NULL":
            msg += f"后接：{next_title}\n"
        else:
            msg += "系统将自动预测下一议题\n"
            
        msg += "\n确认开始清洗？"
        
        if messagebox.askyesno("确认", msg):
            self.update_status("正在生成 PDF...")
            thread = threading.Thread(target=self.run_final_process_thread)
            thread.start()
        else:
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
            
            msg = f"处理完成！\n\n路径: {output_path}\n修改: {mod_count} 处\n删除空白页: {del_count} 页"
            self.root.after(0, lambda: messagebox.showinfo("成功", msg))
            self.update_status("完成")
            
        except Exception as e:
            logger.exception("Final error")
            self.root.after(0, lambda: messagebox.showerror("失败", str(e)))
        finally:
            self.root.after(0, self.reset_ui)

    def reset_ui(self):
        self.start_btn.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = PDFCleanerApp(root)
    root.mainloop()
