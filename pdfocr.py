import logging
import json
import fitz  # PyMuPDF
from openai import OpenAI
from rapidocr_onnxruntime import RapidOCR
import re
import os
# --- 配置日志 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("pdf_whiteout.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
# --- 配置类 ---
class Config:
    # LM Studio API 配置
    LLM_BASE_URL = "http://127.0.0.1:1234/v1"
    LLM_API_KEY = "lm-studio"
    LLM_MODEL = "qwen/qwen3-4b-2507"  
    # 涂白参数
    WHITEOUT_COLOR = (1, 1, 1)  # RGB 白色
    PADDING_Y = 5  # 垂直方向涂白扩充像素
    PADDING_X = 5  # 水平方向扩充
    # OCR 配置
    USE_RAPIDOCR = True
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
    def _extract_layout_ocr(self, pdf_path):
        logger.info(f"Starting OCR for {pdf_path}...")
        if not os.path.exists(pdf_path):
            logger.error(f"File not found: {pdf_path}")
            return []
        doc = fitz.open(pdf_path)
        all_blocks = []
        for page_index in range(len(doc)):
            page = doc[page_index]
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
            result, _ = self.ocr_engine(img_bytes)
            if result:
                img_w, img_h = pix.width, pix.height
                pdf_w, pdf_h = page.rect.width, page.rect.height
                scale_x = pdf_w / img_w
                scale_y = pdf_h / img_h
                for line in result:
                    try:
                        if len(line) == 3:
                            points = line[0]
                            text = line[1]
                            xs = [p[0] for p in points]
                            ys = [p[1] for p in points]
                            x0, y0 = min(xs), min(ys)
                            x2, y2 = max(xs), max(ys)
                        elif len(line) == 5:
                            x0, y0, x2, y2, text = line
                        elif len(line) == 6:
                            x0, y0, x2, y2, text, _ = line
                        else:
                            continue
                        pdf_bbox = (x0 * scale_x, y0 * scale_y, x2 * scale_x, y2 * scale_y)
                        all_blocks.append({
                            'page': page_index,
                            'text': text,
                            'bbox': pdf_bbox
                        })
                    except Exception:
                        continue
        doc.close()
        logger.info(f"OCR completed. Extracted {len(all_blocks)} text blocks.")
        return all_blocks
    def _analyze_structure_with_llm(self, ocr_blocks, user_keyword):
        logger.info("Sending data to LLM for semantic analysis...")
        # 限制送入 LLM 的文本长度
        text_stream = ""
        for i, block in enumerate(ocr_blocks):
            if i > 2000: break
            text_stream += block['text'] + "\n"
        prompt = f"""
你是一个专业的会议纪要分析助手。
**用户输入的关键词**: "{user_keyword}"
**文本内容**:
{text_stream}
**任务定义**:
1. **目标议题**: 找出用户关键词对应的议题。提取该议题的**完整标题**。
2. **下一个议题**: 提取紧跟在目标议题后面的那个议题的**完整标题**。如果目标议题是最后一个，则返回 "NULL"。
3. **纪要结尾**: 提取“出席”或“记录”那一行的一小段独特文字。
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
            json_str = re.sub(r'```json|```', '', content).strip()
            analysis_result = json.loads(json_str)
            if not analysis_result.get("target_title"):
                raise ValueError("LLM failed to identify target_title")
            return analysis_result
        except Exception as e:
            logger.error(f"LLM API Error or Parsing Error: {e}")
            raise
    def _confirm_with_user(self, analysis_result):
        target_title = analysis_result.get("target_title", "")
        next_title = analysis_result.get("next_topic_title", "")
        print(f"\n--------------------------------------------------")
        print(f"识别到您想保留的议题：")
        print(f"{target_title}")
        if next_title and next_title != "NULL":
            print(f"(此议题后接: {next_title})")
        else:
            print(f"(此议题为最后一个议题)")
        print(f"--------------------------------------------------")
        try:
            user_confirm = input("确认保留该议题（并涂白其他无关内容）？ => ")
        except EOFError:
            return False
        if user_confirm.strip().lower() in ['y', 'yes', '是', '确认', 'ok']:
            return True
        return False
    def _find_block_index_flexible(self, ocr_blocks, text_snippet):
        """模糊搜索"""
        if not text_snippet or text_snippet == "NULL":
            return -1
        for i, block in enumerate(ocr_blocks):
            if text_snippet in block['text']:
                return i
        # 尝试匹配前10个字
        if len(text_snippet) > 6:
            short_snippet = text_snippet[:10]
            for i, block in enumerate(ocr_blocks):
                if short_snippet in block['text']:
                    return i
        return -1
    def _calculate_erase_bbox(self, ocr_blocks, analysis_result):
        """
        修复逻辑：
        1. Header -> 找冒号
        2. Target Start -> 找 target_title
        3. Target End -> 找 next_topic_title (如果NULL则用Footer)
        4. Footer -> 找 footer_anchor
        5. 涂白逻辑：必须保留 Footer 之后的内容。
        """
        logger.info("Calculating coordinates for whiteout...")
        # 1. Header 终点 (找冒号)
        h_idx = -1
        for i, block in enumerate(ocr_blocks):
            if i < 20 and "：" in block['text']:
                h_idx = i
                logger.info(f"Found colon at index {i}: '{block['text']}'")
                break
        if h_idx == -1: h_idx = 0
        # 2. 获取位置
        target_title = analysis_result.get("target_title")
        next_title_str = analysis_result.get("next_topic_title")
        footer_text = analysis_result.get("footer_anchor")
        t_idx = self._find_block_index_flexible(ocr_blocks, target_title)
        # 确定结束边界
        end_boundary_idx = -1
        if next_title_str and next_title_str != "NULL":
            end_boundary_idx = self._find_block_index_flexible(ocr_blocks, next_title_str)
            logger.info(f"Found next topic boundary at index: {end_boundary_idx}")
        else:
            # 如果没有下一个议题，边界默认为 Footer（稍后计算）
            pass 
        # 3. 【关键修复】无条件查找 Footer 位置
        f_idx = self._find_block_index_flexible(ocr_blocks, footer_text)
        if f_idx == -1:
            logger.warning("Footer anchor not found. Assuming footer is at the very end.")
            f_idx = len(ocr_blocks) # 如果找不到 Footer，假设在文档最末尾，防止误删
        else:
            logger.info(f"Found footer at index: {f_idx}")
        # 如果没有下一个议题，则边界就是 Footer
        if end_boundary_idx == -1:
            end_boundary_idx = f_idx
        if t_idx == -1:
            logger.error("Cannot locate Target Topic. Aborting.")
            return []
        logger.info(f"Final Indices -> Header: {h_idx}, Target Start: {t_idx}, Target End: {end_boundary_idx}, Footer: {f_idx}")
        erase_rects = [] 
        for i, block in enumerate(ocr_blocks):
            bbox = fitz.Rect(block['bbox'])
            page_idx = block['page']
            should_erase = False
            # 区域判断逻辑
            if i <= h_idx:
                # Header 区域 -> 保留
                should_erase = False
            elif h_idx < i < t_idx:
                # Header 之后，Target 之前 (上一议题) -> 涂白
                should_erase = True
            elif t_idx <= i < end_boundary_idx:
                # Target 范围内 -> 保留
                should_erase = False
            else:
                # i >= end_boundary_idx (目标议题之后的所有内容)
                # 这里包含了：下一个议题、下下个议题... 以及 Footer
                # 逻辑：如果是 Footer 及其之后 -> 保留；否则 -> 涂白
                if i >= f_idx:
                    should_erase = False # Footer 保护
                else:
                    should_erase = True  # 这里的内容是后续的其他无关议题，涂白
            if should_erase:
                bbox.y0 -= Config.PADDING_Y
                bbox.y1 += Config.PADDING_Y
                bbox.x0 -= Config.PADDING_X
                bbox.x1 += Config.PADDING_X
                erase_rects.append((page_idx, bbox))
        logger.info(f"Calculated {len(erase_rects)} blocks to erase.")
        return erase_rects
    def process_pdf(self, pdf_path, output_path, user_keyword):
        try:
            ocr_blocks = self._extract_layout_ocr(pdf_path)
            if not ocr_blocks:
                logger.error("No text extracted.")
                return
            try:
                analysis_result = self._analyze_structure_with_llm(ocr_blocks, user_keyword)
            except Exception as e:
                print(f"LLM 处理失败: {e}")
                return
            if not self._confirm_with_user(analysis_result):
                return
            erase_rects = self._calculate_erase_bbox(ocr_blocks, analysis_result)
            if not erase_rects:
                print("警告：没有计算出需要涂白的区域。")
                return
            logger.info("Applying whiteout to PDF...")
            doc = fitz.open(pdf_path)
            pages_modified = set()
            for page_idx, rect in erase_rects:
                page = doc[page_idx]
                page.draw_rect(rect, color=Config.WHITEOUT_COLOR, fill=Config.WHITEOUT_COLOR, overlay=True)
                pages_modified.add(page_idx)
            doc.save(output_path)
            doc.close()
            logger.info(f"Success! Saved to: {output_path}")
            print(f"\n处理完成！共修改 {len(pages_modified)} 页。文件已保存: {output_path}")
        except Exception as e:
            logger.exception("Fatal error during processing.")
            print(f"发生错误: {e}")
if __name__ == "__main__":
    tool = SemanticWhiteoutTool()
    input_pdf = "meeting_scan.pdf" 
    output_pdf = "meeting_cleaned.pdf"
    if not os.path.exists(input_pdf):
        print(f"错误：当前目录下未找到 '{input_pdf}'。")
    else:
        user_input = input("请输入您想要保留的议题关键词（例如：包二）: ")
        if user_input:
            tool.process_pdf(input_pdf, output_pdf, user_input)