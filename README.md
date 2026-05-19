# TopicKeeper

TopicKeeper 是一个基于本地 OCR 和本地大模型的 PDF 议题保留工具。它面向会议纪要类扫描 PDF：输入关键词后，程序会保留文档开头、目标议题和文档结尾，将其他无关议题涂白或删除。

当前版本：**3.0**

## 功能

- 支持扫描 PDF 的 OCR 识别和坐标定位。
- 使用 OpenAI 兼容的本地 LLM API 分析会议纪要结构。
- 自动识别目标议题、下一个议题和文档结尾锚点。
- 对无关内容执行 redaction 涂白，并对扫描图片叠加白色覆盖以减少视觉残留。
- 处理多行标题、跨页议题、旋转 PDF 页面和页间无关内容。
- 提供 Tkinter 图形界面，支持选择 PDF、输入关键词、确认后输出清洗文件。

## 安装依赖

建议使用 Python 3.10+。

```bash
pip install -r requirements.txt
```

## 本地模型配置

程序通过环境变量读取本地 OpenAI 兼容 API 配置。不要把私有密钥写进源码。

```bash
export TOPICKEEPER_LLM_BASE_URL="http://127.0.0.1:8000/v1"
export TOPICKEEPER_LLM_API_KEY="your-local-api-key"
export TOPICKEEPER_LLM_MODEL="your-model-name"
```

可使用 LM Studio、Ollama/OpenAI 兼容服务或其他本地推理服务，只要提供 `/v1/chat/completions` 接口即可。

## 运行

```bash
python pdfocr.py
```

运行后在界面中选择 PDF，输入要保留的议题关键词，确认模型识别结果后生成 `_cleaned.pdf`。

## 注意

- 本工具适合会议纪要、议题纪要等结构较稳定的 PDF。
- 对扫描 PDF，程序会进行 OCR 和图像级涂白；建议对输出文件做人工复核。
- 大模型推理和高 DPI OCR 会占用较多本地算力，尤其是大模型本地推理阶段。
- 测试 PDF、日志文件、输出 PDF 和本地缓存不应提交到公开仓库。

## 更新日志

见 [CHANGELOG.md](CHANGELOG.md)。
