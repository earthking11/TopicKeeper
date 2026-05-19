# Changelog

## 3.0 - 2026-05-19

### Added

- Added scan-oriented OCR preprocessing with grayscale enhancement, denoising, sharpening, adaptive thresholding, and high-DPI fallback.
- Added multi-line title matching for target and next-topic boundaries.
- Added boundary redactions for target-page top regions, next-topic regions, and unrelated pages between the target and footer.
- Added support for rotated PDF pages by converting OCR coordinates into PyMuPDF drawing coordinates.
- Added visual whiteout fallback after redaction to reduce residual pixels in scanned PDFs.
- Added environment-variable based local LLM configuration:
  - `TOPICKEEPER_LLM_BASE_URL`
  - `TOPICKEEPER_LLM_API_KEY`
  - `TOPICKEEPER_LLM_MODEL`

### Changed

- Updated GUI title to `TopicKeeper v3.0`.
- Improved LLM JSON extraction when the model returns fenced Markdown or extra text.
- Improved footer, target, and next-topic boundary handling for scanned meeting minutes.
- Improved README with public-safe setup instructions.

### Fixed

- Fixed residual OCR fragments around topic boundaries in scanned PDFs.
- Fixed incorrect redaction coordinates for rotated pages.
- Fixed overly loose title matching that could start at the trailing line of a multi-line title.
- Fixed next-topic residual text in cross-page topic extraction cases.

### Security

- Removed hard-coded local model key and model name from the public configuration path.
- Added ignore rules for local PDFs, generated outputs, logs, caches, and environment files.
