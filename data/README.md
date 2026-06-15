# Data Directories

This directory contains video data and intermediate processing results.

## Subdirectories

- `frames/` - Extracted video frames
- `faces/` - Detected faces from frames
- `processed_faces/` - Resized and preprocessed faces

## Usage

Place video files here or reference them directly in predictions:

```python
from fusion_module.fusion_classifier import MultimodalPipeline

pipeline = MultimodalPipeline()
results = pipeline.predict('data/your_video.mp4')
```
