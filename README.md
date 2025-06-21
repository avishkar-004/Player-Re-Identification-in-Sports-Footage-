# Player Re-Identification in Sports Footage

A computer vision pipeline for detecting, tracking, and re-identifying players in soccer footage. Uses YOLOv11 for object detection and ResNet18 appearance embeddings for maintaining consistent player identities across frame re-entries.

## Results

- **96.27% player re-identification accuracy** (target: >80%)
- ~64 FPS processing speed (target: ~17 FPS)
- 4 ID switches for players over 15s clip (target: <5)
- Handles occlusion and frame re-entry during goal events

## Architecture

1. **Detection** - YOLOv11 with per-class confidence thresholds (ball: 0.2, players: 0.25)
2. **Feature Extraction** - ResNet18 embeddings (pretrained, 64x64 input crops)
3. **Tracking** - Multi-modal similarity matching combining spatial distance, appearance cosine similarity, and size ratio with class-dependent weights
4. **Visualization** - Color-coded bounding boxes (red=ball, yellow=goalkeeper, green=player, blue=referee)

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA (recommended)
- Pre-trained YOLOv11 model (`best.pt`) - [Download](https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view)

## Setup

```bash
git clone https://github.com/avishkar-004/Player-Re-Identification-in-Sports-Footage-.git
cd Player-Re-Identification-in-Sports-Footage-
pip install -r requirements.txt
```

Place `best.pt` and input video (`15sec_input_720p.mp4`) in the project root.

## Usage

### Process Video

```bash
python main.py
```

Outputs:
- `output/output.mp4` - Annotated video with tracked player IDs
- `logs/detection_log.txt` - Per-frame detection results
- `logs/tracking_log.txt` - ID assignments and re-identification events

### Evaluate Performance

```bash
python evaluate.py
```

Generates `evaluation_report.txt` with detection accuracy, ID switches, re-ID success rates, and processing speed metrics.

## Project Structure

```
.
  main.py                  # Main processing pipeline
  evaluate.py              # Performance evaluation
  requirements.txt         # Python dependencies
  utils/
    detection.py           # YOLOv11 detector wrapper
    tracking.py            # Enhanced tracker with ResNet18 re-ID
    visualization.py       # Bounding box annotation
  logs/                    # Detection and tracking logs
  output/                  # Generated annotated video
```

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `conf_thres` | 0.25 | Base detection confidence |
| `similarity_threshold` | 0.35 | Minimum similarity for re-ID match |
| `max_disappeared_frames` | 75 | Frames before dropping a lost track (~3s) |
| `max_distance_threshold` | 100.0 | Spatial distance normalization factor |

## License

MIT


## Evaluation Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| Player Re-ID | >80% | 96.27% |
| Processing FPS | ~17 | ~64 |
| Player ID Switches | <5 | 4 |
| Ball Detection | ~1/frame | 0.95 |


## Troubleshooting

- **CUDA out of memory**: Reduce input resolution or batch size
- **Low FPS**: Check GPU availability with `torch.cuda.is_available()`
- **Missing model**: Download `best.pt` from the link above


## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests before submitting
4. Open a pull request


## Acknowledgments

- YOLOv11 by Ultralytics for object detection
- ResNet18 (PyTorch) for appearance feature extraction
- OpenCV for video processing and visualization


## Hardware Requirements

- **Minimum**: 8GB RAM, 4-core CPU
- **Recommended**: 16GB RAM, NVIDIA GPU with 4GB+ VRAM
