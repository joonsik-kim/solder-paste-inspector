# SPI v1.0 - Solder Paste Inspector

Solder fillet segmentation and shape analysis tool for edge devices.

## Folder Structure

```
deploy/
├── spi.py                 # Main CLI tool (single file, self-contained)
├── models/
│   ├── best_model.pth     # PyTorch model
│   └── spi_model.onnx     # ONNX model (lightweight)
├── model_registry.json    # Model version history
├── requirements.txt       # Python dependencies
└── README.md
```

## Setup

**Requirements**: Python 3.10+

```bash
pip install torch torchvision opencv-python numpy albumentations segmentation-models-pytorch onnxruntime
```

Or use `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Commands

### Predict (single image)

```bash
python spi.py predict image.png
python spi.py predict image.png --output results/
```

### Predict (batch - directory)

```bash
python spi.py predict images_folder/ --output results/
python spi.py predict images_folder/ --output results/ --limit 10
```

### Export to ONNX

```bash
python spi.py export --format onnx
```

### Update Model

```bash
python spi.py update-model path/to/new_model.pth --note "retrained with 1000 labels"
```

### Show Info

```bash
python spi.py info
```

## Predict Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output DIR` | `spi_output/` | Output directory |
| `--threshold FLOAT` | `0.5` | Binarization threshold |
| `--pixel-size FLOAT` | `15.0` | Pixel size in um (VT-S730) |
| `--runtime pytorch\|onnx` | `pytorch` | Inference runtime |
| `--model PATH` | `models/best_model.pth` | Custom model path |
| `--no-viz` | - | Skip analysis visualization |
| `--device cuda\|cpu` | `auto` | Compute device |
| `--limit N` | `0` (all) | Max images to process |

## Output Files (per image)

| File | Description |
|------|-------------|
| `{name}_mask.png` | Binary segmentation mask |
| `{name}_overlay.png` | Original + mask overlay |
| `{name}_analysis.png` | 4-panel analysis (height map, slope, profile) |
| `{name}_result.json` | Quantitative results (area, height, slope, CE ratio) |

Batch mode also generates `batch_summary.json` and `batch_summary.csv`.

## ONNX Mode (lightweight)

For edge devices without PyTorch, use ONNX runtime:

```bash
python spi.py predict image.png --runtime onnx
```

Only requires: `onnxruntime`, `opencv-python`, `numpy`
