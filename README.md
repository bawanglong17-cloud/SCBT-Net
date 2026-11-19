# SCBT-Net

**SCBT-Net: Contour-Aligned Multiband Directional Derivatives for Breast Ultrasound Lesion Segmentation**

SCBT-Net refines binary segmentation logits from a frozen parent model. It decomposes aligned features into Gaussian bands, measures their derivatives in the local contour frame, and learns an additive logit residual with **16,189 trainable parameters**. A zero-initialized output layer preserves the parent's logits at initialization.

![SCBT-Net architecture from the manuscript](assets/fig_architecture.png)

- **Breast ultrasound** — Official BUS-BRA fivefold cross-validation with 1,064 patients and 1,875 images. All views of each patient stay in the same partition.
- **Backbone transfer** — The same head was attached to SAM2-UNet, VM-UNet, and FDSS-Net using a parameter-free feature adapter.
- **External evaluation** — The manuscript reports transfer to a deduplicated 38-image BUSI holdout and evaluates the head on a separate 738-image RGB colonoscopy benchmark with video separation where sequence IDs are available.

## Installation

Use Python 3.11 or newer. Select a CPU or CUDA wheel for your system using the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

```sh
python -m venv .venv
# POSIX: source .venv/bin/activate
# PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The core package uses PyTorch, NumPy, and Pillow. Install any chosen backbone's own dependencies separately. For example, `python -m pip install timm scipy scikit-learn` installs common optional research dependencies.

## Minimal plug-and-play example

`SCBTNetRefiner.attach(backbone)` freezes the supplied binary segmenter and wraps it in the parameter-free adapter. The backbone wrapper returns `(coarse_logits, (level1, level2, level3))`. The coarse logits have shape `[B, 1, H, W]`, and each feature is a `[B, C_i, H_i, W_i]` tensor. Features may differ in channel count and resolution. The adapter pools their channels to 16, 17, and 17, resizes them onto the logit grid, and derives a 32-channel band source.

```python
import torch
from models import SCBTNetRefiner
from scripts.demo_inference import DemoBackbone

backbone = DemoBackbone().eval()  # Replace with your binary backbone and decoder taps.
model = SCBTNetRefiner.attach(backbone).eval()
image = torch.rand(1, 3, 64, 64)

with torch.inference_mode():
    coarse, _ = backbone(image)
    refined = model(image)
    assert torch.equal(coarse, refined)
    mask = refined.sigmoid() >= 0.5
```

For native inputs, invoke `head(hidden, coarse_logits, bands=(low, mid, high))` or supply `band_source=aligned_features`. The hidden tensor has 50 channels, and each band or the band source has 32. The Gaussian scales are 1.5 and 3.0 px. The directional input includes all three normal responses, the high-band tangent response, and the high-minus-low normal response. The refiner emits raw logits. The 3D tensor diagram illustrates the feature flow.

```sh
python scripts/demo_inference.py
python scripts/demo_inference.py --device cuda:0 --image ./my_example.png
```

The default demo tests random input and a synthetic image on CPU, then on CUDA if available. It checks the exact parameter count, finite outputs, and bit-exact identity. The optional image is processed locally with the same untrained demo backbone.

## BUS-BRA training and out-of-fold evaluation

1. Obtain [BUS-BRA v1.0](https://doi.org/10.5281/zenodo.8231412). The checked-in split roster preserves the official outer test membership and the paper's fixed internal validation assignments. Prepare a CSV of relative image and mask paths from the official metadata.

   ```sh
   python scripts/prepare_busbra_manifest.py --dataset-root datasets/BUSBRA --metadata datasets/BUSBRA/bus_data.csv --output runs/busbra_manifest.csv
   ```

   For a differently arranged extraction, pass `--image-dir` and `--mask-dir`. A manual manifest uses `image_id,patient_id,image,mask` with optional `image_sha256,mask_sha256`. The scripts verify patient separation and exact OOF coverage against the fold roster.

2. Place the original BUSI parent state in `weights/b0/busi_unet.pth` and the original fold-local G2-V2 states in `weights/b0/fold_0_g2v2.pth` through `weights/b0/fold_4_g2v2.pth`. The loader accepts the U-Net state or the historical `model` wrapper and a G2-V2 state, the `ema` wrapper, or a historical best-fold bundle under `arms.B0.state`. Loading is strict. These weights must be sourced separately.

3. Train the comparison heads on CUDA and evaluate the selected EMA state of each outer fold.

   ```sh
   python scripts/train_5fold.py --dataset-root datasets/BUSBRA --manifest runs/busbra_manifest.csv --parent-weights weights/b0 --output runs/busbra --arms G1 B1 R-Attn R-FFT R-Wavelet --device cuda:0
   python scripts/eval_oof.py --dataset-root datasets/BUSBRA --manifest runs/busbra_manifest.csv --parent-weights weights/b0 --output runs/busbra --arms B0 G1 B1 --save-predictions --device cuda:0
   ```

   Training defaults to 30 epochs with AdamW, three warmup epochs, cosine decay, FP16 AMP, and EMA at 0.995. Internal validation chooses the best state by Dice + BF1. Evaluation binds each checkpoint to the parent state, fold roster, input preprocessing configuration, and data manifest. It saves per-image and per-patient metrics, fivefold summaries, paired intervals, and optional aligned FP32 logits and thresholded masks under the output directory's `results/`.

4. The main statistical family uses 1,000 within-fold patient bootstrap replicates and 100,000 two-sided sign flips for three endpoints and two references. For the separate 18-test peer family, use `--arms B1 R-Attn R-FFT R-Wavelet --endpoints dice iou bf1 hd95 precision recall --sign-flips 10000 --results-dir runs/busbra/peer_results`. All paired differences are target minus reference, so negative HD95 favors the target.

Custom parents use `--parent your_module:load_parent`. The factory accepts `fold` and `weights_dir` keywords and returns a loaded module with the feature contract above. Adjust `--mean` and `--std` to the parent protocol. Third-party model code, initialization weights, and the BUSI and endoscopy datasets are obtained from their respective sources. The released scripts use Pillow for file decoding and CUDA for resizing, augmentation, model execution, metrics, and resampling.

For GPU latency and peak allocated memory, the default protocol is batch 1, 256 × 256, 50 warmup passes, and 500 measured passes. The reported latency covers the full model forward pass.

```sh
python scripts/benchmark_latency.py --parent-weights weights/b0 --arm B1 --weights runs/busbra/checkpoints/fold_0_B1_best.pth --device cuda:0
```

## Manuscript results

Values are the mean ± sample standard deviation of five patient-macro fold scores. Dice, IoU, BF1, precision, and recall are percentages. BF1 uses a 2-px boundary tolerance on the 256 × 256 evaluation grid. HD95 is the larger of the two directed 95th-percentile Euclidean surface distances. Cases with undefined HD95 contribute to the other metrics and are omitted only from applicable distance averages and paired comparisons.

### Table I · BUS-BRA matched refinement comparison

| Arm | Dice (%) | IoU (%) | BF1 (%) | HD95 (px) ↓ | Precision (%) | Recall (%) | Trainable increment |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 76.677 ± 0.840 | 66.306 ± 0.979 | 48.760 ± 0.654 | 34.258 ± 1.293 | 78.629 ± 1.597 | 80.732 ± 1.039 | 0 |
| G1 | 77.200 ± 0.852 | 66.958 ± 1.001 | 49.320 ± 0.672 | 33.927 ± 1.663 | 78.936 ± 1.488 | 81.332 ± 0.901 | 16,073 |
| **B1 (SCBT-Net)** | **77.719 ± 0.872** | **67.660 ± 1.040** | **51.119 ± 0.747** | **32.818 ± 1.424** | **79.427 ± 1.560** | **81.801 ± 0.883** | 16,189 |
| R-Attn | 77.145 ± 0.870 | 66.872 ± 1.022 | 49.496 ± 0.645 | 33.951 ± 1.277 | 78.810 ± 1.579 | 81.381 ± 0.936 | 16,073 |
| R-FFT | 77.318 ± 0.885 | 67.107 ± 1.042 | 49.616 ± 0.736 | 33.573 ± 1.402 | 79.060 ± 1.502 | 81.396 ± 0.818 | 16,262 |
| R-Wavelet | 77.213 ± 0.867 | 66.975 ± 1.026 | 49.445 ± 0.674 | 33.831 ± 1.659 | 78.944 ± 1.560 | 81.353 ± 0.958 | 16,273 |

B0 is frozen during refinement. Its previously trained fold-local G2-V2 head has 13,659 parameters.

### Table IV · Transfer across modern backbones

| Model | Dice (%) | IoU (%) | BF1 (%) | HD95 (px) ↓ | Precision (%) | Recall (%) | Total parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VM-UNet | 89.659 ± 0.784 | 82.666 ± 0.985 | 67.021 ± 0.653 | 11.546 ± 0.924 | 90.736 ± 0.603 | 90.584 ± 1.116 | 27,427,561 |
| VM-UNet + SCBT-Net | 89.699 ± 0.771 | 82.737 ± 0.967 | 67.458 ± 0.494 | 11.549 ± 0.968 | 90.840 ± 0.589 | 90.558 ± 1.119 | 27,443,750 |
| FDSS-Net | 90.259 ± 0.922 | 83.380 ± 1.214 | 68.471 ± 1.264 | 11.313 ± 0.846 | 89.952 ± 0.709 | 92.222 ± 1.106 | 27,192,315 |
| FDSS-Net + SCBT-Net | 90.318 ± 0.930 | 83.476 ± 1.225 | 68.580 ± 1.247 | 11.129 ± 0.837 | 90.711 ± 0.966 | 91.555 ± 1.179 | 27,208,504 |
| SAM2-UNet | 91.331 ± 0.300 | 84.895 ± 0.455 | 70.649 ± 0.582 | 9.800 ± 0.369 | 90.506 ± 0.577 | 93.480 ± 0.613 | 216,529,891 |
| SAM2-UNet + SCBT-Net | 91.474 ± 0.373 | 85.160 ± 0.570 | 71.273 ± 0.290 | 9.452 ± 0.451 | 92.049 ± 0.611 | 92.165 ± 0.546 | 216,546,080 |

The published [FDSS-Net paper](https://doi.org/10.1038/s41598-026-49070-5) describes the third-party backbone. In the BUS-BRA experiment, SAM2-UNet and VM-UNet improved BF1 after Holm correction, and FDSS-Net improved HD95. FDSS-Net's BF1 confidence interval included zero.

## Citation and licenses

Use the manuscript title to cite the project. The following entry identifies this repository's associated manuscript. Update it with the verified author list and publication metadata when available.

```bibtex
@misc{scbtnet2026,
  title        = {{SCBT-Net}: Contour-Aligned Multiband Directional Derivatives for Breast Ultrasound Lesion Segmentation},
  year         = {2026},
  howpublished = {Research software and associated manuscript}
}
```

The source code is available under the [MIT License](LICENSE). The fold roster derives from public BUS-BRA accessions. The architecture illustration contains processed, anonymized BUS-BRA example panels. Credit Wilfrido Gómez-Flores, Maria Julia Gregorio-Calas, and Wagner Coelho de Albuquerque Pereira for the underlying data [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), and cite [Gómez-Flores et al., Medical Physics, 2024](https://doi.org/10.1002/mp.16812).
