# Solar Cell Defect Classification with ResNet Transfer Learning

Classifies electroluminescence (EL) images of photovoltaic cells as **functional** or **defective**
using an ImageNet-pretrained ResNet fine-tuned in two stages, with Grad-CAM to show which
regions of the cell drive each decision.

Related publications by the author:
- Demirci, Besli & Gumuscu (2021). *Efficient deep feature extraction and classification for identifying defective photovoltaic module cells in Electroluminescence images.* Expert Systems with Applications. [doi](https://doi.org/10.1016/j.eswa.2021.114810)
- Demirci, Besli & Gumuscu (2024). *An improved hybrid solar cell defect detection approach using Generative Adversarial Networks and weighted classification.* Expert Systems with Applications. [doi](https://doi.org/10.1016/j.eswa.2024.124230)

## Dataset

[ELPV](https://github.com/zae-bayern/elpv-dataset): 2,624 grayscale 300×300 EL images of mono- and
polycrystalline cells. Experts labelled each cell with a defect probability of 0, 0.33, 0.67 or 1.

| Expert label | 0 | 0.33 | 0.67 | 1 |
|:--|--:|--:|--:|--:|
| Cells | 1,508 | 295 | 106 | 715 |

Cells with a probability of **≥ 0.5 are treated as defective**: 1,803 functional and 821 defective (31%).
The split is stratified by label and cell type: 70% train, 15% validation, 15% test.

## Method

1. **Input:** grayscale image copied to 3 channels, resized to 224×224 and normalised with ImageNet statistics.
2. **Augmentation (training only):** horizontal and vertical flips, ±5° rotation, brightness and contrast jitter.
3. **Model:** ResNet-18/34/50 with ImageNet weights. The 1000-class head is replaced by dropout and a 2-class linear layer.
4. **Stage 1:** backbone frozen, only the new head is trained (5 epochs, LR 1e-3).
5. **Stage 2:** whole network fine-tuned with a lower learning rate for the backbone (3e-5) than the head (1e-4), cosine schedule, early stopping.
6. **Class imbalance:** cross-entropy weighted by inverse class frequency.
7. **Model selection:** best epoch by validation ROC-AUC. The decision threshold is then tuned on the **validation** set to maximise F1; the test set is used only once, for the final report.
8. **Explainability:** Grad-CAM on the last convolutional block.

## Results

*Fill in after running; numbers are written to `runs/<arch>/results.md`.*

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|:--|--:|--:|--:|--:|--:|
| ResNet-18 | | | | | |
| ResNet-50 | | | | | |

The evaluation also reports:
- metrics separately for **mono** and **poly** cells,
- how the model treats each **expert label level**: the 0.33 and 0.67 cells are where the experts themselves were unsure,
- the **most confident mistakes**, and Grad-CAM maps for correct detections, missed defects and false alarms.

## Run it

**Google Colab (free GPU, recommended):** open `notebooks/colab.ipynb`, choose *Runtime → Change runtime type → GPU*, and run all cells.

**Locally:**
```bash
pip install -r requirements.txt
bash download_data.sh
python -m src.train --arch resnet18          # writes runs/resnet18/
python -m src.evaluate --run runs/resnet18   # test metrics and plots
python -m src.gradcam --run runs/resnet18    # Grad-CAM figure
```

Training uses a CUDA GPU or Apple Silicon (MPS) automatically and falls back to the CPU, which works but is slow.

Outputs in `runs/<arch>/`: `training_curves.png`, `test_curves.png` (confusion matrix, ROC, precision–recall),
`test_errors.png`, `gradcam.png`, `results.md`, `test_predictions.csv`.

Tests that need no GPU: `python tests/test_split_and_report.py`

## Project structure

```
src/
  split.py      read labels, stratified splits, class weights
  data.py       PyTorch dataset and transforms
  model.py      ResNet with a new head; freeze / unfreeze helpers
  train.py      two-stage training, early stopping, threshold tuning
  evaluate.py   test metrics, curves, error analysis
  report.py     metric and breakdown tables
  gradcam.py    Grad-CAM visualisation
notebooks/colab.ipynb
tests/test_split_and_report.py
```

## Limitations

- **Possible optimistic estimate:** ELPV does not record which module each cell came from, so cells from the same module can appear in both training and test data.
- **Label uncertainty:** the binary threshold hides expert disagreement in the 0.33 / 0.67 cells. A natural extension is to train on the soft labels directly.
- **Small dataset:** with ~400 test images, differences of 1–2% between models are within noise. Repeating with several seeds gives a fairer comparison.

## Dataset citation

Buerhop-Lutz, C., Deitsch, S., Maier, A., Gallwitz, F., Berger, S., Doll, B., Hauch, J., Camus, C., & Brabec, C. J. (2018). *A Benchmark for Visual Identification of Defective Solar Cells in Electroluminescence Imagery.* EU PVSEC. [doi](https://doi.org/10.4229/35thEUPVSEC20182018-5CV.3.15)

Deitsch, S., Christlein, V., Berger, S., Buerhop-Lutz, C., Maier, A., Gallwitz, F., & Riess, C. (2019). *Automatic classification of defective photovoltaic module cells in electroluminescence images.* Solar Energy, 185, 455–468. [doi](https://doi.org/10.1016/j.solener.2019.02.067)

Deitsch, S., Buerhop-Lutz, C., Sovetkin, E., Steland, A., Maier, A., Gallwitz, F., & Riess, C. (2021). *Segmentation of photovoltaic module cells in uncalibrated electroluminescence images.* Machine Vision and Applications, 32(4). [doi](https://doi.org/10.1007/s00138-021-01191-9)

The ELPV images are licensed under CC BY-NC-SA 4.0.
