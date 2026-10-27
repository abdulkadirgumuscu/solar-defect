# Busbar Slip Detection in PV Electroluminescence Images

Code for:

> Şimşek Kaya, S., Gümüşçü, A., & Beşli, N. (2024). **Efficient Busbar Slip Defects Detection in Photovoltaic Cell Electroluminescence Images.** *Adıyaman Üniversitesi Mühendislik Bilimleri Dergisi*, 23, 363–377. [doi:10.54365/adyumbd.1494765](https://doi.org/10.54365/adyumbd.1494765)

Busbar slip is a manufacturing defect in which a busbar shifts out of position. This project detects it
from electroluminescence (EL) images using **deep features from pretrained CNNs, without fine-tuning**,
classified with simple, fast models. The question it answers: *which network layers carry the most useful
information, and does combining layers from different networks help?*

## Method

**1. Deep feature extraction** (`src/features.py`)

Each image is passed once through ImageNet-pretrained ResNet-50 and SqueezeNet 1.1, and the output
of four layers is kept. Each feature map is reduced to one value per channel by global average pooling.

| Feature | Network | Depth | Layer (MATLAB name) | torchvision equivalent | Dimensions |
|:--|:--|:--|:--|:--|--:|
| Relu4 | ResNet-50 | shallow | `activation_4_relu` | `layer1[0]` output | 256 |
| Relu25 | ResNet-50 | deep | `activation_25_relu` | `layer3[0]` output | 1,024 |
| Relu34 | ResNet-50 | very deep | `activation_34_relu` | `layer3[3]` output | 1,024 |
| Conv10 | SqueezeNet | final conv | `conv10` | `classifier[1]` output | 1,000 |

**2. Feature fusion:** the four single layers plus six pairs joined side by side (for example Relu25 + Conv10 → 2,024 dimensions).

**3. Classification** (`src/classify.py`): k-nearest neighbours (k = 1, Euclidean distance) and Gaussian Naive Bayes.

**4. Evaluation**
- **Hold-out, as in the paper:** stratified split into 375 training and 125 test images (75 flawless, 50 busbar slip).
- **Added here: repeated stratified 5-fold cross-validation (10 repeats).** With 125 test images, one image changes accuracy by 0.008, so cross-validation shows which differences between feature sets are real.

Metrics: accuracy, precision, recall and F-score, with busbar slip as the positive class.

## Dataset

The images used in this work were **selected from a larger EL image dataset**: we extracted the cells
with busbar slip defects and a set of flawless cells, giving 500 images (**300 flawless**, **200 busbar slip**).

**Scope:** the model is trained only on this binary task, busbar slip vs. flawless. It has not seen
other defect types (cracks, inactive areas, finger interruptions and so on) and should not be used
to detect them.

The images are not included in this repository. To run the code on your own EL images, arrange them as:

```
data/busbar/
  flawless/      *.png, *.jpg, *.bmp or *.tif
  busbar_slip/
```

## Run it

**Google Colab (free GPU):** upload your data to Google Drive, open `notebooks/colab.ipynb`, and run all cells.

**Locally:**
```bash
pip install -r requirements.txt
python -m src.features --data data/busbar        # extract and cache features (once)
python -m src.classify                           # hold-out + cross-validation → results/
python -m src.classify --scale                   # optional: standardise features first
```

Feature extraction for 500 images takes a few minutes on a CPU; classification takes seconds, because
features are cached in `features/features.npz`.

Tests (no images or GPU needed): `python tests/test_classify.py`

## Project structure

```
src/
  features.py   pretrained ResNet-50 / SqueezeNet feature extraction with forward hooks
  classify.py   feature fusion, 1-NN and Naive Bayes, hold-out and cross-validation
tests/test_classify.py
notebooks/colab.ipynb
```

## Why this approach

Using a pretrained network as a fixed feature extractor needs no GPU training, works with a few hundred
labelled images, and makes each prediction cheap and fast, which suits inspection on a production line.
The trade-off is that the features are not adapted to EL images; fine-tuning the network is the natural next step when more labelled data is available.
