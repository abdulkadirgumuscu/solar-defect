"""Extract deep features from pretrained CNNs (no fine-tuning).

Layer names follow the paper, which used MATLAB's ResNet-50 and SqueezeNet.
The equivalent layers in torchvision:

    Paper     MATLAB layer          torchvision                           dims
    Relu4     activation_4_relu     resnet50.layer1[0] output             256   (shallow)
    Relu25    activation_25_relu    resnet50.layer3[0] output             1024  (deep)
    Relu34    activation_34_relu    resnet50.layer3[3] output             1024  (very deep)
    Conv10    conv10                squeezenet1_1.classifier[1] output    1000

Each feature map is reduced to one value per channel with global average
pooling, which gives the dimensions reported in the paper.

Expected data layout (one folder per class):
    data/busbar/flawless/*.png
    data/busbar/busbar_slip/*.png

Usage:
    python -m src.features --data data/busbar --out features/features.npz
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

CLASSES = ["flawless", "busbar_slip"]          # label 0, label 1
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

PREPROCESS = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),   # EL images are greyscale; CNNs expect 3 channels
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def list_images(root):
    paths, labels = [], []
    for label, name in enumerate(CLASSES):
        folder = Path(root) / name
        if not folder.is_dir():
            raise FileNotFoundError(f"Expected a folder named '{name}' in {root}")
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT)
        paths += files
        labels += [label] * len(files)
    return paths, np.array(labels)


class DeepFeatures:
    """Runs each image through both networks once and keeps the chosen layers."""

    def __init__(self, device):
        self.device = device
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device).eval()
        self.squeeze = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1).to(device).eval()
        self.store = {}
        layers = {"Relu4": self.resnet.layer1[0],
                  "Relu25": self.resnet.layer3[0],
                  "Relu34": self.resnet.layer3[3],
                  "Conv10": self.squeeze.classifier[1]}
        for name, layer in layers.items():
            layer.register_forward_hook(self._hook(name))

    def _hook(self, name):
        def save(module, inputs, output):
            self.store[name] = output.mean(dim=(2, 3)).cpu()     # global average pooling
        return save

    @torch.no_grad()
    def __call__(self, batch):
        self.store.clear()
        batch = batch.to(self.device)
        self.resnet(batch)
        self.squeeze(batch)
        return {k: v.numpy() for k, v in self.store.items()}


def extract(root, batch_size=32, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths, labels = list_images(root)
    model = DeepFeatures(device)
    feats = {}
    for i in range(0, len(paths), batch_size):
        imgs = []
        for p in paths[i:i + batch_size]:
            with Image.open(p) as im:
                imgs.append(PREPROCESS(im.convert("L")))
        for name, arr in model(torch.stack(imgs)).items():
            feats.setdefault(name, []).append(arr)
        print(f"  {min(i + batch_size, len(paths))}/{len(paths)} images", end="\r")
    print()
    feats = {k: np.concatenate(v) for k, v in feats.items()}
    return feats, labels, [str(p) for p in paths]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/busbar")
    ap.add_argument("--out", default="features/features.npz")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    feats, labels, paths = extract(args.data, args.batch_size)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, labels=labels, paths=np.array(paths), **feats)
    print(f"{len(labels)} images ({(labels == 0).sum()} flawless, {(labels == 1).sum()} busbar slip)")
    for k, v in feats.items():
        print(f"  {k}: {v.shape[1]} features")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
