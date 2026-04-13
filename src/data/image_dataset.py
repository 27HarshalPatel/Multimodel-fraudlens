"""Image dataset for check fraud detection using SigLIP 2 preprocessing."""

import logging
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoImageProcessor

logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    """PyTorch dataset for check images (normal vs tampered).

    Uses SigLIP 2 image processor for consistent preprocessing.
    Applies augmentations during training.
    """

    def __init__(
        self,
        image_dir: str | Path,
        model_name: str = "google/siglip2-base-patch16-224",
        image_size: int = 224,
        augment: bool = True,
        augmentation_config: Optional[dict] = None,
    ):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.augment = augment

        # Discover images
        self.samples: list[tuple[Path, int]] = []

        normal_dir = self.image_dir / "normal"
        tampered_dir = self.image_dir / "tampered"

        if normal_dir.exists():
            for img_path in sorted(normal_dir.glob("*.png")):
                self.samples.append((img_path, 0))
            for img_path in sorted(normal_dir.glob("*.jpg")):
                self.samples.append((img_path, 0))

        if tampered_dir.exists():
            for img_path in sorted(tampered_dir.glob("*.png")):
                self.samples.append((img_path, 1))
            for img_path in sorted(tampered_dir.glob("*.jpg")):
                self.samples.append((img_path, 1))

        # SigLIP 2 processor
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self._use_processor = True
        except Exception:
            logger.warning("Could not load SigLIP 2 processor, using manual transforms.")
            self._use_processor = False

        # Augmentation transforms (applied before processor)
        aug_cfg = augmentation_config or {}
        aug_transforms = []
        if augment:
            if aug_cfg.get("random_crop", True):
                aug_transforms.append(transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)))
            if aug_cfg.get("horizontal_flip", True):
                aug_transforms.append(transforms.RandomHorizontalFlip(p=0.5))
            if aug_cfg.get("color_jitter", 0.2):
                jitter = aug_cfg.get("color_jitter", 0.2)
                aug_transforms.append(transforms.ColorJitter(
                    brightness=jitter, contrast=jitter, saturation=jitter
                ))
            if aug_cfg.get("gaussian_blur", 0.1):
                aug_transforms.append(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)))

        self.augmentation = transforms.Compose(aug_transforms) if aug_transforms else None

        # Fallback transform (when processor is unavailable)
        self.fallback_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        logger.info(
            "ImageDataset: %d samples (%d normal, %d tampered), augment=%s",
            len(self.samples),
            sum(1 for _, l in self.samples if l == 0),
            sum(1 for _, l in self.samples if l == 1),
            augment,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        # Apply augmentation
        if self.augmentation is not None:
            img = self.augmentation(img)

        # Process for SigLIP 2
        if self._use_processor:
            processed = self.processor(images=img, return_tensors="pt")
            pixel_values = processed["pixel_values"].squeeze(0)
        else:
            if not isinstance(img, Image.Image):
                img = transforms.ToPILImage()(img)
            pixel_values = self.fallback_transform(img)

        return {
            "image": pixel_values,
            "label": torch.tensor(label, dtype=torch.float32),
        }
