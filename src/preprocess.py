"""
preprocess.py — image variants, one function per named variant.

The variants exist to be compared against each other on an identical split. The thesis
claims the green→CLAHE→blur chain is worth 10 points on DR and 7 on DME; no run behind
that claim exists (ISSUES.md §1), so it is tested here rather than assumed.

All variants share the retinal-border crop and the resize, so what differs between them is
only the thing under test.
"""
from __future__ import annotations
import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
GAUSS_KERNEL = (5, 5)

VARIANTS = ("rgb", "green_clahe", "green_clahe_raw01")

VARIANT_DOC = {
    "rgb": "plain RGB, ImageNet normalisation. The trivial baseline the elaborate chain "
           "must beat (PROTOCOL.md / 'always build a trivial baseline first').",
    "green_clahe": "the thesis chain — green channel, CLAHE, Gaussian blur, replicated to "
                   "3 channels — but with ImageNet normalisation applied, so the "
                   "comparison against 'rgb' isolates the chain itself.",
    "green_clahe_raw01": "the thesis chain exactly as the old code implemented it: output "
                         "left in [0,1] with NO ImageNet mean/std normalisation. Included "
                         "because the old code fed this straight into an ImageNet-pretrained "
                         "DenseNet121, and that mismatch is a candidate cause of the "
                         "divergence in ISSUES.md §1.",
}


def crop_retina(img: np.ndarray, thresh: int = 12) -> np.ndarray:
    """Crop the black non-retinal border to the bounding box of the fundus disc.

    Shared by every variant so it can never be the thing that differs between them.
    Falls back to the untouched image if the mask is empty (a fully dark frame).
    """
    grey = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = grey > thresh
    if not mask.any():
        return img
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    return img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def _clahe(ch: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE).apply(ch)


def apply_variant(bgr: np.ndarray, variant: str, size: int) -> np.ndarray:
    """uint8 BGR (already border-cropped) -> float32 (3, size, size), channel-first."""
    img = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)

    if variant == "rgb":
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD

    elif variant in ("green_clahe", "green_clahe_raw01"):
        green = img[:, :, 1].copy()          # BGR -> index 1 is green
        green = _clahe(green)
        green = cv2.GaussianBlur(green, GAUSS_KERNEL, 0)
        g = green.astype(np.float32) / 255.0
        x = np.stack([g, g, g], axis=-1)
        if variant == "green_clahe":
            x = (x - IMAGENET_MEAN) / IMAGENET_STD

    else:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")

    return np.ascontiguousarray(x.transpose(2, 0, 1))


def load_cropped(path: str, cache_size: int = 640) -> np.ndarray:
    """Read an image, crop the retinal border, downscale the long side to `cache_size`.

    Cached at 640 rather than at the model's input size so the same cache serves 224 px
    probes and 512 px training runs. IDRiD is 4288x2848 and lives on a network mount, so
    decoding it twice is the expensive mistake to avoid.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    img = crop_retina(img)
    h, w = img.shape[:2]
    if max(h, w) > cache_size:
        s = cache_size / max(h, w)
        img = cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                         interpolation=cv2.INTER_AREA)
    return img
