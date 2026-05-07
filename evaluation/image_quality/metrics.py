"""Image quality evaluation metrics.

Implements:
  - PSNR  (Peak Signal-to-Noise Ratio)
  - SSIM  (Structural Similarity Index)
  - LPIPS (Learned Perceptual Image Patch Similarity)
  - FID   (Frechet Inception Distance)
  - CLIP Score (cosine similarity of CLIP embeddings)
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Tuple, Optional
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim


# ---------------------------------------------------------------------------
# Pair-wise metrics (computed per image pair)
# ---------------------------------------------------------------------------

def compute_psnr(gt: np.ndarray, pred: np.ndarray, data_range: int = 255) -> float:
    """Compute PSNR between two images (numpy uint8 arrays, H x W x C)."""
    return float(_psnr(gt, pred, data_range=data_range))


def compute_ssim(gt: np.ndarray, pred: np.ndarray, data_range: int = 255) -> float:
    """Compute SSIM between two images (numpy uint8 arrays, H x W x C)."""
    return float(_ssim(gt, pred, data_range=data_range, channel_axis=-1))


# ---------------------------------------------------------------------------
# LPIPS  (requires `lpips` package)
# ---------------------------------------------------------------------------

class LPIPSMetric:
    """Wrapper around the `lpips` library for perceptual similarity."""

    def __init__(self, net: str = "alex", device: Optional[str] = None):
        """
        Args:
            net: Backbone network - "alex" (default, fastest) or "vgg".
            device: "cuda" / "cpu".  Auto-detected if None.
        """
        import lpips
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = lpips.LPIPS(net=net).to(self.device).eval()

    @torch.no_grad()
    def compute(self, gt: np.ndarray, pred: np.ndarray) -> float:
        """Return LPIPS distance (lower is better)."""
        gt_t = self._to_tensor(gt)
        pred_t = self._to_tensor(pred)
        dist = self.model(gt_t, pred_t)
        return float(dist.item())

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Convert H x W x C uint8 numpy array to 1 x C x H x W float tensor in [-1, 1]."""
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        t = t * 2.0 - 1.0
        return t.to(self.device)


# ---------------------------------------------------------------------------
# FID  (Frechet Inception Distance) - computed over the full directory pair
# ---------------------------------------------------------------------------

class FIDMetric:
    """Compute FID between two sets of images using InceptionV3 features."""

    def __init__(self, device: Optional[str] = None):
        from torchvision.models import inception_v3, Inception_V3_Weights
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        weights = Inception_V3_Weights.DEFAULT
        self.model = inception_v3(weights=weights)
        self.model.fc = torch.nn.Identity()
        self.model = self.model.to(self.device).eval()

        self.preprocess = weights.transforms()

    @torch.no_grad()
    def extract_features(self, images: List[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """Extract InceptionV3 pool3 features for a list of images."""
        all_feats = []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i: i + batch_size]
            tensors = []
            for img in batch_imgs:
                pil = Image.fromarray(img).convert("RGB")
                tensors.append(self.preprocess(pil))
            batch = torch.stack(tensors).to(self.device)
            feats = self.model(batch)
            all_feats.append(feats.cpu().numpy())
        return np.concatenate(all_feats, axis=0)

    @staticmethod
    def _compute_statistics(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu = np.mean(feats, axis=0)
        sigma = np.cov(feats, rowvar=False)
        return mu, sigma

    @staticmethod
    def _calculate_fid(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
        """Compute the Frechet distance between two multivariate Gaussians."""
        from scipy.linalg import sqrtm

        diff = mu1 - mu2
        covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return float(fid)

    def compute(self, gt_images: List[np.ndarray], pred_images: List[np.ndarray],
                batch_size: int = 32) -> float:
        """Compute FID between two image sets (lower is better)."""
        feats_gt = self.extract_features(gt_images, batch_size)
        feats_pred = self.extract_features(pred_images, batch_size)
        mu1, sigma1 = self._compute_statistics(feats_gt)
        mu2, sigma2 = self._compute_statistics(feats_pred)
        return self._calculate_fid(mu1, sigma1, mu2, sigma2)


# ---------------------------------------------------------------------------
# CLIP Score  (cosine similarity of CLIP image embeddings)
# ---------------------------------------------------------------------------

class CLIPScoreMetric:
    """Compute CLIP-based image-to-image cosine similarity."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 device: Optional[str] = None):
        from transformers import CLIPModel, CLIPProcessor
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def compute(self, gt: np.ndarray, pred: np.ndarray) -> float:
        """Compute cosine similarity between CLIP embeddings of two images."""
        gt_pil = Image.fromarray(gt).convert("RGB")
        pred_pil = Image.fromarray(pred).convert("RGB")

        inputs_gt = self.processor(images=gt_pil, return_tensors="pt").to(self.device)
        inputs_pred = self.processor(images=pred_pil, return_tensors="pt").to(self.device)

        emb_gt = self.model.get_image_features(**inputs_gt)
        emb_pred = self.model.get_image_features(**inputs_pred)

        if not isinstance(emb_gt, torch.Tensor):
            emb_gt = emb_gt.pooler_output if hasattr(emb_gt, 'pooler_output') else emb_gt[0]
        if not isinstance(emb_pred, torch.Tensor):
            emb_pred = emb_pred.pooler_output if hasattr(emb_pred, 'pooler_output') else emb_pred[0]

        emb_gt = F.normalize(emb_gt, dim=-1)
        emb_pred = F.normalize(emb_pred, dim=-1)

        similarity = (emb_gt * emb_pred).sum(dim=-1)
        return float(similarity.item())
