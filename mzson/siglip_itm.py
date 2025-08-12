from __future__ import annotations

from typing import List

import numpy as np
import torch
import open_clip
from PIL import Image


class SigLipITM:
    def __init__(self, device: torch.device | str | None = None,
                 model_name: str = "siglip2_base_patch16_384",
                 pretrained: str = "",
                 backend: str = "timm") -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.backend = backend
        if backend == "timm":
            import timm
            self.model = timm.create_model(model_name, pretrained=True)
            self.model = self.model.to(self.device)
            self.model.eval()
            from timm.data import resolve_data_config, create_transform
            cfg = resolve_data_config({}, model=self.model)
            self.preprocess = create_transform(**cfg)
            self.tokenizer = None
        else:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)

    def _to_pil(self, image_rgb: np.ndarray) -> Image.Image:
        if isinstance(image_rgb, Image.Image):
            return image_rgb
        img = image_rgb
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        return Image.fromarray(img)

    @torch.no_grad()
    def encode_image(self, image_rgb: np.ndarray) -> np.ndarray:
        img = self._to_pil(image_rgb)
        img_t = self.preprocess(img).unsqueeze(0).to(self.device)
        if self.backend == "timm":
            if hasattr(self.model, "encode_image"):
                feats = self.model.encode_image(img_t)
            elif hasattr(self.model, "forward_image"):
                feats = self.model.forward_image(img_t)
            else:
                feats = self.model(img_t)
        else:
            feats = self.model.encode_image(img_t)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.squeeze(0).detach().cpu().numpy()

    @torch.no_grad()
    def image_text_scores(self, image_rgb: np.ndarray, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        img = self._to_pil(image_rgb)
        img_t = self.preprocess(img).unsqueeze(0).to(self.device)
        if self.backend == "timm":
            if hasattr(self.model, "encode_image"):
                i_feat = self.model.encode_image(img_t)
            elif hasattr(self.model, "forward_image"):
                i_feat = self.model.forward_image(img_t)
            else:
                i_feat = self.model(img_t)
            if hasattr(self.model, "encode_texts"):
                t_feat = self.model.encode_texts(texts)
                if isinstance(t_feat, np.ndarray):
                    t_feat = torch.from_numpy(t_feat).to(self.device)
            elif hasattr(self.model, "encode_text"):
                lst = []
                for t in texts:
                    out = self.model.encode_text(t)
                    lst.append(out if isinstance(out, torch.Tensor) else torch.tensor(out, device=self.device))
                t_feat = torch.stack(lst, dim=0)
            else:
                raise NotImplementedError
        else:
            toks = self.tokenizer(texts).to(self.device)
            i_feat = self.model.encode_image(img_t)
            t_feat = self.model.encode_text(toks)
        i_feat = i_feat / i_feat.norm(dim=-1, keepdim=True)
        t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
        sims = (i_feat @ t_feat.T).squeeze(0).detach().cpu().numpy()
        return sims.astype(float)


