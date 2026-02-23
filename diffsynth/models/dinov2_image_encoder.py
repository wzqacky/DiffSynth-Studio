import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv2ImageEncoder(nn.Module):
    """
    DINOv2 ViT-L/14 image encoder with the same ``encode_image`` interface as
    ``WanImageEncoder``.

    Output shape: ``(B, 257, 1024)`` — 1 CLS token + 256 patch tokens,
    feature dim 1024.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    def __init__(self):
        super().__init__()
        self.model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitl14", pretrained=True
        )
        self.model.eval()
        self.model.requires_grad_(False)
        self.image_size = 224

        self.register_buffer(
            "mean", torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1)
        )

    def encode_image(self, videos):
        """
        Parameters
        ----------
        videos : list[Tensor]
            Each tensor has shape ``(1, 3, H, W)`` in range ``[-1, 1]``.
            Same interface as ``WanImageEncoder.encode_image``.

        Returns
        -------
        Tensor
            ``(B, 257, 1024)`` — CLS + 256 patch tokens.
        """
        size = (self.image_size, self.image_size)
        x = torch.cat(
            [
                F.interpolate(u, size=size, mode="bicubic", align_corners=False)
                for u in videos
            ]
        )
        # Convert from [-1, 1] to [0, 1] then apply ImageNet normalisation
        x = x.mul(0.5).add(0.5)
        x = (x - self.mean.to(x)) / self.std.to(x)

        with torch.no_grad():
            features = self.model.get_intermediate_layers(
                x, n=1, reshape=False, return_class_token=True
            )
        # features is a list of (patch_tokens, cls_token) tuples — one per
        # requested layer.
        patch_tokens, cls_token = features[0]  # patch: (B, 256, 1024), cls: (B, 1024)
        cls_token = cls_token.unsqueeze(1)      # (B, 1, 1024)
        out = torch.cat([cls_token, patch_tokens], dim=1)  # (B, 257, 1024)
        return out
