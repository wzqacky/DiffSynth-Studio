import argparse
import os
import glob

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

def compute_hf_map(image, mask=None, thresh=20):
    """Compute a high-frequency detail map from a PIL Image using Sobel edges.

    Parameters
    image : PIL.Image
        Reference image (RGB).
    mask : PIL.Image or None
        Optional object mask.  When provided the mask is eroded to strip
        segmentation-boundary edges before Sobel filtering.
    thresh : int
        Edge magnitude values below this are zeroed out.

    Returns
    numpy.ndarray
        ``(H, W, 3)`` uint8 — edge-weighted colour map.
    """

    img = np.array(image.convert("RGB"))  # (H, W, 3) uint8

    if mask is not None:
        mask_np = np.array(mask.convert("L"))  # (H, W) uint8
        mask_np = (mask_np > 127).astype(np.uint8)
        kernel = np.ones((13, 13), np.uint8)
        mask_np = cv2.erode(mask_np, kernel, iterations=2)
    else:
        mask_np = None

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_x = cv2.convertScaleAbs(sobelx)
    sobel_y = cv2.convertScaleAbs(sobely)
    # TODO: Increasing the strength seems to have added some noise to the surface
    edges = cv2.addWeighted(sobel_x, 1, sobel_y, 1, 0)

    edge_mag = np.max(edges, axis=-1)  # (H, W)
    if mask_np is not None:
        edge_mag = edge_mag * mask_np
    edge_mag[edge_mag < thresh] = 0.0

    edge_3ch = np.stack([edge_mag] * 3, axis=-1)
    hf_map = (edge_3ch.astype(np.float32) / 255.0 * img.astype(np.float32)).astype(np.uint8)
    return mask_np, hf_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="visualization_results/")
    args = parser.parse_args()

    reference_image_paths = glob.glob(os.path.join(args.data_dir, "**/reference_image.jpg"), recursive=True)
    reference_mask_paths = glob.glob(os.path.join(args.data_dir, "**/reference_mask.png"), recursive=True)

    for ref_img_path, ref_mask_path in zip(reference_image_paths, reference_mask_paths):
        ref_image = Image.open(ref_img_path)
        ref_mask = Image.open(ref_mask_path)

        ref_mask, hf_map_image = compute_hf_map(ref_image, ref_mask)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Reference Image")
        plt.imshow(ref_image)
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Object Mask")
        plt.imshow(ref_mask, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("High-Frequency Detail Map")
        plt.imshow(hf_map_image)
        plt.axis("off")

        output_dir = f"{args.output_dir}/{ref_img_path.split(os.sep)[-2]}"
        os.makedirs(output_dir, exist_ok=True)
        plot_path = f"{output_dir}/visualize.png"
        plt.savefig(plot_path)
        plt.close()