import os
import cv2
import argparse
import numpy as np


def find_predicted_video(idx, validation_dir):
    """Find the predicted (generated) video in the validation directory by matching the sample ID."""
    for fname in os.listdir(validation_dir):
        if int(os.path.splitext(fname)[0]) == idx:
            return os.path.join(validation_dir, fname)
    return None


def create_comparison_video(sample_dir, predicted_paths, pred_labels, output_path, target_size=None):
    """
    Create a multi-panel comparison video.
    Row 1: Masked Input | Reference Image | Ground Truth
    Row 2: Predicted (ckpt1) | Predicted (ckpt2) | ...
    """
    masked_input_path = os.path.join(sample_dir, "masked_input.mp4")
    reference_image_path = os.path.join(sample_dir, "reference_image.jpg")
    target_path = os.path.join(sample_dir, "target.mp4")

    # Verify dataset files exist
    for path, name in [(masked_input_path, "masked_input"), (reference_image_path, "reference_image"),
                        (target_path, "target")]:
        if not os.path.exists(path):
            print(f"  [SKIP] Missing {name}: {path}")
            return False

    # Open dataset video captures
    cap_masked = cv2.VideoCapture(masked_input_path)
    cap_target = cv2.VideoCapture(target_path)
    ref_image = cv2.imread(reference_image_path)

    if not cap_masked.isOpened() or not cap_target.isOpened() or ref_image is None:
        print("  [ERROR] Could not open dataset media files.")
        return False

    # Open prediction video captures
    cap_preds = []
    valid_pred_labels = []
    for pred_path, label in zip(predicted_paths, pred_labels):
        if pred_path is None:
            cap_preds.append(None)
            valid_pred_labels.append(label)
            continue
        cap = cv2.VideoCapture(pred_path)
        if not cap.isOpened():
            print(f"  [WARN] Could not open predicted video: {pred_path}")
            cap_preds.append(None)
        else:
            cap_preds.append(cap)
        valid_pred_labels.append(label)

    # Get video properties
    fps = cap_target.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 16.0

    # Determine the panel size
    panel_h = int(cap_target.get(cv2.CAP_PROP_FRAME_HEIGHT))
    panel_w = int(cap_target.get(cv2.CAP_PROP_FRAME_WIDTH))
    if target_size is not None:
        panel_h, panel_w = target_size, target_size

    # Get max frame count across all videos
    frame_counts = [
        int(cap_masked.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cap_target.get(cv2.CAP_PROP_FRAME_COUNT)),
    ]
    for cap in cap_preds:
        if cap is not None:
            frame_counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    max_frames = max(frame_counts)

    # Resize reference image to panel size
    ref_resized = cv2.resize(ref_image, (panel_w, panel_h))

    # Layout
    n_preds = len(cap_preds)
    n_cols = max(3, n_preds)  # At least 3 columns for top row
    n_rows = 2
    label_height = 36
    gap = 2
    grid_w = panel_w * n_cols + gap * (n_cols - 1)
    grid_h = (panel_h + label_height) * n_rows + gap * (n_rows - 1)

    # Labels for each cell
    top_labels = ["Masked Input", "Reference Image", "Ground Truth"]
    bottom_labels = valid_pred_labels

    # Setup video writer
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (grid_w, grid_h))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    font_thickness = 2

    frame_idx = 0
    last_masked, last_target = None, None
    last_preds = [None] * n_preds

    while frame_idx < max_frames:
        # Read dataset frames
        ret_m, frame_m = cap_masked.read()
        ret_t, frame_t = cap_target.read()
        if ret_m:
            last_masked = frame_m
        if ret_t:
            last_target = frame_t

        # Read prediction frames
        for i, cap in enumerate(cap_preds):
            if cap is not None:
                ret, frame = cap.read()
                if ret:
                    last_preds[i] = frame

        # Skip if essential frames haven't been read yet
        if last_masked is None or last_target is None:
            frame_idx += 1
            continue

        # Build canvas
        canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        # --- Row 1: Masked Input | Reference Image | Ground Truth ---
        row1_panels = [
            cv2.resize(last_masked, (panel_w, panel_h)),
            ref_resized.copy(),
            cv2.resize(last_target, (panel_w, panel_h)),
        ]
        for col, (panel, label) in enumerate(zip(row1_panels, top_labels)):
            x_off = col * (panel_w + gap)
            y_off = 0
            # Label
            cv2.rectangle(canvas, (x_off, y_off), (x_off + panel_w, y_off + label_height), (40, 40, 40), -1)
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            text_x = x_off + (panel_w - text_size[0]) // 2
            text_y = y_off + (label_height + text_size[1]) // 2
            cv2.putText(canvas, label, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
            # Panel
            canvas[y_off + label_height:y_off + label_height + panel_h, x_off:x_off + panel_w] = panel

        # --- Row 2: Predictions ---
        row2_y = (panel_h + label_height) + gap
        for col in range(n_preds):
            x_off = col * (panel_w + gap)
            label = bottom_labels[col]

            # Label
            cv2.rectangle(canvas, (x_off, row2_y), (x_off + panel_w, row2_y + label_height), (40, 40, 40), -1)
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            text_x = x_off + (panel_w - text_size[0]) // 2
            text_y = row2_y + (label_height + text_size[1]) // 2
            cv2.putText(canvas, label, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

            # Panel (show black if no video available)
            if last_preds[col] is not None:
                panel = cv2.resize(last_preds[col], (panel_w, panel_h))
            else:
                panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
                # Draw "N/A" text
                na_size = cv2.getTextSize("N/A", font, 1.0, 2)[0]
                na_x = (panel_w - na_size[0]) // 2
                na_y = (panel_h + na_size[1]) // 2
                cv2.putText(panel, "N/A", (na_x, na_y), font, 1.0, (100, 100, 100), 2)
            canvas[row2_y + label_height:row2_y + label_height + panel_h, x_off:x_off + panel_w] = panel

        writer.write(canvas)
        frame_idx += 1

    cap_masked.release()
    cap_target.release()
    for cap in cap_preds:
        if cap is not None:
            cap.release()
    writer.release()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Plot masked_input, reference_image, target, and predicted videos from multiple checkpoints."
    )
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to masked_dataset_resized directory.")
    parser.add_argument("--validation_dirs", type=str, nargs="+", required=True,
                        help="One or more paths to validation result directories (e.g. from different checkpoints).")
    parser.add_argument("--validation_labels", type=str, nargs="*", default=None,
                        help="Labels for each validation directory. Defaults to directory basename.")
    parser.add_argument("--output_dir", type=str, default="./comparison_outputs",
                        help="Directory to save comparison videos.")
    parser.add_argument("--panel_size", type=int, default=None,
                        help="Resize each panel to this square size (e.g. 480). Default: use original size.")
    parser.add_argument("--samples", type=str, nargs="*", default=None,
                        help="Specific sample folder names to process. Default: all matched samples.")
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_dir):
        print(f"Error: Dataset directory not found: {args.dataset_dir}")
        return

    for vdir in args.validation_dirs:
        if not os.path.isdir(vdir):
            print(f"Error: Validation directory not found: {vdir}")
            return

    # Determine labels
    if args.validation_labels:
        if len(args.validation_labels) != len(args.validation_dirs):
            print(f"Error: Number of labels ({len(args.validation_labels)}) must match "
                  f"number of validation dirs ({len(args.validation_dirs)}).")
            return
        labels = args.validation_labels
    else:
        labels = [os.path.basename(os.path.normpath(d)) for d in args.validation_dirs]

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect all sample directories
    sample_dirs = sorted([
        d for d in os.listdir(args.dataset_dir)
        if os.path.isdir(os.path.join(args.dataset_dir, d))
    ])

    if args.samples:
        sample_dirs = [d for d in sample_dirs if d in args.samples]

    print(f"Dataset: {args.dataset_dir}")
    print(f"Validation dirs ({len(args.validation_dirs)}):")
    for vdir, label in zip(args.validation_dirs, labels):
        print(f"  - [{label}] {vdir}")

    matched, skipped = 0, 0
    for idx, sample_name in enumerate(sample_dirs):
        sample_id = sample_name.split("_")[0]
        sample_dir = os.path.join(args.dataset_dir, sample_name)

        # Find predicted video from each validation directory
        predicted_paths = []
        found_any = False
        for vdir in args.validation_dirs:
            pred_path = find_predicted_video(idx, vdir)
            predicted_paths.append(pred_path)
            if pred_path is not None:
                found_any = True

        if not found_any:
            print(f"[SKIP] No predicted video found for {sample_name} in any validation dir")
            skipped += 1
            continue

        output_path = os.path.join(args.output_dir, f"{sample_name}_comparison.mp4")
        print(f"[{matched + 1}] Processing {sample_name} ...")
        for pred_path, label in zip(predicted_paths, labels):
            status = os.path.basename(pred_path) if pred_path else "N/A"
            print(f"     {label}: {status}")

        success = create_comparison_video(sample_dir, predicted_paths, labels, output_path, args.panel_size)
        if success:
            print(f"     Saved: {output_path}")
            matched += 1
        else:
            skipped += 1

    print(f"\nDone. {matched} comparison(s) created, {skipped} skipped.")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()