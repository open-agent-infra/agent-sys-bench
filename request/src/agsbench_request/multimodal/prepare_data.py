"""Multimodal scenario data preparation — HuggingGPT benchmark tasks.

Supports two task types:
  - image-to-text: downloads tiny-imagenet, saves images, generates manifest
    with container-internal image paths
  - text-generation: downloads alpaca instructions (no images needed)

Note: image-to-text mode requires the `datasets` library. The agsbench-request
image doesn't include it (lightweight). To prepare image data, use the
hugginggpt coordinator image instead:
    docker run --rm -v <output>:/output -e HF_ENDPOINT=https://hf-mirror.com \\
      hugginggpt:v1 python -c "from datasets import load_dataset; ..."
Or manually download images and create the manifest with paths like:
    /data/images/0.jpg
"""

import json
import os
import sys

MANIFEST_NAME = "hugginggpt_data.txt"

# Container-internal path where images are mounted in both coordinator and models_server
CONTAINER_IMAGE_DIR = "/data/images"


def extra_args(parser):
    parser.add_argument("--task", default="image-to-text",
                        choices=["image-to-text", "text-generation"],
                        help="Task type: image-to-text (needs images) or text-generation (text only)")
    parser.add_argument("--num_samples", type=int, default=5)


def run(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, MANIFEST_NAME)
    if os.path.exists(manifest_path) and not args.force:
        print(f"{manifest_path} already exists. Use --force to overwrite.")
        return

    if args.task == "image-to-text":
        _prepare_image_to_text(args, output_dir, manifest_path)
    elif args.task == "text-generation":
        _prepare_text_generation(args, output_dir, manifest_path)


def _prepare_image_to_text(args, output_dir, manifest_path):
    """Download tiny-imagenet, save images, generate manifest with container paths."""
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    try:
        from datasets import load_dataset
        print("Loading tiny-imagenet dataset...")
        ds = load_dataset("Maysee/tiny-imagenet", split="valid")
    except ImportError:
        print("WARNING: 'datasets' library not installed (agsbench-request image is lightweight).")
        print("  To prepare image data, use the hugginggpt coordinator image:")
        print("    docker run --rm -v <output>:/output -e HF_ENDPOINT=https://hf-mirror.com \\")
        print("      hugginggpt:v1 python -c \"from datasets import load_dataset; ...\"")
        print("  Or manually download images to <output>/images/ and create manifest.")
        print("  Falling back to text-generation mode.")
        _prepare_text_generation(args, output_dir, manifest_path)
        return
    except Exception as e:
        print(f"WARNING: Cannot load tiny-imagenet: {e}")
        print("  Falling back to text-generation mode.")
        _prepare_text_generation(args, output_dir, manifest_path)
        return

    # Save images locally
    num = min(args.num_samples, len(ds))
    for i in range(num):
        img = ds[i]["image"]
        img.save(os.path.join(image_dir, f"{i}.jpg"))
    print(f"Saved {num} images to {image_dir}")

    # Generate manifest with container-internal paths
    # Both coordinator and models_server mount images at /data/images
    with open(manifest_path, "w", encoding="utf-8") as f:
        for i in range(num):
            img_path = f"{CONTAINER_IMAGE_DIR}/{i}.jpg"
            f.write(f"12, Please talk about the figure {img_path}\n")
    print(f"Wrote {num} image-to-text tasks to {manifest_path}")


def _prepare_text_generation(args, output_dir, manifest_path):
    """Download alpaca instructions (text only, no images needed)."""
    try:
        from datasets import load_dataset
        print("Loading alpaca dataset...")
        ds = load_dataset("tatsu-lab/alpaca")["train"]["instruction"]
    except Exception:
        print("WARNING: Cannot load alpaca. Using builtin sample questions.")
        sample_questions = [
            "What is the capital of France and why is it important?",
            "Explain the difference between supervised and unsupervised learning.",
            "Write a short poem about autumn.",
        ][:args.num_samples]
        with open(manifest_path, "w", encoding="utf-8") as f:
            for q in sample_questions:
                f.write(f"6, {q}\n")
        print(f"Wrote {len(sample_questions)} text-generation tasks (builtin) to {manifest_path}")
        return

    count = 0
    with open(manifest_path, "w", encoding="utf-8") as f:
        for instruction in ds:
            if "\n" in instruction:
                continue
            f.write(f"6, {instruction}\n")
            count += 1
            if count >= args.num_samples:
                break
    print(f"Wrote {count} text-generation tasks to {manifest_path}")
