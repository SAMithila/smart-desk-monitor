#!/usr/bin/env python
"""
Generate a demo GIF from tracked frames for README.

This script combines the tracked visualization frames into an animated GIF
that can be displayed at the top of the README for visual impact.
"""

import argparse
from pathlib import Path
from PIL import Image
import glob


def create_gif(
    frames_dir: str,
    output_path: str,
    fps: int = 10,
    max_frames: int = 50,
    resize_width: int = 640,
):
    """
    Create an animated GIF from a directory of frames.

    Args:
        frames_dir: Directory containing frame images
        output_path: Output path for the GIF
        fps: Frames per second for the GIF
        max_frames: Maximum number of frames to include
        resize_width: Width to resize frames (maintains aspect ratio)
    """
    frames_dir = Path(frames_dir)

    # Find all frame images
    patterns = ["*.jpg", "*.jpeg", "*.png"]
    frame_files = []
    for pattern in patterns:
        frame_files.extend(glob.glob(str(frames_dir / pattern)))

    frame_files = sorted(frame_files)

    if not frame_files:
        print(f"❌ No frames found in {frames_dir}")
        return False

    print(f"📁 Found {len(frame_files)} frames in {frames_dir}")

    # Limit frames
    if len(frame_files) > max_frames:
        # Sample evenly
        step = len(frame_files) // max_frames
        frame_files = frame_files[::step][:max_frames]
        print(f"📊 Sampled {len(frame_files)} frames for GIF")

    # Load and process frames
    frames = []
    for i, frame_path in enumerate(frame_files):
        img = Image.open(frame_path)

        # Resize to reduce file size
        aspect_ratio = img.height / img.width
        new_height = int(resize_width * aspect_ratio)
        img = img.resize((resize_width, new_height), Image.LANCZOS)

        # Convert to RGB if necessary (GIF doesn't support RGBA well)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        frames.append(img)

        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1}/{len(frame_files)} frames...")

    # Calculate frame duration in milliseconds
    duration = int(1000 / fps)

    # Save GIF
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"💾 Saving GIF to {output_path}...")

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,  # Loop forever
        optimize=True,
    )

    # Get file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ GIF created successfully!")
    print(f"   Size: {size_mb:.2f} MB")
    print(f"   Frames: {len(frames)}")
    print(f"   FPS: {fps}")
    print(f"   Resolution: {resize_width}x{new_height}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate demo GIF from tracked frames"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="output/task3.1_video1/tracked",
        help="Directory containing tracked frames",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="assets/demo.gif",
        help="Output path for the GIF",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second (default: 10)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=50,
        help="Maximum frames to include (default: 50)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Width of output GIF (default: 640)",
    )

    args = parser.parse_args()

    success = create_gif(
        frames_dir=args.input,
        output_path=args.output,
        fps=args.fps,
        max_frames=args.max_frames,
        resize_width=args.width,
    )

    if success:
        print(f"\n📝 Add this to your README.md:")
        print(f'   ![Demo](assets/demo.gif)')


if __name__ == "__main__":
    main()
