"""
extract_reward_frames.py  (AV1-compatible version using PyAV)

LeRobot stores videos as AV1-in-MP4. OpenCV cannot software-decode AV1
on most compute nodes. This script uses PyAV (already in your lerobot env)
which decodes AV1 via ffmpeg software codec.

Usage:
  python extract_reward_frames.py \
    --dataset_root ~/rudra/lerobot/datasets/grasp_place \
    --output_dir ./reward_data \
    --camera cam_right_wrist \
    --last_n_frames 5 \
    --first_n_frames 3

Then manually sort reward_data/to_label/cam_right_wrist/ into:
  reward_data/success/   (object in box, gripper released)
  reward_data/failure/   (object outside box, missed grasp, table hit, etc.)
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import av
    HAS_AV = True
except ImportError:
    HAS_AV = False
    print("[WARN] PyAV not found. Trying: pip install av")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ---------------------------------------------------------------------------
# Core: AV1-safe frame extraction using PyAV
# ---------------------------------------------------------------------------

def extract_frames_pyav(video_path: Path, frame_indices: set, output_dir: Path, prefix: str) -> int:
    """
    Extract specific frame indices from an AV1 video using PyAV.
    Returns number of frames saved.
    Uses sequential decode (no seeking) for AV1 which doesn\'t support random access well.
    """
    if not HAS_AV:
        raise RuntimeError("PyAV not installed. Run: pip install av")

    saved = 0
    target = set(frame_indices)
    max_target = max(target) if target else 0

    try:
        container = av.open(str(video_path))
        video_stream = container.streams.video[0]
        # Force software decoding — disable hardware acceleration
        video_stream.codec_context.thread_type = av.codec.context.ThreadType.AUTO

        for frame in container.decode(video=0):
            fi = frame.index
            if fi in target:
                # Convert AV frame → numpy BGR (for cv2/PIL saving)
                img_rgb = frame.to_ndarray(format="rgb24")  # (H, W, 3) uint8
                out_path = output_dir / f"{prefix}_frame{fi:05d}.jpg"

                if HAS_CV2:
                    img_bgr = img_rgb[:, :, ::-1]  # RGB → BGR for cv2
                    cv2.imwrite(str(out_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                elif HAS_PIL:
                    Image.fromarray(img_rgb).save(str(out_path), quality=95)
                else:
                    # Fallback: use ffmpeg subprocess for just this frame
                    _save_frame_ffmpeg(video_path, fi, out_path)

                saved += 1

            if fi > max_target:
                break  # no need to decode the rest of the video

        container.close()

    except Exception as e:
        print(f"    [ERROR] PyAV failed on {video_path.name}: {e}")
        # Fallback to ffmpeg subprocess
        saved = extract_frames_ffmpeg_subprocess(video_path, frame_indices, output_dir, prefix)

    return saved


def extract_frames_ffmpeg_subprocess(
    video_path: Path, frame_indices: set, output_dir: Path, prefix: str
) -> int:
    """
    Fallback: use ffmpeg subprocess to extract frames one at a time.
    Slower but completely reliable for any codec.
    """
    import subprocess
    saved = 0
    for fi in sorted(frame_indices):
        out_path = output_dir / f"{prefix}_frame{fi:05d}.jpg"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"select=eq(n\\,{fi})",
            "-vframes", "1",
            "-q:v", "2",  # high quality JPEG
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0 and out_path.exists():
            saved += 1
        else:
            stderr = result.stderr.decode("utf-8", errors="replace")
            print(f"    [WARN] ffmpeg frame {fi}: {stderr[-200:]}")
    return saved


def _save_frame_ffmpeg(video_path: Path, frame_idx: int, out_path: Path):
    import subprocess
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"select=eq(n\\,{frame_idx})",
        "-vframes", "1", "-q:v", "2", str(out_path),
    ], capture_output=True)


# ---------------------------------------------------------------------------
# Dataset metadata parsing
# ---------------------------------------------------------------------------

def load_all_parquets(directory: Path) -> pd.DataFrame:
    dfs = []
    for chunk_dir in sorted(directory.glob("chunk-*")):
        for pq in sorted(chunk_dir.glob("*.parquet")):
            dfs.append(pd.read_parquet(pq))
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def get_episode_to_chunk_map(dataset_root: Path):
    """
    Returns: {episode_idx: (chunk_file_idx, parquet_path)}
    Each meta/episodes/chunk-000/file-NNN.parquet corresponds to
    videos/*/chunk-000/file-NNN.mp4
    """
    meta_dir = dataset_root / "meta" / "episodes" / "chunk-000"
    ep_to_chunk = {}
    for i, pq in enumerate(sorted(meta_dir.glob("*.parquet"))):
        df = pd.read_parquet(pq)
        if "episode_index" in df.columns:
            for ep in df["episode_index"].unique():
                ep_to_chunk[int(ep)] = (i, pq)
    return ep_to_chunk


def get_episode_frame_range(dataset_root: Path, episode_idx: int, chunk_file_idx: int):
    """
    Returns (start_frame, end_frame) as indices into the video file.
    The data parquet files mirror the video structure: each file-NNN.parquet
    contains the frame data for the corresponding file-NNN.mp4.
    """
    data_pq = dataset_root / "data" / "chunk-000" / f"file-{chunk_file_idx:03d}.parquet"
    if not data_pq.exists():
        print(f"    [WARN] Data parquet not found: {data_pq}")
        return None, None

    df = pd.read_parquet(data_pq)
    if "episode_index" not in df.columns:
        # No episode_index column — return full file range
        return 0, len(df)

    ep_mask = df["episode_index"] == episode_idx
    if not ep_mask.any():
        return None, None

    # Row positions in parquet = frame positions in the corresponding video
    indices = np.where(ep_mask.values)[0]
    return int(indices[0]), int(indices[-1] + 1)  # [start, end)


def get_video_total_frames_pyav(video_path: Path) -> int:
    """Get total frame count using PyAV without decoding all frames."""
    try:
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        frames = stream.frames  # may be 0 if not in container metadata
        if frames == 0:
            # Count via duration and fps
            fps = float(stream.average_rate)
            duration = float(stream.duration * stream.time_base)
            frames = int(fps * duration)
        container.close()
        return frames
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from LeRobot AV1 videos for reward classifier labelling"
    )
    parser.add_argument("--dataset_root", type=str,
                        default=str(Path.home() / "rudra/lerobot/datasets/grasp_place"))
    parser.add_argument("--output_dir", type=str, default="./reward_data")
    parser.add_argument("--camera", type=str, default="cam_right_wrist",
                        choices=["cam_right_wrist", "cam_high"])
    parser.add_argument("--last_n_frames",  type=int, default=5,
                        help="Last N frames per episode → SUCCESS candidates")
    parser.add_argument("--first_n_frames", type=int, default=3,
                        help="First N frames per episode → FAILURE candidates")
    parser.add_argument("--mid_n_frames",   type=int, default=3,
                        help="N uniformly sampled mid-episode frames → FAILURE candidates")
    parser.add_argument("--episodes",       type=int, nargs="+", default=None,
                        help="Only extract these episode indices (default: all)")
    args = parser.parse_args()

    if not HAS_AV:
        print("ERROR: PyAV is required. Install with: pip install av")
        print("(It should already be in your lerobot conda env)")
        return

    dataset_root = Path(args.dataset_root).expanduser()
    output_dir   = Path(args.output_dir)
    camera_key   = f"observation.images.{args.camera}"
    video_base   = dataset_root / "videos" / camera_key / "chunk-000"

    to_label_dir = output_dir / "to_label" / args.camera
    to_label_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "success").mkdir(parents=True, exist_ok=True)
    (output_dir / "failure").mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Dataset     : {dataset_root}")
    print(f"[INFO] Camera      : {camera_key}")
    print(f"[INFO] Video dir   : {video_base}")
    print(f"[INFO] Output      : {to_label_dir}")
    print(f"[INFO] Decoder     : PyAV (AV1 software decode)")

    # Check if video dir exists and list files
    if not video_base.exists():
        print(f"[ERROR] Video directory not found: {video_base}")
        return

    video_files = sorted(video_base.glob("*.mp4"))
    print(f"[INFO] Found {len(video_files)} video file(s): {[v.name for v in video_files]}")

    # Try to load episode → chunk mapping from metadata
    ep_to_chunk = get_episode_to_chunk_map(dataset_root)

    if not ep_to_chunk:
        print("[WARN] No episode metadata found in meta/episodes/. "
              "Falling back to whole-video extraction.")
        _extract_whole_videos(video_files, to_label_dir, args)
        _print_guide(output_dir)
        return

    print(f"[INFO] Found {len(ep_to_chunk)} episodes in metadata")

    # Filter episodes if requested
    episodes_to_process = sorted(ep_to_chunk.keys())
    if args.episodes:
        episodes_to_process = [e for e in episodes_to_process if e in args.episodes]
        print(f"[INFO] Processing episodes: {episodes_to_process}")

    total_saved = 0
    for ep_idx in episodes_to_process:
        chunk_idx, _ = ep_to_chunk[ep_idx]
        video_path = video_base / f"file-{chunk_idx:03d}.mp4"

        if not video_path.exists():
            print(f"  [SKIP] ep{ep_idx:03d}: video {video_path.name} not found")
            continue

        start, end = get_episode_frame_range(dataset_root, ep_idx, chunk_idx)
        if start is None:
            print(f"  [SKIP] ep{ep_idx:03d}: could not find frame range in parquet")
            continue

        ep_len = end - start
        if ep_len <= 0:
            print(f"  [SKIP] ep{ep_idx:03d}: empty episode (start={start}, end={end})")
            continue

        # Build target frame sets (absolute indices in the video file)
        last_frames = set(
            range(max(start, end - args.last_n_frames), end)
        )
        first_frames = set(
            range(start, min(start + args.first_n_frames, end))
        )
        mid_frames = set()
        for k in range(1, args.mid_n_frames + 1):
            mid_frames.add(start + ep_len * k // (args.mid_n_frames + 1))

        # Save with descriptive prefixes so you know which to put in success/failure
        n1 = extract_frames_pyav(video_path, last_frames,  to_label_dir, f"ep{ep_idx:03d}_LAST")
        n2 = extract_frames_pyav(video_path, first_frames, to_label_dir, f"ep{ep_idx:03d}_FIRST")
        n3 = extract_frames_pyav(video_path, mid_frames,   to_label_dir, f"ep{ep_idx:03d}_MID")

        total_saved += n1 + n2 + n3
        print(f"  ep{ep_idx:03d}  (video={video_path.name}, frames {start}–{end-1}, "
              f"len={ep_len}):  {n1} last + {n2} first + {n3} mid  →  {n1+n2+n3} saved")

    print(f"\\n{'='*60}")
    print(f"[DONE] {total_saved} frames saved to: {to_label_dir}")
    print(f"{'='*60}")
    _print_guide(output_dir)


def _extract_whole_videos(video_files, to_label_dir, args):
    """Fallback when no episode metadata is available."""
    total_saved = 0
    for video_path in video_files:
        stem = video_path.stem
        total = get_video_total_frames_pyav(video_path)
        if total == 0:
            print(f"  [WARN] Could not determine frame count for {video_path.name}, decoding all...")
            total = 99999

        print(f"  {video_path.name}: {total} frames (estimated)")

        last_frames  = set(range(max(0, total - args.last_n_frames), total))
        first_frames = set(range(min(args.first_n_frames, total)))
        mid_frames   = {total // 4, total // 2, 3 * total // 4}

        n1 = extract_frames_pyav(video_path, last_frames,  to_label_dir, f"{stem}_LAST")
        n2 = extract_frames_pyav(video_path, first_frames, to_label_dir, f"{stem}_FIRST")
        n3 = extract_frames_pyav(video_path, mid_frames,   to_label_dir, f"{stem}_MID")
        saved = n1 + n2 + n3
        total_saved += saved
        print(f"    Saved {saved} frames")

    print(f"\\n[DONE] {total_saved} total frames saved")


def _print_guide(output_dir: Path):
    print(f"""
┌─────────────────────────────────────────────────────────────┐
│  LABELLING GUIDE                                            │
├─────────────────────────────────────────────────────────────┤
│  Review frames in:                                          │
│    {str(output_dir / "to_label"):<55} │
│                                                             │
│  Frame name key:                                            │
│    ep###_LAST_*   = last frames of episode (end state)      │
│    ep###_FIRST_*  = first frames (pre-grasp, always fail)   │
│    ep###_MID_*    = mid-episode (transition frames)         │
│                                                             │
│  → SUCCESS (copy to reward_data/success/):                  │
│    ep###_LAST frames where object IS in the box             │
│    Gripper open/released, object settled in container       │
│                                                             │
│  → FAILURE (copy to reward_data/failure/):                  │
│    ALL ep###_FIRST frames (object not touched yet)          │
│    ep###_LAST frames where task FAILED (not in box)         │
│    ep###_MID frames where gripper missed or collided        │
│    Any frame showing dangerous contact / table hit          │
│                                                             │
│  Target: 25–40 success + 25–40 failure images               │
│  (one camera only — cam_right_wrist recommended)            │
└─────────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    main()