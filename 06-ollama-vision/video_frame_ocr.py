"""
Extract frames from a video and transcribe on-screen text with Ollama.

Common use-cases: slide decks recorded as video, screencasts, tutorial recordings,
dashboards, subtitles, and any video containing readable text.

Prerequisites:
    pip install ollama opencv-python pillow
    ollama pull llama3.2-vision   # or: llava
"""

import json
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import cv2
import ollama
from PIL import Image

VISION_MODEL = "llama3.2-vision"


# ── Data types ─────────────────────────────────────────────────────────────────
@dataclass
class FrameResult:
    frame_index: int
    timestamp_s: float
    text: str
    has_text: bool


@dataclass
class VideoOCRReport:
    video_path: str
    total_frames_sampled: int
    frames_with_text: int
    results: list[FrameResult] = field(default_factory=list)

    def to_transcript(self) -> str:
        """Return a plain-text transcript with timestamps."""
        lines = [f"OCR Transcript: {self.video_path}\n{'─' * 50}"]
        for r in self.results:
            if r.has_text:
                ts = _format_timestamp(r.timestamp_s)
                lines.append(f"[{ts}]\n{r.text.strip()}\n")
        return "\n".join(lines)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _format_timestamp(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _frame_to_bytes(frame) -> bytes:
    """Convert an OpenCV BGR frame to JPEG bytes."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)

    # Downscale if large — keeps inference fast
    max_dim = 1024
    w, h = img.size
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _ocr_frame(image_bytes: bytes, model: str = VISION_MODEL) -> tuple[str, bool]:
    """
    Send a frame to the vision model and extract any text present.
    Returns (extracted_text, has_text).
    """
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Extract ALL text visible in this image exactly as it appears. "
                    "If there is no readable text, respond with exactly: NO_TEXT"
                ),
                "images": [image_bytes],
            }
        ],
    )
    text = response["message"]["content"].strip()
    has_text = text.upper() != "NO_TEXT" and len(text) > 0
    return text, has_text


# ── Core pipeline ──────────────────────────────────────────────────────────────
def extract_frames(
    video_path: Path,
    interval_s: float = 5.0,
) -> list[tuple[int, float, bytes]]:
    """
    Sample one frame every `interval_s` seconds.

    Returns list of (frame_index, timestamp_s, jpeg_bytes).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(fps * interval_s))

    frames = []
    frame_idx = 0

    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        timestamp_s = frame_idx / fps
        frames.append((frame_idx, timestamp_s, _frame_to_bytes(frame)))
        frame_idx += step

    cap.release()
    return frames


def ocr_video(
    video_path: Path,
    interval_s: float = 5.0,
    model: str = VISION_MODEL,
    verbose: bool = True,
) -> VideoOCRReport:
    """
    Full pipeline: sample frames → OCR each → return structured report.

    Args:
        video_path:  Path to the input video file.
        interval_s:  Seconds between sampled frames (lower = more coverage, slower).
        model:       Ollama vision model to use.
        verbose:     Print progress to stdout.
    """
    frames = extract_frames(video_path, interval_s)
    report = VideoOCRReport(
        video_path=str(video_path),
        total_frames_sampled=len(frames),
        frames_with_text=0,
    )

    if verbose:
        print(f"Sampled {len(frames)} frames from '{video_path.name}'")

    for i, (frame_idx, timestamp_s, img_bytes) in enumerate(frames, 1):
        if verbose:
            ts = _format_timestamp(timestamp_s)
            print(f"  [{i}/{len(frames)}] t={ts} — running OCR ...", end="\r")

        text, has_text = _ocr_frame(img_bytes, model)

        result = FrameResult(
            frame_index=frame_idx,
            timestamp_s=timestamp_s,
            text=text,
            has_text=has_text,
        )
        report.results.append(result)

        if has_text:
            report.frames_with_text += 1

    if verbose:
        print(f"\nDone. {report.frames_with_text}/{len(frames)} frames contained text.")

    return report


def save_report(report: VideoOCRReport, output_dir: Path = Path(".")) -> None:
    """Save JSON report and plain-text transcript to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(report.video_path).stem

    # JSON — full structured data
    json_path = output_dir / f"{stem}_ocr.json"
    json_path.write_text(
        json.dumps(
            {
                "video": report.video_path,
                "total_frames_sampled": report.total_frames_sampled,
                "frames_with_text": report.frames_with_text,
                "results": [
                    {
                        "frame": r.frame_index,
                        "timestamp_s": r.timestamp_s,
                        "timestamp": _format_timestamp(r.timestamp_s),
                        "has_text": r.has_text,
                        "text": r.text,
                    }
                    for r in report.results
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Plain-text transcript
    txt_path = output_dir / f"{stem}_transcript.txt"
    txt_path.write_text(report.to_transcript(), encoding="utf-8")

    print(f"✓ JSON report  → {json_path}")
    print(f"✓ Transcript   → {txt_path}")


# ── Single-frame convenience function ─────────────────────────────────────────
def ocr_image_file(image_path: Path, model: str = VISION_MODEL) -> str:
    """OCR a single image file (JPEG, PNG, etc.)."""
    img_bytes = image_path.read_bytes()
    text, _ = _ocr_frame(img_bytes, model)
    return text


# ── Demo ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Usage: python video_frame_ocr.py path/to/video.mp4
        video = Path(sys.argv[1])
        interval = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0

        report = ocr_video(video, interval_s=interval)
        print("\n" + report.to_transcript())
        save_report(report, output_dir=Path("./ocr_output"))

    else:
        # Demo: OCR a single screenshot if no video is provided
        print("No video path provided.")
        print("Usage: python video_frame_ocr.py video.mp4 [interval_seconds]")
        print()
        print("For a quick test, create a screenshot and run:")
        print("  python video_frame_ocr.py screen.png")
        print()

        # If a test image exists, demo the single-image OCR
        test_img = Path("test_frame.jpg")
        if test_img.exists():
            print(f"Found {test_img} — running single-frame OCR ...")
            result = ocr_image_file(test_img)
            print(f"\nExtracted text:\n{result}")