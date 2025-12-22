#pip install -U torch transformers matplotlib

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_NAME = "fakespot-ai/roberta-base-ai-text-detection-v1"


@dataclass
class SegmentScore:
    idx: int
    label: str
    ai_percent: float
    text: str


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def split_paragraphs(text: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return parts if parts else ([text.strip()] if text.strip() else [])


def split_words(text: str, words_per_chunk: int, overlap: int) -> List[str]:
    words = text.split()
    if len(words) <= words_per_chunk:
        return [text]
    step = max(1, words_per_chunk - overlap)
    chunks = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + words_per_chunk]).strip()
        if len(chunk.split()) >= 25:
            chunks.append(chunk)
        if start + words_per_chunk >= len(words):
            break
    return chunks if chunks else [text]


class AIDetectionModel:
    def __init__(self, model_name: str = MODEL_NAME, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.ai_label_id = self.infer_ai_label_id()

    def infer_ai_label_id(self) -> int:
        id2label = {int(k): str(v) for k, v in getattr(self.model.config, "id2label", {}).items()}
        for i, lab in id2label.items():
            low = lab.lower()
            if "ai" in low or "generated" in low or "machine" in low or "fake" in low:
                return i
        return 1

    @torch.inference_mode()
    def ai_prob(self, text: str) -> float:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        logits = self.model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)
        return float(probs[self.ai_label_id].item())


def analyze_text(
    text: str,
    model: AIDetectionModel,
    per_paragraph: bool,
    chunk_words: int,
    overlap_words: int,
    aggregate: str
) -> Tuple[float, List[SegmentScore]]:
    text = clean_text(text)
    if not text:
        return 0.0, []
    segments = split_paragraphs(text) if per_paragraph else [text]
    scored: List[SegmentScore] = []
    for i, seg in enumerate(segments, 1):
        chunks = split_words(seg, words_per_chunk=chunk_words, overlap=overlap_words)
        probs = [model.ai_prob(c) for c in chunks]
        if aggregate == "max":
            p = max(probs) if probs else 0.0
        else:
            p = sum(probs) / len(probs) if probs else 0.0
        scored.append(SegmentScore(i, "Paragraph" if per_paragraph else "Text", round(p * 100, 2), seg))
    overall = round(sum(s.ai_percent for s in scored) / len(scored), 2) if scored else 0.0
    return overall, scored


def risk(ai_percent: float) -> str:
    if ai_percent >= 70:
        return "HIGH"
    if ai_percent >= 40:
        return "MEDIUM"
    return "LOW"


def render_report_png(
    overall_ai_percent: float,
    segments: List[SegmentScore],
    out_path: str,
    title: str = "AI-Generated Text Likelihood Report"
) -> str:
    n = max(1, len(segments))
    fig_h = 2.6 + min(0.45 * n, 6.0)
    fig = plt.figure(figsize=(11, fig_h), dpi=160)
    fig.patch.set_alpha(0.0)

    ax = fig.add_axes([0.06, 0.12, 0.92, 0.78])
    ax.axis("off")

    header = f"{title}\nOverall AI likelihood: {overall_ai_percent:.2f}%   (Human: {100-overall_ai_percent:.2f}%)"
    ax.text(0.0, 1.02, header, fontsize=14, fontweight="bold", va="bottom")

    y = 0.92
    line_h = 0.085 if n <= 6 else 0.06
    max_preview = 175

    for s in segments:
        preview = re.sub(r"\s+", " ", s.text).strip()
        preview = preview[:max_preview] + ("…" if len(preview) > max_preview else "")
        row = f"{s.label} {s.idx:>2}:  AI {s.ai_percent:>6.2f}%  Risk {risk(s.ai_percent):<6}  |  {preview}"
        ax.text(0.0, y, row, fontsize=10.5, va="top")
        y -= line_h
        if y < 0.02:
            break

    ax2 = fig.add_axes([0.06, 0.04, 0.92, 0.05])
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.barh([0.5], [overall_ai_percent], height=0.55)
    ax2.text(min(overall_ai_percent + 1, 98), 0.5, f"{overall_ai_percent:.2f}%", va="center", fontsize=10)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return out_path


def read_input_text(args) -> str:
    if args.text:
        return args.text
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            return f.read()
    if not sys.stdin.isatty():
        return sys.stdin.read()
    print("Paste your text. End with an empty line, then press Enter:", flush=True)
    lines = []
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        if line.strip() == "":
            break
        lines.append(line.rstrip("\n"))
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", type=str, default=None)
    p.add_argument("--input-file", type=str, default=None)
    p.add_argument("--output-image", type=str, default="ai_detection_report.png")
    p.add_argument("--output-json", type=str, default=None)
    p.add_argument("--no-paragraphs", action="store_true")
    p.add_argument("--chunk-words", type=int, default=180)
    p.add_argument("--overlap-words", type=int, default=40)
    p.add_argument("--aggregate", type=str, choices=["mean", "max"], default="mean")
    p.add_argument("--model", type=str, default=MODEL_NAME)
    args = p.parse_args()

    text = read_input_text(args)
    text = clean_text(text)
    if not text:
        print("AI likelihood (%): 0.00")
        print("Human likelihood (%): 100.00")
        return

    model = AIDetectionModel(args.model)
    overall, segments = analyze_text(
        text=text,
        model=model,
        per_paragraph=not args.no_paragraphs,
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
        aggregate=args.aggregate
    )

    print(f"AI likelihood (%): {overall:.2f}")
    print(f"Human likelihood (%): {100-overall:.2f}")
    for s in segments:
        print(f"{s.label} {s.idx}: {s.ai_percent:.2f}% ({risk(s.ai_percent)})")

    render_report_png(overall, segments, args.output_image)

    if args.output_json:
        payload = {
            "model": args.model,
            "ai_likelihood_percent": overall,
            "human_likelihood_percent": round(100 - overall, 2),
            "segments": [
                {"index": s.idx, "label": s.label, "ai_percent": s.ai_percent, "risk": risk(s.ai_percent), "text": s.text}
                for s in segments
            ],
            "output_image": args.output_image
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
