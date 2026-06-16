"""Generate the final 3-page PDF report (11pt)."""

import argparse
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)


def build():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/home/elicer/project/report.pdf")
    p.add_argument("--wer_json", default="/home/elicer/project/runs/fs2/eval/test_500/wer_summary.json")
    p.add_argument("--wer_jsonl", default="/home/elicer/project/runs/fs2/eval/test_500/wer_results.jsonl")
    p.add_argument("--train_curve", default="/home/elicer/project/report_assets/train_curves.png")
    p.add_argument("--loss_total", default="/home/elicer/project/report_assets/loss_total.png")
    args = p.parse_args()

    doc = SimpleDocTemplate(
        args.out, pagesize=A4,
        leftMargin=1.7 * cm, rightMargin=1.7 * cm,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
        title="Assignment 3 Report", author="Hyoeun Kim",
    )

    styles = getSampleStyleSheet()
    base = ParagraphStyle("base", parent=styles["BodyText"],
                          fontSize=11, leading=14, spaceAfter=4)
    body = ParagraphStyle("body", parent=base)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=14,
                        spaceAfter=4, spaceBefore=6, leading=17)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12,
                        spaceAfter=2, spaceBefore=5, leading=15)
    small = ParagraphStyle("small", parent=base, fontSize=9, leading=12)
    mono = ParagraphStyle("mono", parent=base, fontName="Courier",
                          fontSize=9, leading=11)

    story = []
    story.append(Paragraph("Assignment 3 — Duration-based TTS (FastSpeech2 + HiFi-GAN)", h1))
    story.append(Paragraph(
        "Hyoeun Kim&nbsp;&nbsp;·&nbsp;&nbsp;Spoken Language Processing"
        "&nbsp;&nbsp;·&nbsp;&nbsp;2026", small))

    # 1. Overview
    story.append(Paragraph("1. Overview", h2))
    story.append(Paragraph(
        "I trained a FastSpeech2-style duration-based TTS on LJSpeech "
        "(12,500 train / 100 val / 500 test). Phoneme sequences and per-phoneme "
        "durations come from the provided IPA-compressed manifest, so no forced "
        "alignment was run. At inference the duration <b>predictor</b> determines "
        "frame counts (not the ground-truth labels). The 80-bin mel spectrogram is "
        "vocoded by a pretrained HiFi-GAN (<i>LJ_FT_T2_V1</i>). Only the duration "
        "predictor is used — pitch / energy predictors are omitted, as permitted.",
        body))

    # 2. Model
    story.append(Paragraph("2. Model", h2))
    model_rows = [
        ["Block", "Setting"],
        ["Embedding", "vocab 50 (46 IPA + pad/bos/eos/unk), d=256"],
        ["Encoder", "4 × FFT block: 2-head MHA + ConvFFN (kernel 9), d=256, FF 1024, dropout 0.1"],
        ["Duration predictor", "2 × Conv1d (k=3, d=256) + LayerNorm + ReLU → Linear, dropout 0.5; receives encoder output with stop-gradient"],
        ["Length regulator", "repeat encoder hidden states by integer durations"],
        ["Decoder", "4 × FFT block (same config as encoder)"],
        ["Mel head + postnet", "Linear 256 → 80, then a 5-layer 1-D conv postnet (tanh + BN, residual)"],
    ]
    t = Table(model_rows, colWidths=[3.7 * cm, 13.6 * cm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(t)
    story.append(Spacer(1, 2))
    story.append(Paragraph(
        "Sinusoidal positional encodings are added at both encoder and decoder "
        "inputs. Total trainable parameters: <b>41.49 M</b>. Mel spec follows the "
        "HiFi-GAN configuration (22.05 kHz, n_fft 1024, hop 256, win 1024, 80 mels, "
        "f_min 0, f_max 8000, log magnitude with dynamic-range compression).",
        body))

    # 3. Training
    story.append(Paragraph("3. Training", h2))
    story.append(Paragraph(
        "Phoneme durations (seconds) are converted to integer mel frames via "
        "round(end·sr/hop) − round(start·sr/hop), so the sum of phoneme frames "
        "matches the mel length exactly. Optimizer: AdamW (β = 0.9 / 0.98, ε = 1e-9, "
        "weight decay 1e-6). LR schedule: Noam (d_model 256, warmup 4 000). "
        "Batch size 32, gradient clip 1.0. Three losses are summed: mel L1 before postnet, "
        "mel L1 after postnet, and MSE between predicted log-duration and log(d_gt + 1) "
        "over real phonemes. The duration loss does not back-propagate into the encoder "
        "(encoder output is detached before the duration predictor). 50 000 steps total "
        "(~4.8 h on a single A100 MIG 2g.20gb).",
        body))

    # 3.1 wandb tracking screenshot
    story.append(Paragraph("3.1 Experiment tracking (wandb)", h2))
    story.append(Paragraph(
        "All training metrics were logged to "
        "<a href='https://wandb.ai/hyoeunkim-sungkyunkwan-university/slp-fs2-ljspeech/runs/q5bpag6e'>wandb</a> "
        "(run <font face='Courier'>q5bpag6e</font>): training/validation losses, "
        "learning rate, and gradient norm.", body))

    story.append(Image(args.train_curve, width=17.5 * cm, height=4.3 * cm))
    story.append(Image(args.loss_total, width=11 * cm, height=5.5 * cm))

    # 3.2 final metrics table
    final_rows = [
        ["Metric", "train (step 50 000)", "validation"],
        ["total loss",   "1.024", "1.577"],
        ["mel L1 (before)", "0.476", "0.750"],
        ["mel L1 (after)",  "0.465", "0.752"],
        ["duration MSE",   "0.083", "0.075"],
        ["lr",             "2.80 × 10⁻⁴", "—"],
        ["grad-norm",      "1.07",        "—"],
    ]
    t2 = Table(final_rows, colWidths=[4.3 * cm, 5.5 * cm, 4.5 * cm])
    t2.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]))
    story.append(t2)
    story.append(Spacer(1, 2))

    # 4. Evaluation
    story.append(Paragraph("4. Evaluation (Whisper-large-v3-turbo)", h2))
    with open(args.wer_json) as f:
        wer = json.load(f)
    wer_rows = [
        ["# samples", "WER", "subs.", "ins.", "del.", "hits"],
        [str(wer["n_samples"]), f"{wer['wer']*100:.2f}%",
         str(wer["substitutions"]), str(wer["insertions"]),
         str(wer["deletions"]), str(wer["hits"])],
    ]
    t3 = Table(wer_rows, colWidths=[2.0 * cm, 2.2 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm])
    t3.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(t3)
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "Each of the 500 test utterances was synthesized end-to-end with the "
        "<b>predicted</b> durations and then transcribed by Whisper "
        "(<i>large-v3-turbo</i>, English, FP16). Both reference and hypothesis are "
        "lower-cased and stripped of non-alphanumerics (apostrophes preserved) "
        "before WER is computed with <i>jiwer.process_words</i>.",
        body))

    # 5. Generation samples
    story.append(PageBreak())
    story.append(Paragraph("5. Generation samples (5 specific test utterances)", h2))
    story.append(Paragraph(
        "Synthesized at 22 050 Hz. References and Whisper hypotheses are below; "
        "see <font face='Courier'>samples/</font> in the submission for the wav files.",
        body))

    sample_ids = ["LJ012-0091", "LJ014-0066", "LJ003-0011", "LJ002-0272", "LJ015-0205"]
    results = {}
    with open(args.wer_jsonl) as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in sample_ids:
                results[r["id"]] = r
    sample_rows = [["ID", "Reference", "Whisper hypothesis"]]
    for uid in sample_ids:
        r = results.get(uid)
        if r is None:
            sample_rows.append([uid, "(missing)", ""])
        else:
            sample_rows.append([
                uid,
                Paragraph(r["ref"], small),
                Paragraph(r["hyp"], small),
            ])
    t4 = Table(sample_rows, colWidths=[2.6 * cm, 7.4 * cm, 7.4 * cm], repeatRows=1)
    t4.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t4)

    # 6. Discussion / notes
    story.append(Paragraph("6. Notes", h2))
    story.append(Paragraph(
        "• The val mel-L1 plateaus around 0.75 while train continues to drop "
        "(small over-fitting gap); audio remains intelligible. "
        "• The vocoder <i>LJ_FT_T2_V1</i> was chosen because it was fine-tuned on "
        "Tacotron2-generated mels, which behave more like FastSpeech2 outputs than "
        "ground-truth mels. "
        "• Most WER errors are reasonable phonetic confusions or Whisper "
        "normalizations (e.g. <i>whitecross</i> → <i>white cross</i>, "
        "<i>eighteen fifteen</i> → <i>1815</i>), not artefacts of the TTS.",
        body))

    doc.build(story)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    build()
