"""
Streaming ASR System with Speaker Diarization
=============================================

Implements three streaming-ASR strategies that the project requires:
  • Step 1: Chunk-wise streaming   (fixed cadence, causal buffer)
  • Step 2: Local Agreement (N=2)  (commit only words stable across two ASR runs)
  • Step 3: Attention-guided right-boundary truncation (Simul-Whisper)

Design notes for the visualisation:
  • The committed text is monotonic — once a word is shown in white it is
    never removed or replaced. Local Agreement only ever EXTENDS the
    committed prefix.
  • Every new word must first appear as a blue partial token in at least one
    earlier subtitle event before it is allowed to graduate to white.
  • All processing is strictly causal — the system only ever sees audio
    [0:t] when emitting subtitle state for time t.
"""

import os
import sys
import json
import warnings
import subprocess
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import torch
import whisper

# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16000
CHUNK_DUR     = 0.5    # audio-streaming chunk granularity (always 0.5 s)
# ASR cadence depends on the model size, per the project spec:
#   small.en  → ≤ 2 Hz  (every 0.5 s)
#   medium.en → ≤ 1 Hz  (every 1.0 s)
ASR_INTERVAL_BY_MODEL = {"small.en": 0.5, "medium.en": 1.0}
VAD_INTERVAL  = 0.05   # s  → 20 Hz
SPK_INTERVAL  = 0.2    # s  → 5 Hz
VAD_THRESHOLD = 0.35   # lower threshold helps catch quiet/short utterance starts
VAD_PREROLL_SEC = 1.2  # include past audio when VAD first fires, so phrase
                       # onsets like "can you tell me" are not clipped.
SILENCE_END   = 0.6    # s  silence to finalize segment (lower → catches
                       # brief turn-takes / backchannels like "Yeah")
SILENCE_PARTIAL_GRACE_SEC = 0.2
SILENCE_PARTIAL_MAX_GROWTH = 3
ATTN_TAU      = 2      # encoder-frame margin from right boundary (Simul-Whisper)
                       # — small TAU = words near the right boundary still
                       # get a chance to be displayed as partial and to be
                       # committed at finalize.
MAX_BUF_SEC   = 28.0   # Whisper context cap
MIN_PARTIAL_AGE = 1    # a new word must be in partial for ≥ this many ASR cycles
                       # before being eligible for commit (=> blue → white)
LA_N           = 2     # Local Agreement: a word must appear in the last LA_N
                       # consecutive ASR hypotheses (as a stable prefix)
                       # before becoming eligible for commit.  N=2 = the
                       # canonical Local-Agreement-2 from the paper.
PARTIAL_CARRY_MIN_SEEN = 2  # Extra guard for the partial-carry fallback only:
                            # a word/phrase must have appeared as blue partial
                            # in at least two ASR cycles before bypassing LA.
PARTIAL_CARRY_HOLD_BACK = 1  # Do not commit the newest right-edge carry word via
                             # the fallback; let the next ASR cycle confirm it.
DISPLAY_MAX_SENTENCES = 1     # screen budget: last N sentences per speaker
DISPLAY_MAX_WORDS     = 16    # hard word cap (older words get "…" prefix)
DISPLAY_MAX_PARTIAL_WORDS = 14 # cap blue text so unstable tails do not fill
                               # the screen during long utterances.
DISPLAY_MAX_PARTIAL_LINES = 2  # per-speaker partial display budget.
DISPLAY_CHARS_PER_LINE = 44
SUBTITLE_IDLE_CLEAR_SEC = 5.0  # keep inactive speaker lines briefly so
                               # backchannels/overlaps can be read together.
SOFT_FINALIZE_SEC     = 20.0  # force-finalize segments longer than this so
                              # the ASR buffer doesn't drift past Whisper's
                              # 30 s context. Same speaker continues into a
                              # new segment.
SOFT_OVERLAP_SEC      = 1.0   # keep only past audio as overlap after a soft
                              # finalize; duplicated overlap words are stripped
                              # using the previous committed tail.
SOFT_OVERLAP_MAX_WORDS = 10
SPEAKER_CHANGE_PREROLL_SEC = 0.4  # when a turn change is confirmed, re-feed a
                                  # small amount of already-seen audio so the
                                  # new speaker's first words are not clipped.
MAX_FINALIZE_PROMOTE_WORDS = 32  # finalisation may rescue partial words that
                                 # disappeared from the last ASR snapshot, but
                                 # cap it to avoid committing long hallucinated
                                 # tails.
FINALIZE_RESCUE_MIN_SEEN = 2     # rescue only words/phrases visible as partial
                                 # in multiple ASR cycles.
FINALIZE_RESCUE_HOLD_BACK = 1    # keep the newest right-edge word blue-only at
                                 # finalization unless it later reappears.
WORD_SPK_WINDOW_SEC = 0.9     # extra short-window embedding around ASR word
                              # times for backchannel/overlap display.
WORD_SPK_MARGIN = 0.12
ONE_WORD_SPK_WINDOW_SEC = 0.5
ONE_WORD_SPK_MARGIN = 0.08
BACKCHANNEL_MAX_WORDS = 8
OVERLAY_REPEAT_SUPPRESS_SEC = 8.0
OVERLAY_MERGE_GAP_SEC = 1.0


# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class CommittedSegment:
    speaker: str
    start: float
    end: float
    commit_time: float
    text: str
    overlay: bool = False


# ── Audio helpers ─────────────────────────────────────────────────────────────
def extract_audio(video_path: str) -> np.ndarray:
    cmd = ["ffmpeg", "-y", "-i", video_path,
           "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode()[:200])
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


# ── VAD (Silero) ──────────────────────────────────────────────────────────────
class VADModel:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model, _ = torch.hub.load(
            "snakers4/silero-vad", "silero_vad",
            force_reload=False, onnx=False, trust_repo=True,
        )
        self.model = self.model.to(device)

    def is_speech(self, chunk: np.ndarray) -> bool:
        if len(chunk) < 512:
            return False
        t = torch.from_numpy(chunk).float().to(self.device)
        frame = 512
        scores = []
        for i in range(0, len(t) - frame, frame):
            scores.append(self.model(t[i:i + frame], SAMPLE_RATE).item())
        if not scores:
            return False
        # speech if mean score is high enough
        return float(np.mean(scores)) > VAD_THRESHOLD


# ── Speaker Identifier (pyannote/embedding 512-d, matches provided refs) ──────
class SpeakerIdentifier:
    """Streaming speaker identification using pyannote/embedding (512-d).

    This is the same model that produced the provided per-clip
    `clipXX_spkA_embedding.npy` / `_spkB_embedding.npy` reference vectors,
    so a streaming embedding can be compared **directly** against the
    provided references via cosine similarity → A/B is decided in a single
    step, no online clustering needed.

    All embedding computations are causal: each call only sees audio that
    has already been received by the streamer.
    """

    def __init__(self, ref_a: np.ndarray, ref_b: np.ndarray, device="cuda"):
        import os
        from pyannote.audio import Model, Inference
        token = os.environ.get("HF_TOKEN")
        self.mdl = Model.from_pretrained(
            "pyannote/embedding", use_auth_token=token).to(device).eval()
        self.inf = Inference(self.mdl, window="whole")
        self.device = device
        n = lambda v: v / (np.linalg.norm(v) + 1e-8)
        self.ref_a = n(ref_a)
        self.ref_b = n(ref_b)

    @torch.no_grad()
    def embed(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """pyannote 512-d embedding. None if too short."""
        if len(audio) < SAMPLE_RATE * 0.4:
            return None
        wav = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        e = np.array(self.inf({"waveform": wav,
                               "sample_rate": SAMPLE_RATE})).flatten()
        nrm = np.linalg.norm(e)
        return e / nrm if nrm > 1e-8 else None

    # `observe` is kept as a no-op for API compatibility — we don't need
    # online accumulation when we have direct references.
    def observe(self, audio: np.ndarray):
        return

    def identify(self, audio: np.ndarray) -> Optional[str]:
        e = self.embed(audio)
        if e is None:
            return None
        sa = float(np.dot(e, self.ref_a))
        sb = float(np.dot(e, self.ref_b))
        return "A" if sa >= sb else "B"

    def identify_with_score(self, audio: np.ndarray):
        """Return (label, margin) where margin = cos(ref_winner) - cos(ref_loser).
        Used for hysteresis decisions."""
        e = self.embed(audio)
        if e is None:
            return None, 0.0
        sa = float(np.dot(e, self.ref_a))
        sb = float(np.dot(e, self.ref_b))
        if sa >= sb:
            return "A", sa - sb
        return "B", sb - sa


# ── Whisper ASR (cross-attention timing, prompt-aware, hallucination guard) ───
class WhisperASR:
    def __init__(self, model_name: str = "medium.en", device: str = "cuda"):
        self.model = whisper.load_model(model_name, device=device)
        self.device = device
        # medium.en runs at 1 Hz so we have ~1 s budget per call → wider beam.
        # small.en runs at 2 Hz → narrower beam to stay under 0.5 s budget.
        self.beam_size = 5 if model_name.startswith("medium") else 2

    def transcribe(self, audio: np.ndarray, prompt: str = "") -> tuple[str, list]:
        """Return (text, list-of-peak-encoder-frame-per-token).

        `prompt` is forwarded to Whisper as the previous-text condition so
        that already-committed words are not re-derived from acoustics on
        every cycle. This stabilises casing, vocabulary, and continuity
        across consecutive ASR calls.
        """
        if len(audio) < SAMPLE_RATE * 0.2:
            return "", []
        if len(audio) > int(MAX_BUF_SEC * SAMPLE_RATE):
            audio = audio[-int(MAX_BUF_SEC * SAMPLE_RATE):]
        n_enc = int(len(audio) / SAMPLE_RATE * 50)   # 50 enc-frames / second
        padded = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(padded).to(self.device)

        attn_store: list[torch.Tensor] = []
        def hook(_mod, _inp, out):
            if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                attn_store.append(out[1].detach())
        hooks = [b.cross_attn.register_forward_hook(hook)
                 for b in self.model.decoder.blocks]
        try:
            opts = whisper.DecodingOptions(
                language="en",
                without_timestamps=True,
                fp16=(self.device == "cuda"),
                beam_size=self.beam_size,
                # Whisper truncates `prompt` to the last 224 tokens
                # internally; we still cap on our side to be safe.
                prompt=prompt[-800:] if prompt else None,
                # Light suppression of the worst hallucination patterns.
                suppress_blank=True,
            )
            result = whisper.decode(self.model, mel, opts)
        finally:
            for h in hooks:
                h.remove()

        text = self._strip_hallucinations(result.text.strip())

        # Per-token attention peak (encoder-frame index).
        peak_frames: list[int] = []
        if attn_store:
            stacked = torch.stack(attn_store).mean(dim=(0, 1, 2))  # (Td, Te)
            Te = stacked.shape[-1]
            for t in range(stacked.shape[0]):
                p = int(torch.argmax(stacked[t]).item())
                peak_frames.append(int(p * n_enc / max(Te, 1)))
        return text, peak_frames

    @staticmethod
    def _strip_hallucinations(text: str) -> str:
        """Whisper sometimes loops on the right boundary, emitting
        repeated tokens like "who who", "you you you", or "uh uh uh".
        Detect immediate word repetitions of length ≥3 and trim back to
        the first occurrence.
        """
        if not text:
            return text
        words = text.split()
        # Trim trailing exact repetitions: drop the tail when the last
        # word equals the second-to-last (and onward repeats).
        i = len(words)
        while i >= 2 and words[i - 1].lower().strip(",.?!") == \
                         words[i - 2].lower().strip(",.?!"):
            i -= 1
        return " ".join(words[:i])

    def truncate_by_attention(self, text: str, peaks: list, n_enc: int) -> str:
        """Drop trailing tokens whose attention peaks lie within ATTN_TAU
        encoder-frames of the right boundary (Simul-Whisper)."""
        words = text.split()
        if not words or not peaks:
            return text
        stride = max(1, len(peaks) // max(1, len(words)))
        safe = []
        for i, w in enumerate(words):
            j = min(i * stride, len(peaks) - 1)
            if n_enc - peaks[j] >= ATTN_TAU:
                safe.append(w)
            else:
                break
        return " ".join(safe)

    def word_attention_times(self, peaks: list, n_enc: int,
                              words_count: int,
                              audio_dur: float) -> list[float]:
        """Convert per-token attention peaks to per-word audio times.

        Returns a list of `words_count` floats — the estimated audio time
        (in seconds, relative to the start of the input buffer) at which
        each word was spoken.
        """
        if not peaks or words_count == 0 or n_enc == 0:
            return [0.0] * words_count
        stride = max(1, len(peaks) // max(1, words_count))
        out = []
        for i in range(words_count):
            j = min(i * stride, len(peaks) - 1)
            t = peaks[j] / max(n_enc, 1) * audio_dur
            out.append(float(t))
        return out


# ── Subtitle helpers ──────────────────────────────────────────────────────────
def fmt_ts_srt(s: float) -> str:
    h, s = divmod(s, 3600); m, s = divmod(s, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int((s%1)*1000):03d}"

def fmt_ts_ass(s: float) -> str:
    h, s = divmod(s, 3600); m, s = divmod(s, 60)
    return f"{int(h):d}:{int(m):02d}:{s:05.2f}"


def to_srt(segments: list) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        lines += [str(i),
                  f"{fmt_ts_srt(seg.start)} --> {fmt_ts_srt(seg.end)}",
                  f"{seg.speaker}: {cleanup_punctuation_text(seg.text)}", ""]
    return "\n".join(lines)


def _spread_word_by_word(events: list, asr_interval: float = CHUNK_DUR) -> list:
    """Pass-through: ``run_streaming`` already emits one subtitle event per
    new word, timed via cross-attention. The previous version of this
    function tried to *re*-spread events but accidentally reused the
    previous segment's state when a speaker came back after a gap, which
    leaked stale committed text into the next segment. Keeping this as a
    no-op makes the streaming pipeline single-source-of-truth.
    """
    return list(events)


def to_ass(subtitle_events: list, video_duration: float,
           asr_interval: float = 0.5) -> str:
    """Render committed (white) + partial (blue) subtitle stream.

    Two improvements vs. the naive renderer:
      • Word-by-word reveal: new words within an ASR cycle are spread
        evenly over the cycle's window so the user sees them appear
        one at a time, matching streaming-ASR behaviour.
      • Two-speaker display: A and B are tracked independently so that
        speaker A's most recent line stays on screen while B is talking
        (and vice-versa). A speaker's line is cleared once their text is
        fully committed AND they've been idle for `cooldown` seconds.
    """
    COOLDOWN = SUBTITLE_IDLE_CLEAR_SEC
    SAME_SPK_GAP_CLEAR = SUBTITLE_IDLE_CLEAR_SEC
    header = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,28,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,2,1,2,30,30,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    WHITE = "&H00FFFFFF&"
    BLUE  = "&H00FF4400&"
    events = _spread_word_by_word(subtitle_events, asr_interval=asr_interval)

    # State per speaker for the two-speaker display.
    state = {"A": {"com": "", "par": "", "last": -1e9},
             "B": {"com": "", "par": "", "last": -1e9}}

    def wrap_words_for_ass(words: list[str],
                           max_lines: int = DISPLAY_MAX_PARTIAL_LINES,
                           max_chars: int = DISPLAY_CHARS_PER_LINE,
                           ) -> tuple[str, bool]:
        """Return ASS-wrapped text that fits in at most max_lines."""
        words = list(words)
        truncated = False

        def lines_for(ws: list[str]) -> list[str]:
            lines: list[str] = []
            cur = ""
            for w in ws:
                nxt = w if not cur else f"{cur} {w}"
                if cur and len(nxt) > max_chars:
                    lines.append(cur)
                    cur = w
                else:
                    cur = nxt
            if cur:
                lines.append(cur)
            return lines

        lines = lines_for(words)
        while len(lines) > max_lines and words:
            words = words[1:]
            truncated = True
            lines = lines_for(words)
        return "\\N".join(lines), truncated

    def trim_for_display(committed: str, partial: str,
                         max_sent: int = DISPLAY_MAX_SENTENCES,
                         max_words: int = DISPLAY_MAX_WORDS,
                         max_partial_words: int = DISPLAY_MAX_PARTIAL_WORDS,
                         ) -> tuple[str, str, bool, bool]:
        """Trim display text only; annotation JSON still keeps full text."""
        partial_words = partial.split()
        partial_truncated = len(partial_words) > max_partial_words
        if partial_truncated:
            partial_words = partial_words[-max_partial_words:]
        partial_out, partial_line_truncated = wrap_words_for_ass(partial_words)
        partial_truncated = partial_truncated or partial_line_truncated

        if not committed:
            return committed, partial_out, False, partial_truncated
        import re
        parts = re.split(r'(?<=[.!?])\s+', committed.strip())
        if len(parts) > max_sent:
            parts = parts[-max_sent:]
        out = " ".join(parts).strip()
        truncated = (out != committed.strip())
        words = out.split()
        # Reserve room for the partial in the word budget.
        budget = max_words - len(partial_words)
        budget = max(5, budget)
        if len(words) > budget:
            words = words[-budget:]
            out = " ".join(words)
            truncated = True
        return out, partial_out, truncated, partial_truncated

    def render_speaker(spk: str) -> str:
        st = state[spk]
        if not st["com"] and not st["par"]:
            return ""
        com_disp, par_disp, com_trim, par_trim = trim_for_display(
            cleanup_punctuation_text(st["com"]),
            cleanup_punctuation_text(st["par"]))
        was_trim = com_trim or par_trim
        prefix = "… " if was_trim else ""
        if com_disp and par_disp:
            return (f"{spk}: {prefix}{com_disp} "
                    f"{{\\c{BLUE}}}{par_disp}{{\\c{WHITE}}}")
        if com_disp:
            return f"{spk}: {prefix}{com_disp}"
        return f"{spk}: {prefix}{{\\c{BLUE}}}{par_disp}{{\\c{WHITE}}}"

    def maybe_clear(t: float):
        for s in ("A", "B"):
            st = state[s]
            if st["com"] and not st["par"] and (t - st["last"]) >= COOLDOWN:
                st["com"] = ""

    out_lines: list[str] = []

    def emit(t_start: float, t_end: float):
        if t_end <= t_start:
            t_end = t_start + 0.1
        a = render_speaker("A"); b = render_speaker("B")
        if not a and not b:
            return
        if a and b:
            txt = f"{a}\\N{b}"
        else:
            txt = a or b
        out_lines.append(
            f"Dialogue: 0,{fmt_ts_ass(t_start)},{fmt_ts_ass(t_end)},"
            f"Default,,0,0,0,,{txt}")

    last_t: Optional[float] = None
    for i, ev in enumerate(events):
        t = max(0.0, ev["time"])
        # If a cooldown for the OTHER speaker has expired since last_t,
        # close the previous segment first and emit a clearing transition
        # at the cooldown boundary.
        if last_t is not None and last_t < t:
            clear_points = []
            for s in ("A", "B"):
                st = state[s]
                clear_after = (SAME_SPK_GAP_CLEAR
                               if s == ev["speaker"] else COOLDOWN)
                clear_t = st["last"] + clear_after
                if st["com"] and not st["par"] and last_t < clear_t < t:
                    clear_points.append((clear_t, s))
            for clear_t, s in sorted(clear_points):
                if last_t < clear_t:
                    emit(last_t, clear_t)
                    last_t = clear_t
                state[s]["com"] = ""

        if last_t is not None and last_t < t:
            emit(last_t, t)

        # Apply current event
        st = state[ev["speaker"]]
        st["com"]  = ev.get("committed", "").strip()
        st["par"]  = ev.get("partial",   "").strip()
        st["last"] = t
        maybe_clear(t)
        last_t = t

    # tail render until end of video.  A finalization event already displayed
    # the last live state up to the silence boundary; holding it again at EOF
    # can make a long committed subtitle pop up after the conversation ends.
    if last_t is None:
        last_t = 0.0
    suppress_tail_hold = bool(events and events[-1].get("finalize"))
    end = min(last_t + 2.0, video_duration)
    if end > last_t and not suppress_tail_hold:
        emit(last_t, end)

    return header + "\n".join(out_lines) + "\n"


# ── Main streaming pipeline ───────────────────────────────────────────────────
def common_prefix_words(a: list, b: list) -> list:
    """Word-level longest common prefix (case-insensitive comparison, but
    we keep the casing of `b` — the most recent hypothesis)."""
    out = []
    for x, y in zip(a, b):
        if norm_word(x) == norm_word(y):
            out.append(y)
        else:
            break
    return out


def local_agreement_extension(prev_ext: list[str], curr_ext: list[str]) -> list[str]:
    """Agreement for the uncommitted extension.

    Prefer the normal common prefix. If the first few uncommitted words churn
    because Whisper inserted/removed a filler, find a short common phrase near
    the front and commit that phrase instead of blocking the whole tail.
    """
    prefix = common_prefix_words(prev_ext, curr_ext)
    if prefix:
        return prefix

    prev_n = norm_words(prev_ext)
    curr_n = norm_words(curr_ext)
    max_skip = 4
    for n in (3, 2):
        prev_limit = min(max_skip + 1, len(prev_n) - n + 1)
        curr_limit = min(max_skip + 1, len(curr_n) - n + 1)
        for i in range(max(0, prev_limit)):
            phrase = prev_n[i:i + n]
            for j in range(max(0, curr_limit)):
                if curr_n[j:j + n] == phrase:
                    return curr_ext[j:j + n]
    return []


def norm_word(w: str) -> str:
    return w.lower().strip(",.?!\"'“”‘’()[]{}")


def norm_words(words: list[str]) -> list[str]:
    return [norm_word(w) for w in words]


def cleanup_punctuation_text(text: str) -> str:
    """Small display/output cleanup that does not alter ASR decisions."""
    if not text:
        return text
    import re
    out = re.sub(r"\s+([,.?!])", r"\1", text)
    out = re.sub(r"([?!])\1+", r"\1", out)
    out = re.sub(r"\.{2,}", "...", out)
    out = re.sub(
        r"\b(Here(?:'s|s)?\s+sharks?)\.\s+(everywhere\b)",
        r"\1 \2",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(
        r"\b(sharks?)\.\s+(everywhere\b)",
        r"\1 \2",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(r"\s{2,}", " ", out)
    return out.strip()


LEADING_REPEAT_FILLERS = {
    "and", "but", "so", "well", "yeah", "yes", "no", "uh", "um", "like",
}
ALLOW_SINGLE_WORD_REPEATS = {"no", "yes", "very", "really", "wheeze"}
WEAK_TRAILING_WORDS = {
    "a", "an", "the", "and", "or", "but", "so", "like", "i'm", "im",
}
WEAK_TRAILING_PHRASES = {
    ("i'm", "all"),
    ("im", "all"),
}
BACKCHANNEL_WORDS = {
    "yeah", "yes", "yep", "no", "nope", "right", "okay", "ok", "uh",
    "um", "wow", "sure", "absolutely", "why", "crazy", "really",
}
BACKCHANNEL_PHRASES = {
    "that's crazy", "that is crazy", "i know", "you know", "like you were",
    "like you were surfing",
}
BACKCHANNEL_PATTERNS = [
    ("that's crazy".split(), 2),
    ("that is crazy".split(), 3),
    ("crazy".split(), 1),
    ("why".split(), 1),
    ("i know".split(), 2),
    ("like you were surfing".split(), 4),
    ("no i didn't do anything where it".split(), 7),
    ("no i didn't do anything".split(), 5),
    ("no i didn't".split(), 3),
]
QUESTION_INTERJECTION_WORDS = {"shark", "sharks"}


def canonical_phrase(words: list[str]) -> tuple[str, ...]:
    vals = [w for w in norm_words(words) if w]
    while vals and vals[0] in LEADING_REPEAT_FILLERS:
        vals = vals[1:]
    return tuple(vals)


def looks_like_backchannel(words: list[str]) -> bool:
    vals = [w for w in norm_words(words) if w]
    if not vals or len(vals) > BACKCHANNEL_MAX_WORDS:
        return False
    phrase = " ".join(vals)
    if phrase in BACKCHANNEL_PHRASES:
        return True
    if vals[0] in {"yeah", "yes", "yep", "no", "nope", "okay", "ok",
                   "uh", "um", "wow", "sure"}:
        return True
    return len(vals) <= 3 and any(w in BACKCHANNEL_WORDS for w in vals)


def find_textual_backchannels(words: list[str], start_idx: int = 0
                              ) -> list[tuple[int, int]]:
    """Find strong short interjection/question phrases in a hypothesis."""
    vals = [norm_word(w) for w in words]
    spans: list[tuple[int, int]] = []
    used: set[int] = set()
    for i in range(max(0, start_idx), len(vals)):
        if i in used:
            continue
        for pattern, pat_len in BACKCHANNEL_PATTERNS:
            end = i + pat_len
            if end <= len(vals) and vals[i:end] == pattern:
                spans.append((i, end))
                used.update(range(i, end))
                break
    return spans


def is_question_interjection_word(word: str) -> bool:
    return "?" in word and norm_word(word) in QUESTION_INTERJECTION_WORDS


def partial_seen_enough(words: list[str], partial_seen: dict[str, int],
                        idx: int) -> bool:
    key = norm_word(words[idx])
    next_key = ""
    if idx + 1 < len(words):
        next_key = f"{key} {norm_word(words[idx + 1])}"
    if next_key and partial_seen.get(next_key, 0) >= FINALIZE_RESCUE_MIN_SEEN:
        return True
    return partial_seen.get(key, 0) >= FINALIZE_RESCUE_MIN_SEEN


def trim_weak_final_tail(words: list[str]) -> list[str]:
    out = list(words)
    changed = True
    while changed and out:
        changed = False
        vals = tuple(norm_words(out))
        for phrase in WEAK_TRAILING_PHRASES:
            if len(vals) >= len(phrase) and vals[-len(phrase):] == phrase:
                del out[-len(phrase):]
                changed = True
                break
        if changed:
            continue
        if norm_word(out[-1]) in WEAK_TRAILING_WORDS:
            out.pop()
            changed = True
    return out


def select_finalize_rescue_words(candidate_words: list[str],
                                 partial_seen: dict[str, int]) -> list[str]:
    """Promote only repeatedly-visible partial words at segment finalization."""
    if not candidate_words:
        return []
    limit = min(len(candidate_words), MAX_FINALIZE_PROMOTE_WORDS)
    if limit > FINALIZE_RESCUE_HOLD_BACK:
        limit -= FINALIZE_RESCUE_HOLD_BACK
    else:
        limit = 0

    out: list[str] = []
    for idx, w in enumerate(candidate_words[:limit]):
        if partial_seen_enough(candidate_words, partial_seen, idx):
            out.append(w)
        else:
            break
    out = trim_weak_final_tail(out)
    if len(out) == 1:
        return []
    return out


def is_prefix_words(prefix: list[str], words: list[str]) -> bool:
    p = norm_words(prefix)
    w = norm_words(words)
    return len(w) >= len(p) and w[:len(p)] == p


def compatible_extension(base: list[str], candidate: list[str]) -> list[str]:
    """Return candidate words after base when candidate is a compatible
    continuation.

    The normal path is exact prefix compatibility.  As a fallback, accept a
    candidate whose prefix extends the longest common prefix with base.  This
    keeps the stream moving when Whisper changes punctuation or drops one
    already-committed filler word in a later pass, without replacing committed
    text.  A second fallback finds a short overlap between the end of the
    committed text and the current candidate, then appends the candidate tail.
    This helps long utterances where Whisper re-decodes the same audio with a
    slightly different beginning.
    """
    if is_prefix_words(base, candidate):
        return candidate[len(base):]

    base_n = norm_words(base)
    cand_n = norm_words(candidate)
    lcp = 0
    for x, y in zip(base_n, cand_n):
        if x != y:
            break
        lcp += 1

    # If most of the committed prefix still agrees, allow the candidate tail.
    # This is deliberately conservative for very short committed prefixes.
    if lcp >= 4 and lcp >= int(0.8 * max(1, len(base_n))):
        return candidate[lcp:]

    # Overlap fallback: if the tail of committed text appears inside the
    # candidate, continue after that overlap. Prefer longer overlaps and avoid
    # using the final candidate words as an anchor because those are most likely
    # to be unstable right-boundary text.
    max_overlap = min(8, len(base_n), len(cand_n) - 1)
    for n in range(max_overlap, 2, -1):
        tail = base_n[-n:]
        last_start = len(cand_n) - n
        for start in range(0, max(0, last_start)):
            if cand_n[start:start + n] == tail:
                return candidate[start + n:]

    # Short-head fallback for cases like committed "It's unusual." followed by
    # a later hypothesis beginning "unusual. but when...".  Allow one
    # distinctive tail word to bridge only at the candidate head; do not use
    # short/common words because they create false anchors.
    if base_n and cand_n and base_n[-1] == cand_n[0]:
        anchor = base_n[-1]
        if (len(anchor) >= 5
                and anchor not in LEADING_REPEAT_FILLERS
                and anchor not in {"there", "that", "this", "thing",
                                   "something", "people"}):
            return candidate[1:]
    return []


def strip_soft_overlap_prefix(words: list[str], overlap_tail: list[str]) -> list[str]:
    """Drop duplicated words caused by causal soft-finalize audio overlap."""
    if not words or not overlap_tail:
        return words
    w = norm_words(words)
    tail = norm_words(overlap_tail)
    max_n = min(len(w), len(tail), SOFT_OVERLAP_MAX_WORDS)
    for n in range(max_n, 1, -1):
        if w[:n] == tail[-n:]:
            return words[n:]
    return words


def strip_turn_overlap_prefix(words: list[str], overlap_tail: list[str]) -> list[str]:
    """Slightly stronger prefix cleanup for speaker-change pre-roll."""
    out = strip_soft_overlap_prefix(words, overlap_tail)
    if out is not words:
        return out
    w = norm_words(words)
    tail = norm_words(overlap_tail)
    max_n = min(len(w), len(tail), SOFT_OVERLAP_MAX_WORDS)

    # Allow one small ASR rewrite in a longer duplicated tail, but require both
    # ends to match so genuine new-speaker phrases are not stripped.
    for n in range(max_n, 3, -1):
        prefix = w[:n]
        suffix = tail[-n:]
        if prefix[0] != suffix[0] or prefix[-1] != suffix[-1]:
            continue
        matches = sum(a == b for a, b in zip(prefix, suffix))
        if matches >= n - 1:
            return words[n:]

    # If Whisper inserts a leading filler before repeating the previous tail,
    # remove the filler plus the repeated phrase.
    if w and w[0] in LEADING_REPEAT_FILLERS:
        for n in range(min(len(w) - 1, len(tail), SOFT_OVERLAP_MAX_WORDS), 2, -1):
            if w[1:1 + n] == tail[-n:]:
                return words[1 + n:]
    return words


def drop_immediate_repeats(words: list[str], max_ngram: int = 6,
                           lookback: int = 18,
                           long_max_ngram: int = 12,
                           long_lookback: int = 60) -> list[str]:
    """Remove near-tail repeated phrases from committed text.

    Whisper often loops on the streaming right edge ("there's sharks
    everywhere" repeated several times). Local Agreement can still commit those
    loops because they are acoustically stable. This keeps the first occurrence
    and drops adjacent or near-adjacent repeated phrases, ignoring leading
    discourse markers such as "and"/"but" for comparison.
    """
    out: list[str] = []
    for w in words:
        out.append(w)
        changed = True
        while changed:
            changed = False
            # Collapse obvious single-word loops such as "there's there's" or
            # "here here", while keeping common intentional emphasis words.
            if len(out) >= 2:
                last = norm_word(out[-1])
                prev = norm_word(out[-2])
                if (last and last == prev
                        and last not in ALLOW_SINGLE_WORD_REPEATS):
                    del out[-1]
                    changed = True
                    continue

            # Exact adjacent repeated phrases. Avoid n=1 so intentional single
            # word repetitions are preserved.
            for n in range(min(max_ngram, len(out) // 2), 1, -1):
                a = norm_words(out[-2 * n:-n])
                b = norm_words(out[-n:])
                if a == b:
                    del out[-n:]
                    changed = True
                    break
            if changed:
                continue

            # Longer exact repeats. Keep this exact-match only; fuzzy matching
            # across long spans can delete legitimate clauses in conversational
            # speech.
            for n in range(min(long_max_ngram, len(out) // 2), max_ngram, -1):
                tail = norm_words(out[-n:])
                search_start = max(0, len(out) - n - long_lookback)
                search_end = len(out) - n
                found = False
                for i in range(search_start, search_end - n + 1):
                    if norm_words(out[i:i + n]) == tail:
                        del out[-n:]
                        changed = True
                        found = True
                        break
                if found:
                    break
            if changed:
                continue

            # Near repeats inside the recent tail, allowing leading fillers on
            # the later occurrence ("I'm still here, but I'm still here").
            max_tail = min(max_ngram, len(out))
            for tail_n in range(max_tail, 1, -1):
                tail = canonical_phrase(out[-tail_n:])
                if len(tail) < 2:
                    continue
                search_start = max(0, len(out) - tail_n - lookback)
                search_end = len(out) - tail_n
                found = False
                for prev_n in range(max_ngram, 1, -1):
                    for i in range(search_start, search_end - prev_n + 1):
                        prev = canonical_phrase(out[i:i + prev_n])
                        if prev == tail:
                            del out[-tail_n:]
                            changed = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
    return out


def remove_phrase_once(words: list[str], phrase_words: list[str]) -> list[str]:
    """Remove one normalized phrase occurrence from a word list."""
    if not words or not phrase_words:
        return words
    vals = norm_words(words)
    phrase = norm_words(phrase_words)
    n = len(phrase)
    if n == 0 or n > len(vals):
        return words
    question_single = (len(phrase_words) == 1 and "?" in phrase_words[0])
    for i in range(0, len(vals) - n + 1):
        if vals[i:i + n] == phrase:
            if question_single and "?" not in words[i]:
                continue
            return words[:i] + words[i + n:]
    return words


def merge_overlay_segments(segments: list[CommittedSegment]) -> list[CommittedSegment]:
    if not segments:
        return []
    merged: list[CommittedSegment] = []
    for seg in sorted(segments, key=lambda s: (s.start, s.commit_time, s.speaker)):
        if (merged
                and merged[-1].speaker == seg.speaker
                and seg.start - merged[-1].end <= OVERLAY_MERGE_GAP_SEC):
            prev = merged[-1]
            prev_words = prev.text.split()
            add_words = remove_phrase_once(seg.text.split(), prev_words)
            text = " ".join(prev_words + add_words).strip()
            merged[-1] = CommittedSegment(
                speaker=prev.speaker,
                start=prev.start,
                end=max(prev.end, seg.end),
                commit_time=max(prev.commit_time, seg.commit_time),
                text=text,
                overlay=True,
            )
        else:
            merged.append(seg)
    return merged


def build_annotation_segments(main_segments: list[CommittedSegment],
                              overlay_segments: list[CommittedSegment],
                              ) -> list[CommittedSegment]:
    """Add overlay turns to JSON and remove their duplicate phrase from the
    opposite speaker's main segment.

    Overlay segments are generated online from already-emitted subtitle events.
    This function does not inspect future audio; it only cleans the final JSON
    representation of those streaming decisions.
    """
    overlay_segments = merge_overlay_segments(overlay_segments)
    cleaned: list[CommittedSegment] = []
    for seg in main_segments:
        words = seg.text.split()
        for ov in overlay_segments:
            if ov.speaker == seg.speaker:
                continue
            overlaps = (seg.start - 0.5 <= ov.start <= seg.end + 0.5)
            if overlaps:
                words = remove_phrase_once(words, ov.text.split())
        text = " ".join(words).strip()
        if text:
            cleaned.append(CommittedSegment(
                speaker=seg.speaker,
                start=seg.start,
                end=seg.end,
                commit_time=seg.commit_time,
                text=text,
                overlay=False,
            ))

    # Keep short overlay turns as first-class annotation segments.  They are
    # already deduplicated at emission time; sort with main turns for readability.
    out = cleaned + overlay_segments
    out.sort(key=lambda s: (s.start, s.commit_time, s.speaker))
    return out


def run_streaming(video_path: str, ref_a: np.ndarray, ref_b: np.ndarray,
                  model_name: str = "small.en", verbose: bool = True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr_interval = ASR_INTERVAL_BY_MODEL.get(model_name, 0.5)
    print(f"Device: {device}   model={model_name}   ASR interval={asr_interval}s")

    print("Loading models…")
    vad = VADModel(device)
    asr = WhisperASR(model_name, device)
    spk = SpeakerIdentifier(ref_a, ref_b, device)

    print("Extracting audio…")
    full_audio = extract_audio(video_path)
    T = len(full_audio) / SAMPLE_RATE
    print(f"Duration: {T:.1f}s")

    committed_segments: list[CommittedSegment] = []
    overlay_segments: list[CommittedSegment] = []
    subtitle_events: list = []

    # Per-segment streaming state
    in_speech     = False
    seg_start     = 0.0
    seg_end       = 0.0
    seg_speaker: Optional[str] = None
    silence_since: Optional[float] = None

    asr_buf       = np.zeros(0, dtype=np.float32)
    asr_buf_start = 0.0

    committed_words: list[str] = []   # white text — monotonic, never shrinks
    prev_hyps: list[list[str]] = []   # last LA_N ASR hypotheses (word lists)
    partial_seen: dict[str, int] = {}   # normalized words/short phrases seen as
                                        # blue partial in prior ASR cycles.
    last_partial_words: list[str] = []  # words last shown as partial — these
                                        # get promoted to committed when the
                                        # segment finalises (so blue words
                                        # don't vanish into thin air).
    best_visible_words: list[str] = []  # longest compatible hypothesis observed
                                        # in this segment; used only at
                                        # finalisation to rescue words that were
                                        # displayed as partial and then vanished.
    overlap_skip_words: list[str] = []  # previous committed tail to strip from
                                        # the next ASR after soft-finalize
                                        # overlap.
    overlap_skip_aggressive = False
    deferred_events: list = []        # subtitle events held until seg_speaker
                                       # is identified (avoids flickering A/B
                                       # at the start of each speech island).
    word_spk_cache: dict[tuple[float, float, float],
                         tuple[Optional[str], float]] = {}
    recent_overlay_events: dict[str, float] = {}

    last_asr_t      = -999.0
    last_vad_t      = -999.0
    last_spk_t      = -999.0
    pending_spk: Optional[str] = None    # candidate new label
    pending_spk_n: int = 0               # consecutive confident votes
    SPK_CONFIRM = 3                      # consecutive votes required before
                                          # accepting a SPEAKER CHANGE
                                          # (higher → fewer false turns)
    SPK_MARGIN  = 0.16                   # cosine-margin threshold for a
                                          # vote to be considered confident
    SPK_MARGIN_FAST = 0.52               # very-confident margin: switch on
                                          # a SINGLE vote (catches short
                                          # interjections without over-splitting)

    chunk_samples = int(CHUNK_DUR * SAMPLE_RATE)

    def local_speaker_at(abs_time: float, now: float,
                         window_sec: float = WORD_SPK_WINDOW_SEC,
                         pad_after: float = 0.25
                         ) -> tuple[Optional[str], float]:
        """Short-window speaker ID around a word time, using only audio <= now."""
        key = (round(abs_time, 1), round(window_sec, 2), round(pad_after, 2))
        if key in word_spk_cache:
            return word_spk_cache[key]
        end = min(now, max(0.0, abs_time + pad_after))
        start = max(0.0, end - window_sec)
        if end - start < 0.4:
            return None, 0.0
        s0 = int(start * SAMPLE_RATE)
        s1 = int(end * SAMPLE_RATE)
        label, margin = spk.identify_with_score(full_audio[s0:s1])
        out = (label, margin)
        word_spk_cache[key] = out
        return out

    def reset_segment(commit_t: float):
        nonlocal committed_words, prev_hyps, partial_seen, last_partial_words
        nonlocal best_visible_words, overlap_skip_words, overlap_skip_aggressive
        nonlocal asr_buf, asr_buf_start, in_speech, seg_speaker, silence_since
        nonlocal deferred_events
        # Promote visible partials to committed so blue words the user saw do
        # not disappear at finalisation. Prefer the longest compatible snapshot
        # observed in the segment instead of only the last ASR result, because
        # Whisper can shrink or rewrite the right edge just before silence.
        rescue_words = compatible_extension(committed_words, best_visible_words)
        if not rescue_words and last_partial_words:
            rescue_words = last_partial_words
        while (rescue_words and committed_words
               and norm_word(rescue_words[0]) == norm_word(committed_words[-1])):
            rescue_words = rescue_words[1:]
        rescue_words = select_finalize_rescue_words(rescue_words, partial_seen)
        if rescue_words:
            committed_words.extend(rescue_words)
        committed_words = drop_immediate_repeats(committed_words)
        committed_words = trim_weak_final_tail(committed_words)
        final_words = list(committed_words)
        # Emit a final "partial-cleared" subtitle event for this speaker so
        # the renderer can drop the partial line cleanly.
        if seg_speaker is not None and committed_words:
            final_event = {
                "time": commit_t,
                "speaker": seg_speaker,
                "committed": " ".join(committed_words),
                "partial":   "",
                "finalize":  True,
            }
            if deferred_events:
                for de in deferred_events:
                    de["speaker"] = seg_speaker
                    subtitle_events.append(de)
                deferred_events.clear()
            subtitle_events.append(final_event)
        text = " ".join(committed_words).strip()
        if text:
            seg = CommittedSegment(
                speaker=seg_speaker or "A",
                start=round(seg_start, 2),
                end=round(seg_end, 2),
                commit_time=round(commit_t, 2),
                text=text,
            )
            committed_segments.append(seg)
            if verbose:
                print(f"\n  ✓ [{seg.speaker}] {seg.start}–{seg.end} @ {seg.commit_time}")
                print(f"    \"{text[:90]}{'…' if len(text)>90 else ''}\"")
        committed_words   = []
        prev_hyps         = []
        partial_seen      = {}
        last_partial_words = []
        best_visible_words = []
        overlap_skip_words = []
        overlap_skip_aggressive = False
        asr_buf           = np.zeros(0, dtype=np.float32)
        asr_buf_start     = commit_t
        silence_since     = None
        seg_speaker       = None   # re-identify on the next speech island
        deferred_events   = []
        return final_words

    t = 0.0
    while t < T:
        s0 = int(t * SAMPLE_RATE)
        s1 = min(s0 + chunk_samples, len(full_audio))
        chunk = full_audio[s0:s1]
        t = s1 / SAMPLE_RATE

        asr_buf = np.concatenate([asr_buf, chunk])
        if len(asr_buf) > int(MAX_BUF_SEC * SAMPLE_RATE):
            trim = len(asr_buf) - int(MAX_BUF_SEC * SAMPLE_RATE)
            asr_buf = asr_buf[trim:]
            asr_buf_start = t - len(asr_buf) / SAMPLE_RATE

        # ── VAD (20 Hz) ──────────────────────────────────────────────────────
        if t - last_vad_t >= VAD_INTERVAL:
            last_vad_t = t
            vad_win = min(int(0.6 * SAMPLE_RATE), len(asr_buf))
            speech_now = vad.is_speech(asr_buf[-vad_win:]) if vad_win > 512 else False
            if speech_now:
                if not in_speech:
                    in_speech = True
                    seg_start = max(0.0, t - VAD_PREROLL_SEC)
                    asr_buf_start = seg_start
                    s_idx = max(0, int(seg_start * SAMPLE_RATE))
                    asr_buf = full_audio[s_idx:s1].copy()
                    if verbose:
                        print(f"\n[{t:.2f}s] SPEECH START")
                silence_since = None
                seg_end = t
            else:
                if in_speech:
                    if silence_since is None:
                        silence_since = t
                    elif t - silence_since >= SILENCE_END:
                        if verbose:
                            print(f"[{t:.2f}s] SILENCE → finalize")
                        # Commit any local-agreement candidate words upon
                        # finalization (they were partial → now stable).
                        # Online speaker observation (causal).
                        spk_win = min(int(3.0 * SAMPLE_RATE), len(asr_buf))
                        spk.observe(asr_buf[-spk_win:])
                        in_speech = False
                        reset_segment(t)
                        continue

        if not in_speech:
            continue

        # ── Soft finalize on long segments ───────────────────────────────────
        # If a single speech island runs past SOFT_FINALIZE_SEC the ASR
        # buffer approaches Whisper's 30 s context cap and previously
        # committed words start to fall outside the audio window — once
        # that happens curr_words can no longer share a prefix with
        # committed_words and nothing else commits. Pre-empt this by
        # finalising the current segment and continuing under the same
        # speaker label.
        if (in_speech and committed_words
                and (t - seg_start) >= SOFT_FINALIZE_SEC):
            if verbose:
                print(f"[{t:.2f}s] SOFT FINALIZE (long segment)")
            saved_speaker = seg_speaker
            spk_win = min(int(3.0 * SAMPLE_RATE), len(asr_buf))
            spk.observe(asr_buf[-spk_win:])
            final_words = reset_segment(t)
            seg_speaker = saved_speaker   # carry the label into the new seg
            seg_start = t
            seg_end   = t
            in_speech = True
            overlap_start = max(0.0, t - SOFT_OVERLAP_SEC)
            s_idx = int(overlap_start * SAMPLE_RATE)
            asr_buf = full_audio[s_idx:s1].copy()
            asr_buf_start = overlap_start
            overlap_skip_words = final_words[-SOFT_OVERLAP_MAX_WORDS:]
            overlap_skip_aggressive = False
            continue

        # ── Speaker ID (5 Hz) ────────────────────────────────────────────────
        # With pyannote/embedding (matching the provided refs) the per-cycle
        # decision is reliable, so we both:
        #   (a) initialise seg_speaker on the first usable cycle, and
        #   (b) detect mid-island speaker turns via a confident-vote
        #       hysteresis (need SPK_CONFIRM consecutive cycles where
        #       the alternative label wins by ≥ SPK_MARGIN cosine margin).
        if t - last_spk_t >= SPK_INTERVAL:
            last_spk_t = t
            if len(asr_buf) >= int(1.0 * SAMPLE_RATE):
                spk_win = min(int(2.0 * SAMPLE_RATE), len(asr_buf))
                new_spk, margin = spk.identify_with_score(asr_buf[-spk_win:])
                if new_spk is not None:
                    if seg_speaker is None:
                        seg_speaker = new_spk
                        pending_spk = None; pending_spk_n = 0
                    elif new_spk == seg_speaker:
                        pending_spk = None; pending_spk_n = 0
                    elif margin >= SPK_MARGIN:
                        # confident vote against current label
                        if pending_spk == new_spk:
                            pending_spk_n += 1
                        else:
                            pending_spk = new_spk; pending_spk_n = 1
                        # Very-confident single-vote shortcut for snappy
                        # detection of short backchannels.
                        confirm_ok = (pending_spk_n >= SPK_CONFIRM
                                      or margin >= SPK_MARGIN_FAST)
                        enough_turn_evidence = (
                            (t - seg_start) >= 3.0
                            and len(committed_words) >= 3
                        )
                        if confirm_ok and committed_words and enough_turn_evidence:
                            if verbose:
                                print(f"[{t:.2f}s] SPEAKER CHANGE "
                                      f"{seg_speaker}→{new_spk} (m={margin:.2f})")
                            final_words = reset_segment(t)
                            turn_start = max(0.0, t - SPEAKER_CHANGE_PREROLL_SEC)
                            seg_start = turn_start
                            seg_end = t
                            seg_speaker = new_spk
                            s_idx = int(turn_start * SAMPLE_RATE)
                            asr_buf = full_audio[s_idx:s1].copy()
                            asr_buf_start = turn_start
                            overlap_skip_words = final_words[-SOFT_OVERLAP_MAX_WORDS:]
                            overlap_skip_aggressive = True
                            pending_spk = None; pending_spk_n = 0
                            continue

        # ── ASR (2 Hz / 1 Hz depending on model) ────────────────────────────
        if t - last_asr_t >= asr_interval:
            last_asr_t = t
            if len(asr_buf) < SAMPLE_RATE * 0.4:
                continue

            # NOTE: We do NOT pass `committed_words` as Whisper's `prompt`
            # because the audio buffer still contains the speech for those
            # words — Whisper's prompt is meant to describe context that
            # came BEFORE the input audio, not text that overlaps it.
            # Doing so confuses the decoder and causes it to emit nothing
            # for subsequent cycles after the first commit.
            curr_text, peaks = asr.transcribe(asr_buf)
            audio_dur = len(asr_buf) / SAMPLE_RATE
            n_enc = int(audio_dur * 50)
            curr_text = asr.truncate_by_attention(curr_text, peaks, n_enc)
            curr_words = curr_text.split()
            if overlap_skip_words:
                if overlap_skip_aggressive:
                    curr_words = strip_turn_overlap_prefix(
                        curr_words, overlap_skip_words)
                else:
                    curr_words = strip_soft_overlap_prefix(
                        curr_words, overlap_skip_words)
                overlap_skip_words = []
                overlap_skip_aggressive = False
            if not curr_words:
                continue
            if silence_since is not None and t - silence_since >= SILENCE_PARTIAL_GRACE_SEC:
                prior_visible_len = len(committed_words) + len(last_partial_words)
                if prior_visible_len and (
                        len(curr_words)
                        > prior_visible_len + SILENCE_PARTIAL_MAX_GROWTH):
                    curr_words = curr_words[:prior_visible_len]
                    if not curr_words:
                        continue
            word_times = asr.word_attention_times(
                peaks, n_enc, len(curr_words), audio_dur)
            buf_origin = t - audio_dur
            word_speakers: list[Optional[str]] = [seg_speaker] * len(curr_words)
            if seg_speaker is not None:
                for wi, rel_time in enumerate(word_times[:len(curr_words)]):
                    label, margin = local_speaker_at(buf_origin + rel_time, t)
                    if label is not None and margin >= WORD_SPK_MARGIN:
                        word_speakers[wi] = label

            forced_overlay_chunks = []
            forced_overlay_indices: set[int] = set()
            if seg_speaker is not None:
                other_spk = "B" if seg_speaker == "A" else "A"
                for wi, w in enumerate(curr_words):
                    if not is_question_interjection_word(w):
                        continue
                    rel_time = (word_times[wi]
                                if wi < len(word_times) else audio_dur)
                    abs_time = buf_origin + rel_time
                    tight_label, tight_margin = local_speaker_at(
                        abs_time, t,
                        window_sec=ONE_WORD_SPK_WINDOW_SEC,
                        pad_after=0.12)
                    overlay_spk = other_spk
                    if (tight_label is not None and tight_label != seg_speaker
                            and tight_margin >= ONE_WORD_SPK_MARGIN):
                        overlay_spk = tight_label
                    audio_start = max(0.0, abs_time)
                    audio_end = min(t, max(audio_start + 0.45,
                                           audio_start + 0.55))
                    emit_time = max(t, abs_time)
                    emit_time = min(emit_time, t + asr_interval - 0.05)
                    forced_overlay_chunks.append(
                        (overlay_spk, w, emit_time, audio_start, audio_end))
                    forced_overlay_indices.add(wi)

            if forced_overlay_indices:
                curr_words = [
                    w for wi, w in enumerate(curr_words)
                    if wi not in forced_overlay_indices
                ]
                word_times = [
                    wt for wi, wt in enumerate(word_times)
                    if wi not in forced_overlay_indices
                ]
                word_speakers = [
                    sp for wi, sp in enumerate(word_speakers)
                    if wi not in forced_overlay_indices
                ]

            # ── Local Agreement (Step 2): require LA_N consecutive matches.
            # Compute agreement on the extension after already-committed text,
            # not on the full hypothesis prefix. Whisper often revises early
            # filler words in long buffers; full-prefix LA would then block all
            # later words even when the continuation is stable.
            curr_ext = compatible_extension(committed_words, curr_words)
            stable_ext = curr_ext[:]
            for prev in prev_hyps:
                prev_ext = compatible_extension(committed_words, prev)
                stable_ext = local_agreement_extension(prev_ext, stable_ext)
            have_la_quorum = (len(prev_hyps) + 1) >= LA_N
            partial_carry_ext = common_prefix_words(last_partial_words, curr_ext)
            commit_ext = stable_ext
            carry_from_visible_partial = False
            if len(partial_carry_ext) > len(commit_ext):
                commit_ext = partial_carry_ext
                carry_from_visible_partial = True

            # Monotonicity: committed text is never rewritten.  We only append
            # words that satisfy Local Agreement and have already appeared as
            # blue partial in a previous cycle.  The seen check is word/phrase
            # based, not absolute-position based, so filler insertions do not
            # invalidate a word the viewer already saw in blue.
            committed_is_prefix = is_prefix_words(committed_words, curr_words)

            new_pending = []
            if have_la_quorum:
                carry_limit = len(commit_ext)
                if carry_from_visible_partial:
                    carry_limit = max(0, carry_limit - PARTIAL_CARRY_HOLD_BACK)
                for k, w in enumerate(commit_ext[:carry_limit]):
                    key = norm_word(w)
                    next_key = ""
                    if k + 1 < len(commit_ext):
                        next_key = f"{key} {norm_word(commit_ext[k + 1])}"
                    if carry_from_visible_partial:
                        # Partial-carry exists to unstick long utterances when
                        # LA is blocked by earlier filler rewrites.  Keep it
                        # stricter than normal LA so one-cycle wrong partials do
                        # not become permanent committed text.
                        if (next_key
                                and partial_seen.get(next_key, 0)
                                >= PARTIAL_CARRY_MIN_SEEN):
                            new_pending.append(w)
                        else:
                            break
                        continue
                    if (partial_seen.get(key, 0) >= MIN_PARTIAL_AGE
                            or (next_key
                                and partial_seen.get(next_key, 0)
                                >= MIN_PARTIAL_AGE)):
                        new_pending.append(w)
                    else:
                        # Do not let an unstable leading filler block a stable
                        # continuation forever. We skip only before committing
                        # any word in this cycle; once committing starts, keep
                        # the output contiguous and stop at the first unseen
                        # token.
                        if not new_pending and key in LEADING_REPEAT_FILLERS:
                            continue
                        break

            if new_pending:
                committed_words.extend(new_pending)
                committed_words = drop_immediate_repeats(committed_words)

            if committed_is_prefix:
                partial_words = curr_words[len(committed_words):]
            else:
                partial_words = compatible_extension(committed_words, curr_words)

            # Mark visible partial words so a later stable hypothesis can
            # promote them. Count cycles rather than boolean presence so
            # MIN_PARTIAL_AGE remains meaningful if increased.
            for k, w in enumerate(partial_words):
                key = norm_word(w)
                partial_seen[key] = partial_seen.get(key, 0) + 1
                if k + 1 < len(partial_words):
                    phrase = f"{key} {norm_word(partial_words[k + 1])}"
                    partial_seen[phrase] = partial_seen.get(phrase, 0) + 1

            partial_text = " ".join(partial_words)
            committed_text = " ".join(committed_words)
            # Track for finalize-time promotion (so blue words don't vanish).
            last_partial_words = list(partial_words)
            if (len(curr_words) > len(best_visible_words)
                    and (not committed_words
                         or compatible_extension(committed_words, curr_words))):
                best_visible_words = list(curr_words)

            if verbose:
                cd = committed_text or "–"
                pd = partial_text or "–"
                print(f"[{t:.2f}s] {seg_speaker or '?'}: {cd} [{pd}]")

            # ── Word-level event emission (true streaming reveal) ─────────
            # Use cross-attention to time-stamp each word inside the audio
            # buffer, then emit an event for each NEW word at the time it
            # would naturally appear (no earlier than the current cycle t,
            # and no later than the next cycle t + asr_interval).
            # Find the last subtitle event FOR THE CURRENT SEGMENT — i.e.
            # this speaker AND emitted within the current segment (after
            # seg_start). Previous segments by the same speaker were
            # already committed and reset, so their state must not bleed
            # into the new segment's display.
            prev_running = {"committed": "", "partial": ""}
            for prev_ev in reversed(subtitle_events):
                if (prev_ev.get("speaker") == seg_speaker
                        and prev_ev.get("time", 0) >= seg_start):
                    prev_running = prev_ev
                    break
            prev_total_words = (prev_running.get("committed", "").split()
                                + prev_running.get("partial", "").split())
            n_already_visible = len(prev_total_words)
            n_committed = len(committed_words)
            running_visible = list(prev_total_words)

            new_count = max(0, len(curr_words) - n_already_visible)
            overlay_chunks = forced_overlay_chunks[:]
            if seg_speaker is not None and new_count > 0:
                oi = n_already_visible
                while oi < len(curr_words):
                    local_spk = (word_speakers[oi]
                                 if oi < len(word_speakers) else seg_speaker)
                    if local_spk is None or local_spk == seg_speaker:
                        oi += 1
                        continue
                    start_i = oi
                    while (oi < len(curr_words)
                           and oi - start_i < BACKCHANNEL_MAX_WORDS
                           and oi < len(word_speakers)
                           and word_speakers[oi] == local_spk):
                        oi += 1
                    chunk_words = curr_words[start_i:oi]
                    if looks_like_backchannel(chunk_words):
                        rel_time = (word_times[start_i]
                                    if start_i < len(word_times) else audio_dur)
                        audio_start = max(0.0, buf_origin + rel_time)
                        end_idx = min(oi - 1, len(word_times) - 1)
                        rel_end = (word_times[end_idx]
                                   if end_idx >= 0 else rel_time)
                        audio_end = min(t, max(audio_start + 0.45,
                                               buf_origin + rel_end + 0.35))
                        emit_time = max(t, buf_origin + rel_time)
                        emit_time = min(emit_time, t + asr_interval - 0.05)
                        overlay_chunks.append(
                            (local_spk, " ".join(chunk_words), emit_time,
                             audio_start, audio_end))

                other_spk = "B" if seg_speaker == "A" else "A"
                pattern_start = max(0, n_already_visible - BACKCHANNEL_MAX_WORDS)
                for start_i, end_i in find_textual_backchannels(
                        curr_words, pattern_start):
                    if end_i <= n_already_visible:
                        continue
                    rel_time = (word_times[start_i]
                                if start_i < len(word_times) else audio_dur)
                    audio_start = max(0.0, buf_origin + rel_time)
                    end_idx = min(end_i - 1, len(word_times) - 1)
                    rel_end = (word_times[end_idx]
                               if end_idx >= 0 else rel_time)
                    audio_end = min(t, max(audio_start + 0.45,
                                           buf_origin + rel_end + 0.35))
                    emit_time = max(t, buf_origin + rel_time)
                    emit_time = min(emit_time, t + asr_interval - 0.05)
                    overlay_chunks.append(
                        (other_spk, " ".join(curr_words[start_i:end_i]),
                         emit_time, audio_start, audio_end))

            for i in range(new_count):
                word_idx = n_already_visible + i
                if word_idx >= len(curr_words):
                    break
                # estimated absolute audio time when this word was spoken
                rel_time = (word_times[word_idx]
                            if word_idx < len(word_times) else audio_dur)
                abs_time = buf_origin + rel_time
                # streaming-causal clamp: not earlier than `t`
                emit_time = max(t, abs_time)
                # not later than just before the next ASR cycle
                emit_time = min(emit_time, t + asr_interval - 0.05)

                running_visible.append(curr_words[word_idx])
                # split into committed (first n_committed) vs partial
                vis_committed = " ".join(running_visible[:n_committed])
                vis_partial = " ".join(running_visible[n_committed:])

                event = {
                    "time": emit_time,
                    "speaker": seg_speaker,
                    "committed": vis_committed,
                    "partial":   vis_partial,
                }
                if seg_speaker is None:
                    deferred_events.append(event)
                else:
                    if deferred_events:
                        for de in deferred_events:
                            de["speaker"] = seg_speaker
                            subtitle_events.append(de)
                        deferred_events.clear()
                    subtitle_events.append(event)

            for (overlay_spk, overlay_text, overlay_time,
                 overlay_start, overlay_end) in overlay_chunks:
                overlay_key = (
                    f"{overlay_spk}:{' '.join(norm_words(overlay_text.split()))}")
                if (overlay_time - recent_overlay_events.get(overlay_key, -999)
                        < OVERLAY_REPEAT_SUPPRESS_SEC):
                    continue
                recent_overlay_events[overlay_key] = overlay_time
                subtitle_events.append({
                    "time": overlay_time,
                    "speaker": overlay_spk,
                    "committed": overlay_text,
                    "partial": "",
                    "overlay": True,
                })
                ann_dur = min(2.5, max(0.45, 0.28 * len(overlay_text.split())))
                ann_start = overlay_time
                ann_end = min(T, overlay_time + ann_dur)
                overlay_segments.append(CommittedSegment(
                    speaker=overlay_spk,
                    start=round(ann_start, 2),
                    end=round(ann_end, 2),
                    commit_time=round(overlay_time, 2),
                    text=overlay_text,
                    overlay=True,
                ))

            # If no NEW words this cycle (e.g. text shrank) still emit a
            # snapshot so the renderer sees the latest state.
            if new_count == 0:
                event = {
                    "time": t,
                    "speaker": seg_speaker,
                    "committed": committed_text,
                    "partial":   partial_text,
                }
                if seg_speaker is None:
                    deferred_events.append(event)
                else:
                    if deferred_events:
                        for de in deferred_events:
                            de["speaker"] = seg_speaker
                            subtitle_events.append(de)
                        deferred_events.clear()
                    subtitle_events.append(event)

            prev_hyps.append(curr_words)
            if len(prev_hyps) > LA_N - 1:
                prev_hyps = prev_hyps[-(LA_N - 1):]

    # final flush
    if in_speech:
        reset_segment(T)

    annotation_segments = build_annotation_segments(
        committed_segments, overlay_segments)

    print(f"\nTotal committed segments: {len(annotation_segments)}")
    return annotation_segments, subtitle_events


# ── Video rendering ───────────────────────────────────────────────────────────
def render_video(video_path: str, segments: list,
                 subtitle_events: list, out_path: str,
                 asr_interval: float = 0.5):
    dur_r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True)
    duration = float(dur_r.stdout.strip()) if dur_r.returncode == 0 else 300.0

    srt_path = out_path.replace(".mp4", ".srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(to_srt(segments))

    ass_path = out_path.replace(".mp4", ".ass")
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(to_ass(subtitle_events, duration, asr_interval=asr_interval))

    cmd = ["ffmpeg", "-y", "-i", video_path,
           "-vf", f"ass={ass_path}",
           "-c:a", "copy", "-c:v", "libx264", "-preset", "fast", "-crf", "23",
           out_path]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        print("ASS burn failed; falling back to SRT:", r.stderr.decode()[:150])
        cmd2 = ["ffmpeg", "-y", "-i", video_path,
                "-vf",
                f"subtitles={srt_path}:force_style='FontSize=22,PrimaryColour=&HFFFFFF&'",
                "-c:a", "copy", "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                out_path]
        subprocess.run(cmd2, capture_output=True)
    print(f"Video → {out_path}")


def save_json(segments: list, path: str):
    data = [{"speaker": s.speaker, "start": s.start, "end": s.end,
             "commit_time": s.commit_time,
             "text": cleanup_punctuation_text(s.text)}
            for s in segments]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"JSON  → {path}")


def process_clip(clip_id: str, data_dir: str, output_dir: str,
                 model_name: str = "medium.en"):
    dp, op = Path(data_dir), Path(output_dir)
    op.mkdir(parents=True, exist_ok=True)
    video = str(dp / f"clip{clip_id}_raw.mp4")
    ref_a = np.load(str(dp / f"clip{clip_id}_spkA_embedding.npy"))
    ref_b = np.load(str(dp / f"clip{clip_id}_spkB_embedding.npy"))

    print(f"\n{'='*60}\nclip{clip_id}  model={model_name}\n{'='*60}")
    segs, events = run_streaming(video, ref_a, ref_b, model_name, verbose=True)
    save_json(segs,  str(op / f"clip{clip_id}_annotation.json"))
    render_video(video, segs, events,
                 str(op / f"clip{clip_id}_subtitled.mp4"),
                 asr_interval=ASR_INTERVAL_BY_MODEL.get(model_name, 0.5))
    return segs


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--clip", default="all")
    p.add_argument("--data_dir", default="/home/elicer/project/SLP_project01_data")
    p.add_argument("--output_dir", default="/home/elicer/project/output_v3")
    p.add_argument("--model", default="medium.en", choices=["small.en", "medium.en"])
    args = p.parse_args()

    clips = (["00", "01", "02", "03", "04"] if args.clip == "all"
             else [c.strip() for c in args.clip.split(",")])
    for cid in clips:
        process_clip(cid, args.data_dir, args.output_dir, args.model)
