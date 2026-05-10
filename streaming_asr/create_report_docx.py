import html
import zipfile
from pathlib import Path


OUT = Path("/home/elicer/project/output_submit/project1_report.docx")


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def para(text: str = "", style: str | None = None) -> str:
    ppr = ""
    if style:
        ppr = f"<w:pPr><w:pStyle w:val=\"{style}\"/></w:pPr>"
    return (
        "<w:p>"
        f"{ppr}"
        "<w:r><w:t xml:space=\"preserve\">"
        f"{esc(text)}"
        "</w:t></w:r>"
        "</w:p>"
    )


def bullet(text: str) -> str:
    return para(f"- {text}")


def heading(text: str, level: int = 1) -> str:
    return para(text, f"Heading{level}")


def document_xml() -> str:
    parts: list[str] = []
    parts.append(para("Project 1 Report: Streaming ASR Subtitle System", "Title"))
    parts.append(para("Spoken Language Processing"))
    parts.append(para("Output clips: 00-04"))

    parts.append(heading("1. Task Overview"))
    parts.append(para(
        "The goal of this project is to build a streaming ASR subtitle system "
        "for two-speaker conversation videos. The system must display live "
        "speaker-attributed subtitles during playback and save the final "
        "committed subtitles as JSON annotation files. This is not an offline "
        "transcription alignment task: every ASR, VAD, speaker, and subtitle "
        "decision is made causally using only audio available up to the current "
        "playback time."
    ))
    parts.append(para(
        "The implementation processes clips 00-04 and produces one subtitled "
        "MP4 and one committed annotation JSON for each clip. The JSON format "
        "matches the provided demo annotations: speaker, start, end, "
        "commit_time, and text."
    ))

    parts.append(heading("2. System Design"))
    parts.append(para(
        "The system combines chunk-wise streaming, Silero VAD, pyannote speaker "
        "embeddings, Whisper ASR, Local Agreement, and attention-based "
        "right-boundary truncation."
    ))
    parts.append(bullet("Audio is read in 0.5 s chunks to simulate playback."))
    parts.append(bullet("Silero VAD is evaluated every 0.05 s, satisfying the 20 Hz limit."))
    parts.append(bullet("Whisper medium.en is evaluated every 1.0 s, satisfying the 1 Hz limit."))
    parts.append(bullet("Pyannote speaker embedding inference is evaluated every 0.2 s, satisfying the 5 Hz limit."))
    parts.append(bullet("Only past audio is used. The implementation never runs full-video offline transcription."))

    parts.append(heading("3. Streaming ASR and Stabilization"))
    parts.append(para(
        "The ASR module uses Whisper medium.en with a causal speech buffer. "
        "The buffer is capped at 28 s to stay within Whisper's context limit. "
        "For speech onset, a 1.2 s VAD pre-roll is included so short phrase "
        "beginnings are not clipped. This pre-roll only reuses already observed "
        "past audio and therefore does not violate the streaming constraint."
    ))
    parts.append(para(
        "Committed text is stabilized using Local Agreement with N=2. A word "
        "or phrase becomes committed only when it remains stable across "
        "consecutive ASR hypotheses and has first appeared as a partial "
        "subtitle. The committed text is monotonic: once a word is shown as "
        "committed, it is not revised or removed."
    ))
    parts.append(para(
        "To reduce unstable right-boundary tokens, the implementation records "
        "Whisper decoder cross-attention and removes tokens whose attention "
        "peaks are too close to the current audio boundary. This follows the "
        "idea of attention-guided truncation used in Simul-Whisper."
    ))
    parts.append(para(
        "Long speech islands are soft-finalized after 20 s. The next buffer "
        "keeps a short 1.0 s causal overlap, and duplicated overlap text is "
        "removed from the following hypothesis. This avoids long-buffer drift "
        "without using future audio."
    ))

    parts.append(heading("4. Speaker Attribution and Overlap Handling"))
    parts.append(para(
        "The speaker module uses the provided Speaker A and Speaker B reference "
        "embeddings. For each active speech region, pyannote embeddings are "
        "compared with the references using cosine similarity. Speaker changes "
        "are accepted only after confident evidence, with a fast path for "
        "very high-margin turns."
    ))
    parts.append(para(
        "To reduce clipped first words at turn changes, the final version uses "
        "a 0.4 s speaker-change pre-roll. When a speaker switch is confirmed, "
        "the new speaker buffer starts slightly before the detection time. "
        "This again uses only already observed audio. A stronger overlap-strip "
        "step removes duplicated previous-speaker tail text."
    ))
    parts.append(para(
        "Short interjections and backchannels are handled as overlay segments. "
        "Examples include 'That's crazy. Why? I know.', 'Like you were "
        "surfing.', 'No I didn't do anything where it', and one-word question "
        "interjections such as 'Sharks?'. These are displayed on the other "
        "speaker's line and also saved as committed annotation segments."
    ))

    parts.append(heading("5. Subtitle Visualization"))
    parts.append(para(
        "The renderer writes ASS subtitles and burns them into the output MP4. "
        "Committed words are shown in white and partial words in blue. Speaker "
        "A and Speaker B are tracked independently so short overlaps can be "
        "displayed on two lines. Per-speaker partial text is capped to two "
        "lines and long subtitles are trimmed for display only; the JSON keeps "
        "the committed annotation text."
    ))
    parts.append(para(
        "Several display cleanups were added after visual inspection: inactive "
        "speaker lines clear after 5 s, final silence no longer causes a long "
        "subtitle to pop up at the end, and minor punctuation cleanup fixes "
        "cases such as 'Here's sharks. everywhere' to 'Here's sharks everywhere'."
    ))

    parts.append(heading("6. Output and Evaluation"))
    parts.append(para(
        "The final output directory is /home/elicer/project/output_submit. "
        "For each clip XX in 00-04, the system produces clipXX_subtitled.mp4 "
        "and clipXX_annotation.json. A combined project1_annotation.json is "
        "also generated for convenience."
    ))
    parts.append(para(
        "The implementation satisfies the project requirements for Step 1 "
        "(chunk-wise streaming), Step 2 (Local Agreement), and Step 3 "
        "(attention-based right-boundary truncation). It also respects the "
        "specified runtime frequencies and the no-future-audio constraint."
    ))

    parts.append(heading("7. Limitations"))
    parts.append(para(
        "The main remaining limitation is ASR recognition quality in noisy, "
        "overlapped, or highly conversational regions. Since the system is "
        "strictly streaming, it cannot use future audio or an offline full-video "
        "transcript to correct earlier words. Some Whisper errors therefore "
        "remain, but the stabilization and display logic reduce flicker, "
        "delayed commits, repeated hallucinations, and speaker-attribution "
        "errors in short interjections."
    ))

    body = "\n".join(parts)
    return f"""<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>
<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\">
  <w:body>
    {body}
    <w:sectPr>
      <w:pgSz w:w=\"12240\" w:h=\"15840\"/>
      <w:pgMar w:top=\"1440\" w:right=\"1440\" w:bottom=\"1440\" w:left=\"1440\"/>
    </w:sectPr>
  </w:body>
</w:document>
"""


CONTENT_TYPES = """<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>
<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">
  <Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>
  <Default Extension=\"xml\" ContentType=\"application/xml\"/>
  <Override PartName=\"/word/document.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/>
  <Override PartName=\"/word/styles.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml\"/>
</Types>
"""

RELS = """<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>
<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">
  <Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"word/document.xml\"/>
</Relationships>
"""

WORD_RELS = """<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>
<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"/>
"""

STYLES = """<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>
<w:styles xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\">
  <w:style w:type=\"paragraph\" w:default=\"1\" w:styleId=\"Normal\">
    <w:name w:val=\"Normal\"/>
    <w:rPr><w:rFonts w:ascii=\"Calibri\" w:hAnsi=\"Calibri\"/><w:sz w:val=\"22\"/></w:rPr>
  </w:style>
  <w:style w:type=\"paragraph\" w:styleId=\"Title\">
    <w:name w:val=\"Title\"/>
    <w:pPr><w:spacing w:after=\"240\"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val=\"32\"/></w:rPr>
  </w:style>
  <w:style w:type=\"paragraph\" w:styleId=\"Heading1\">
    <w:name w:val=\"heading 1\"/>
    <w:pPr><w:spacing w:before=\"240\" w:after=\"120\"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val=\"26\"/></w:rPr>
  </w:style>
  <w:style w:type=\"paragraph\" w:styleId=\"Heading2\">
    <w:name w:val=\"heading 2\"/>
    <w:pPr><w:spacing w:before=\"180\" w:after=\"100\"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val=\"24\"/></w:rPr>
  </w:style>
</w:styles>
"""


def main() -> None:
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", RELS)
        z.writestr("word/_rels/document.xml.rels", WORD_RELS)
        z.writestr("word/styles.xml", STYLES)
        z.writestr("word/document.xml", document_xml())
    print(OUT)


if __name__ == "__main__":
    main()
