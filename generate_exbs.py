# This script compiles exb for human checking and editing.
# Data sources are audio and ASR on segments, made from diarization
#
from pathlib import Path

import pandas as pd
from lxml import etree as ET

from utils import read_json, read_rttm


min_duration_seconds = 0.1
files = [
    file.with_suffix("").name
    for file in Path("data/audio_16khz_mono_wav/").glob("*.wav")
]
for file in files:
    audio_path = Path(f"data/audio_16khz_mono_wav/{file}.wav")
    diarization_path = Path(f"data/asr/{file}.csv")
    template_path = Path("exb_template.xml")
    out_path = Path(f"data/exbs/{file}.exb")

    exb = ET.fromstring(template_path.read_bytes())
    diarization_df = pd.read_csv(diarization_path)

    def add_df_to_template(exb: ET.Element, df: pd.DataFrame) -> ET.Element:
        """Adds the transcription data from df to the template.

        Right now, only inclusion of one transcription is enabled.

        Args:
            exb (ET.Element): parsed EXB template
            df (pd.DataFrame): dataframe with columns start, end, speaker_name, duration, whisper

        Returns:
            ET.Element: template with transcription tiers.
        """
        df2 = df[["start", "end", "speaker_name", "duration", "whisper"]].copy()
        df2["speaker_name"] = df.speaker_name.apply(lambda s: s + "_whisper")
        df2 = df2.rename(columns={"whisper": "text"})
        df = df2
        df["text"] = df.text.fillna("")
        # Add speakers:
        for speaker_name in sorted(df.speaker_name.unique()):
            speaker = ET.Element("speaker", attrib={"id": speaker_name})
            abbreviation = ET.Element("abbreviation")
            abbreviation.text = speaker_name
            speaker.append(abbreviation)
            exb.find(".//speakertable").append(speaker)

        # Add <tli>:
        timeline = exb.find(".//common-timeline")
        N = len(timeline.findall(".//tli"))
        for t in sorted(list(set(df.start.values).union(set(df.end.values)))):
            tli = ET.Element("tli", attrib={"id": f"T{N}", "time": str(t)})
            timeline.append(tli)
            N += 1
        # Sort timeline
        timeline[:] = sorted(timeline, key=lambda child: float(child.get("time")))
        # Prepare inverse mapper (seconds -> id):
        mapper = {tli.get("time"): tli.get("id") for tli in timeline.findall("tli")}
        # Add new tier(s):
        for speaker_name in sorted(df.speaker_name.unique()):
            tier = ET.Element(
                "tier",
                attrib=dict(
                    id=speaker_name,
                    category="v",
                    type="t",
                    display_name=speaker_name,
                    speaker=speaker_name,
                ),
            )
            for i, row in df[df.speaker_name == speaker_name].iterrows():
                event = ET.Element(
                    "event",
                    attrib=dict(
                        start=mapper.get(str(row["start"])),
                        end=mapper.get(str(row["end"])),
                    ),
                )
                event.text = (
                    row["text"]
                    if float(row["duration"]) >= min_duration_seconds
                    else "-"
                )
                tier.append(event)
            exb.find("basic-body").append(tier)
        return exb

    exb = add_df_to_template(exb, diarization_df)
    exb.find(".//referenced-file").set("url", audio_path.name)
    ET.indent(exb, space="\t")
    exb.getroottree().write(
        Path(out_path),
        pretty_print=True,
        encoding="utf8",
        xml_declaration='<?xml version="1.0" encoding="UTF-8"?>',
    )

    Path(out_path).write_text(
        Path(out_path)
        .read_text()
        .replace(
            "<?xml version='1.0' encoding='UTF8'?>",
            '<?xml version="1.0" encoding="UTF-8"?>',
        )
    )
