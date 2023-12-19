from lxml import etree as ET
from pathlib import Path
import pandas as pd

idx = 0

audio_path = Path(f"data/audio_16khz_mono_wav/{idx}.wav")
diarization_path = Path(f"data/diarization/{idx}.rttm")
vad_path = Path(f"data/vad/{idx}.rttm")
asr_path = Path(f"data/asr/{idx}.json")
template_path = Path("exb_template.xml")
out_path = "test.exb"

exb = ET.fromstring(template_path.read_bytes())


def read_rttm(path):
    df = pd.read_csv(
        path,
        sep=" ",
        header=None,
        names=[
            "type",
            "fileid",
            "channelid",
            "start",
            "duration",
            "ortography_field",
            "speaker_type",
            "speaker_name",
            "confidence_score",
            "signal_lookahead_time",
        ],
    )
    df["start"] = df.start.astype(float).round(decimals=3)
    df["duration"] = df.duration.astype(float).round(decimals=3)
    df["end"] = df.start + df.duration
    return df[["start", "end", "speaker_name"]]

def read_json(path):
    import json
    d = json.loads(path.read_text())
    df = pd.DataFrame(d.get("chunks"))
    df["start"] = df.timestamp.apply(lambda l: float(l[0])).round(decimals=3)
    df["end"] = df.timestamp.apply(lambda l: float(l[1])).round(decimals=3)
    return df[["start", "end", "text"]]
    
diarization_df = read_rttm(diarization_path)
vad_df = read_rttm(vad_path)
asr_df = read_json(asr_path)

def add_df_to_template(exb: ET.Element, df: pd.DataFrame, tier_name: str, diarization:bool=False)-> ET.Element:
    # Add speaker for the purpose:
    if diarization:
        for speaker_name in df.speaker_name.unique():
            speaker = ET.Element("speaker", attrib={"id": speaker_name})
            abbreviation = ET.Element("abbreviation")
            abbreviation.text = speaker_name
            speaker.append(
                abbreviation
            )
            exb.find(".//speakertable").append(speaker)
    else:
        speaker = ET.Element("speaker", attrib={"id": tier_name})
        abbreviation = ET.Element("abbreviation")
        abbreviation.text = tier_name
        speaker.append(
            abbreviation
        )
        exb.find(".//speakertable").append(speaker)

    # Add <tli>:
    timeline = exb.find(".//common-timeline")
    N = len(timeline.findall(".//tli"))
    for t in set(df.start.values).union( set(df.end.values)):
        tli = ET.Element("tli", attrib={"id": f"T{N}", "time":str(t)})
        timeline.append(tli)
        N += 1
    # Sort timeline
    timeline[:] = sorted(timeline, key=lambda child: float(child.get("time")))
    # Prepare inverse mapper (seconds -> id):
    mapper = {tli.get("time"): tli.get("id") for tli in timeline.findall("tli")}
    # Add new tier(s):
    if diarization:
        for speaker_name in df.speaker_name.unique():
            tier = ET.Element("tier", attrib=dict(
                id=speaker_name,
                category="v",
                type="t",
                display_name=speaker_name,
                speaker=speaker_name
            ))
            for i, row in df[df.speaker_name == speaker_name].iterrows():
                event = ET.Element("event", attrib=dict(
                    start= mapper.get(str(row["start"])),
                    end=mapper.get(str(row["end"]))
                ))
                event.text = "-"
                tier.append(event)
            exb.find("basic-body").append(tier)
    else:
        tier = ET.Element("tier", attrib=dict(
            id=tier_name,
            category="v",
            type="t",
            display_name=tier_name,
            speaker=tier_name
        ))
        for i, row in df.iterrows():
            event = ET.Element("event", attrib=dict(
                start= mapper.get(str(row["start"])),
                end=mapper.get(str(row["end"]))
            ))
            try:
                event.text = row["speaker_name"]
            except:
                event.text = row["text"]
            tier.append(event)
        exb.find("basic-body").append(tier)
    return exb

exb = add_df_to_template(exb, vad_df, tier_name="vad")
exb = add_df_to_template(exb, diarization_df, tier_name="diarization", diarization=True)
exb = add_df_to_template(exb, asr_df, tier_name="asr")
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
