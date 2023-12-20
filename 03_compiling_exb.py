# This script accepts RTTM and json datasources and compiles them all in EXB for manual checking
# and decision making. RTTMs could be from VAD, diarization, or resegmentation. JSON should be from 
# ASR-ing the entire audio.
from pathlib import Path
import pandas as pd
from lxml import etree as ET
from utils import read_json, read_rttm

idx = 0

audio_path = Path(f"data/audio_16khz_mono_wav/{idx}.wav")
diarization_path = Path(f"data/diarization/{idx}.rttm")
vad_path = Path(f"data/vad/{idx}.rttm")
asr_path = Path(f"data/asr/{idx}.json")
template_path = Path("exb_template.xml")
out_path = "test.exb"

exb = ET.fromstring(template_path.read_bytes())
diarization_df = read_rttm(diarization_path)
vad_df = read_rttm(vad_path)
asr_df = read_json(asr_path)

def add_df_to_template(exb: ET.Element, df: pd.DataFrame, tier_name: str = "", diarization:bool=False)-> ET.Element:
    """Adds data from a pandas dataframe with columns ["start", "end", "speaker_name"] or ["start", "end", "text"]. Adds one or more tiers. Adds appropriate <tli> and speakers.

    Args:
        exb (ET.Element): parsed EXB template (as in the root of this repo.)
        df (pd.DataFrame): dataframe with data to be added.
        tier_name (str, optional): what the added tier should be called. This argument is moot for cases where diarization=True, but should be passed otherwise.
        diarization (bool, optional): If the input data is diarization, this should be set to True. This means that for every speaker in the column 'speaker_name', a new tier will be constructed with placeholder text ('-'). Defaults to False.

    Returns:
        ET.Element: lxml.etree.Element with all of the added data.
    """    
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
        if tier_name == "":
            raise AttributeError("Please pass a name for this tier.")
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
exb = add_df_to_template(exb, diarization_df,  diarization=True)
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
