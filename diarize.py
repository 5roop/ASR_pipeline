import json
import os
from pathlib import Path
import torch
from pyannote.audio import Pipeline

with open("secrets.json") as f:
    secrets = json.load(f)

# Load the pipeline:
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1", use_auth_token=secrets["hf_token"]
).to(torch.device("cuda"))


files = [
    file.with_suffix("").name
    for file in Path("data/audio_16khz_mono_wav/").glob("*.wav")
]
for file in files:
    # run the pipeline on an audio file
    diarization = pipeline(
        f"data/audio_16khz_mono_wav/{file}.wav",
        #  min_speakers=2, max_speakers=4
    )
    str_to_write = diarization.to_rttm()
    from pathlib import Path

    Path("data/diarization/").mkdir(exist_ok=True)
    outpath = Path(f"data/diarization/{file}.rttm")
    if outpath.exists():
        outpath.unlink()
    outpath.write_text(str_to_write)
