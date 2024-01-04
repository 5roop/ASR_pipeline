from pathlib import Path

from pydub import AudioSegment

from utils import process_whisper, read_rttm
from os import environ

ASR_LANGUAGE = environ.get("ASR_LANGUAGE", "croatian")

files = [
    file.with_suffix("").name
    for file in Path("data/audio_16khz_mono_wav/").glob("*.wav")
]
for file in files:
    audio_path = Path(f"data/audio_16khz_mono_wav/{file}.wav")
    diarization_path = Path(f"data/diarization/{file}.rttm")
    diarization_df = read_rttm(diarization_path)
    diarization_df["duration"] = diarization_df.end - diarization_df.start
    tempdir = Path("./temp_files")
    tempdir.mkdir(exist_ok=True)
    audio = AudioSegment.from_wav(audio_path)
    filenames = []
    for i, row in diarization_df.iterrows():
        filename = (
            audio_path.with_suffix("").name + f"__{row['start']}__{row['end']}.wav"
        )
        start = int(1000 * row["start"])
        end = int(1000 * row["end"])
        outpath = str(tempdir / filename)
        filenames.append(outpath)
        audio[start:end].export(outpath, format="wav")
    diarization_df["path"] = filenames

    files = diarization_df.path.tolist()
    diarization_df["whisper"] = process_whisper(files, lang=ASR_LANGUAGE)
    for i in tempdir.glob("*.wav"):
        i.unlink()
    tempdir.rmdir()
    diarization_df.to_csv(f"data/asr/{file}_diarization_whisper.csv", index=False)
