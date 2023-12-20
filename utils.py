import json
from pathlib import Path

import pandas as pd


def read_rttm(path: Path) -> pd.DataFrame:
    """Reads RTTM format (output of VAD/diarization).

    Args:
        path (Path): Path to read. Should be RTTM formatted.

    Returns:
        pd.DataFrame: dataframe with columns ["start", "end", "speaker_name"]. Start and end cols
        are floats. End is calculated from start and duration.
    """    
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


def read_json(path: Path) -> pd.DataFrame:
    """Reads json file, which it expects to be a dump of ASR pipeline.

    Args:
        path (Path): json location.

    Returns:
        pd.DataFrame: dataframe with columns ["start", "end", "text"].
    """    
    import json

    d = json.loads(path.read_text())
    df = pd.DataFrame(d.get("chunks"))
    df["start"] = df.timestamp.apply(lambda l: float(l[0])).round(decimals=3)
    df["end"] = df.timestamp.apply(lambda l: float(l[1])).round(decimals=3)
    return df[["start", "end", "text"]]


def process_whisper(files_to_process: list[str | Path]) -> list[str]:
    from datasets import Dataset, Audio
    from transformers.pipelines.pt_utils import KeyDataset

    ds = Dataset.from_dict({"audio": files_to_process}).cast_column(
        "audio", Audio(sampling_rate=16000)
    )
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    from pathlib import Path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3"
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
    except:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=False,
        )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    result = pipe(
        KeyDataset(ds, "audio"),
        generate_kwargs={"language": "croatian"},
    )
    transcripts = [i.get("text") for i in result]
    return transcripts


def process_nemo(files_to_process: list[str | Path]) -> list[str]:
    import nemo.collections.asr as nemo_asr

    asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
        "nvidia/stt_hr_conformer_transducer_large"
    )
    result = asr_model.transcribe(files_to_process)
    return result[0]
