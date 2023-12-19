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
  