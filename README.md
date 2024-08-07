# ASR_pipeline
Maintainable ASR pipeline for batch processing audio files.

Performs diarization and ASR. End results are formatted as EXB (Exmaralda) files.

### Setting up python environment

* Using `conda`: use the provided `environment.yml` file:
```bash
conda env create -f environment.yml
conda activate asr
```

* Using pip from a new python 3.11.6 environment:
```python
pip install -r requirements.txt
```

## Directory structure

```
ASR_pipeline
└── data
    ├── audio_input <- input audio
    ├── audio_16khz_mono_wav <- properly formatted wav files
    ├── asr
    ├── diarization
    └── exbs
```

The expected file structure can be generated with

```bash
mkdir data; cd data; mkdir audio_input audio_16khz_mono_wav asr diarization exbs; cd ..
```

## Audio preprocessing

Audio files should be aptly named, in wav format, sampled at 16kHz, with one channel only. To achieve this, one can use 
```bash
ffmpeg -i infile -ac 1 -ar 16000 -acodec pcm_s16le outfile.wav
```
to convert individual `infile` in appropriately formatted `outfile` with a reasonable filename you have to set yourself. 

For convenience this can be done also with `bash convert_audio.sh` script to transform all entries in `data/audio_input`, sequentially rename them (to 0.wav, 1.wav, ...) and save them to `data/audio_16khz_mono_wav`. Keep in mind, though, that the original file names will not be preserved.


## Sorting out HF credentials

Write your HF token from `hf.co/settings/tokens` in `ASR_pipeline/secrets.json` as:
```json
{
    "hf_token": "your token"
}
```
When running the code for the first time, you will be prompted to go to a HF modelcard and request access. Follow the URLs in the error messages and request access, it should be granted immediately. 

## Diarization

Diarization is performed with pyannote and is the basis for segmentation as well. RTTM format is used as an industry standard for this.

A script has been prepared to diarize all of the audios in `data/audio_16khz_mono_wav` iteratively, with only a single loading of the pipeline. It can be run as:
```bash
export CUDA_VISIBLE_DEVICES=0 # Select GPU core
python diarize.py
```

With this script for every wav file in `data/audio_16khz_mono_wav` a new file with the same name will be created in `data/diarization`.

Right now no restrictions are imposed on the maximal and minimal number of speakers allowed in the recoding, meaning that sometimes music/soundbites/laughter can be diarized as a separate speaker.

This step is about 100x faster if GPU is available, so while not neccessary, it is reccommended to use one. In case no GPU is to be used, the `export CUDA_VISIBLE_DEVICES` statement is moot and can be omitted.

## Segmentation and ASR

The script `chunk_and_transcribe.py` segments the audio files, saves them on disk, transcribes them, and finally cleans up the segmented wavs. Right now whisper is used to transcribe the files and the language can be set in the code.

```bash
export CUDA_VISIBLE_DEVICES=0 # Select GPU core
export ASR_LANGUAGE=croatian # Language to use for ASR, passed to Whisper internally.
python chunk_and_transcribe.py
```

This step was not benchmarked on CPU, but it is thought to be about 100x faster if GPU is available, so while not neccessary, it is reccommended to use one. In case no GPU is to be used, the `export CUDA_VISIBLE_DEVICES` statement is moot and can be omitted.

## Compiling EXB

For inspection and manual downstream tasks an EXB is produced for every input wav. This is done with `python generate_exbs.py` for all available files, and the results are saved to `data/exbs`.

# TL;DR:

Run:
```bash
pip install -r requirements.txt
mkdir data; cd data; mkdir audio_input audio_16khz_mono_wav asr diarization exbs; cd ..
```
Place your audio files in `audio_input`. Write your HF token to `secrets.json`. Run
```bash
bash convert_audio.sh
export CUDA_VISIBLE_DEVICES=6 # Select GPU core
python diarize.py
export ASR_LANGUAGE=croatian
python chunk_and_transcribe.py
python generate_exbs.py
```