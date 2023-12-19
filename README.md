# ASR_pipeline
Maintainable ASR pipeline


## Directory structure

ASR_pipeline/
- data
    - audio_input <- place your audio files here
    - audio_16khz_mono_wav
    - asr
    - vad
    - diarization
## On downloading mixcloud files

Since there is no option to download audio directly from mixcloud, I used [this website](https://mixes.cloud/soundcloud-downloader/). Insert the mixcloud url in the form and click the `Download from Mixcloud` button. After some processing, a red button labelled `Download mix` pops up. You can copy its url, and then use wget to download it anywhere:

```bash
cd data/audio_input
wget https://stream11.mixcloud.com/secure/c/m4a/64/c/1/9/d/ad78-7373-4960-88ee-acccf3761515.m4a?sig=_2TbrScVf1qA5HXBnF7emA
cd ../..
```

It's likely the audio won't stay available at this link forever, so this is not a permanent solution and only good for a few files. Potentially it could be automated at some point.

## Converting audio to wav

A [script](01_convert_audio_to_16khz_mono_wav.sh) was written to convert the audio in a repeatable fashion. It also reorders the data sequentially, meaning that if another file is added and the script is rerun, the naming might change.

## VAD

Pyannote will be used as it seems to be the biggest player on HF. It requires entering some info in the model card forms, but grants access immediately.
Write your HF token from `hf.co/settings/tokens` in `ASR_pipeline/secrets.json` as:
```json
{
    "hf_token": "your token"
}
```

And request access on `hf.co/pyannote/segmentation`. Note you might have to register on multiple models to get stuff to work, but the error messages will lead you through it.

This is quite fast, for my exemplary file I needed about 50s for a 50min file.

I tested it out using [this notebook](02_vad_testing.ipynb).

## Diarization

Again I went with pyannote. Again it would not work without signing up for some more models, but the error messages lead me through it.

This part, investigated in [this notebook](02_diarization_testing.ipynb) was pretty slow. For 50 minutes of audio I needed 35 minutes of CPU time. It can be run on GPU, but I did not manage to due to some obscure nvidia errors that would require Damjan install new drivers. Perhaps with a careful downgrade of torch and pyannote this could be overcome.

2023-12-15T09:23:52: While working on Južne vesti I found a configuration that works. I'll add it to this repo.

2023-12-15T09:38:01: Wooo diarization with GPU only takes 46 seconds!

## ASR before segmenting

To investigate the potential usability of non-vad segmentation, I pass the whole file through the pipeline.

In addition, nvidia Nemo models are now being tested, soon to be included beside whisper, إِنْ شَاءَ ٱللَّٰهُ


# Further steps:

We shall take diarization output as the basis for further segmentation. Segments with < 0.1s duration shall not be transcribed, but they should stay in there to be obvious that something is happening there. A new framework to handle it will have to be devised.



### Bookkeeping:

Export environment to yaml file: `mamba env export > environment.yaml`
