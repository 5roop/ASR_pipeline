COUNTER=0
for audio_in in $(ls data/audio_input)
do
    echo $audio_in
    ffmpeg -i "data/audio_input/$audio_in" -ac 1 -ar 16000 -acodec pcm_s16le "data/audio_16khz_mono_wav/$COUNTER.wav"
    let COUNTER++
done