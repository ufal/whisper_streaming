# whisper_streaming
Whisper realtime streaming for long speech-to-text transcription and translation

## Installation

```
pip install git+https://github.com/linto-ai/whisper-timestamped
XDG_CACHE_HOME=$(pwd)/pip-cache pip install git+https://github.com/linto-ai/whisper-timestamped
pip install librosa
pip install opus-fast-mosestokenizer
pip install torch
```

## Usage

```
(p3) $ python3 whisper_online.py -h
usage: whisper_online.py [-h] [--min-chunk-size MIN_CHUNK_SIZE] [--model MODEL] [--model_dir MODEL_DIR] [--lan LAN] [--start_at START_AT] audio_path

positional arguments:
  audio_path

options:
  -h, --help            show this help message and exit
  --min-chunk-size MIN_CHUNK_SIZE
                        Minimum audio chunk size in seconds. It waits up to this time to do processing. If the processing takes shorter time, it waits, otherwise it processes the whole segment that was received by this time.
  --model MODEL         name of the Whisper model to use (default: large-v2, options: {tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large}
  --model_dir MODEL_DIR
                        the path where Whisper models are saved (or downloaded to). Default: ./disk-cache-dir
  --lan LAN, --language LAN
                        Language code for transcription, e.g. en,de,cs.
  --start_at START_AT   Start processing audio at this time.
```

Example:

```
python3 whisper_online.py en-demo16.wav --language en --min-chunk-size 1 > out.txt
```

## Output format

```
2691.4399 300 1380 Chairman, thank you.
6914.5501 1940 4940 If the debate today had a
9019.0277 5160 7160 the subject the situation in
10065.1274 7180 7480 Gaza
11058.3558 7480 9460 Strip, I might
12224.3731 9460 9760 have
13555.1929 9760 11060 joined Mrs.
14928.5479 11140 12240 De Kaiser and all the
16588.0787 12240 12560 other
18324.9285 12560 14420 colleagues across the
```

[See description here](https://github.com/ufal/whisper_streaming/blob/d915d790a62d7be4e7392dde1480e7981eb142ae/whisper_online.py#L361)



## Background

Default Whisper is intended for audio chunks of at most 30 seconds that contain one full sentence. Longer audio files must be split to shorter chunks and merged with "init prompt". In low latency simultaneous streaming mode, the simple and naive chunking fixed-sized windows does not work well, it can split a word in the middle. It is also necessary to know when the transcribt is stable, should be confirmed ("commited") and followed up, and when the future content makes the transcript clearer. 

For that, there is LocalAgreement-n policy: if n consecutive updates, each with a newly available audio stream chunk, agree on a prefix transcript, it is confirmed. (Reference: CUNI-KIT at IWSLT 2022 etc.)

In this project, we re-use the idea of Peter Polák from this demo: https://github.com/pe-trik/transformers/blob/online_decode/examples/pytorch/online-decoding/whisper-online-demo.py However, it doesn't do any sentence segmentation, but Whisper produces punctuation and `whisper_transcribed` makes word-level timestamps. In short: we consecutively process new audio chunks, emit the transcripts that are confirmed by 2 iterations, and scroll the audio processing buffer on a timestamp of a confirmed complete sentence. The processing audio buffer is not too long and the processing is fast.

In more detail: we use the init prompt, we handle the inaccurate timestamps, we re-process confirmed sentence prefixes and skip them, making sure they don't overlap, and we limit the processing buffer window. 

This project is work in progress. Contributions are welcome.

### Tests

Rigorous quality and latency tests are pending.

Small initial debugging shows that on a fluent monologue speech without pauses, both the quality and latency of English and German ASR is impressive. 

Czech ASR tests show that multi-speaker interview with pauses and disfluencies is challenging. However, parameters should be tuned.

## Contact

Dominik Macháček, machacek@ufal.mff.cuni.cz



