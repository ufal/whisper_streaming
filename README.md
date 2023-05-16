# whisper_streaming
Whisper realtime streaming for long speech-to-text transcription and translation

## Installation

This code work with two kinds of backends. Both require

```
pip install librosa
pip install opus-fast-mosestokenizer
```

The most recommended backend is [faster-whisper](https://github.com/guillaumekln/faster-whisper) with GPU support. Follow their instructions for NVIDIA libraries -- we succeeded with CUDNN 8.5.0 and CUDA 11.7. Install with `pip install faster-whisper`.

Alternative, less restrictive, but slowe backend is [whisper-timestamped](https://github.com/linto-ai/whisper-timestamped): `pip install git+https://github.com/linto-ai/whisper-timestamped`

The backend is loaded only when chosen. The unused one does not have to be installed.

## Usage

```
usage: whisper_online.py [-h] [--min-chunk-size MIN_CHUNK_SIZE] [--model {tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large}] [--model_cache_dir MODEL_CACHE_DIR] [--model_dir MODEL_DIR] [--lan LAN] [--task {transcribe,translate}]
                         [--start_at START_AT] [--backend {faster-whisper,whisper_timestamped}] [--offline] [--vad]
                         audio_path

positional arguments:
  audio_path            Filename of 16kHz mono channel wav, on which live streaming is simulated.

options:
  -h, --help            show this help message and exit
  --min-chunk-size MIN_CHUNK_SIZE
                        Minimum audio chunk size in seconds. It waits up to this time to do processing. If the processing takes shorter time, it waits, otherwise it processes the whole segment that was received by this time.
  --model {tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large}
                        Name size of the Whisper model to use (default: large-v2). The model is automatically downloaded from the model hub if not present in model cache dir.
  --model_cache_dir MODEL_CACHE_DIR
                        Overriding the default model cache dir where models downloaded from the hub are saved
  --model_dir MODEL_DIR
                        Dir where Whisper model.bin and other files are saved. This option overrides --model and --model_cache_dir parameter.
  --lan LAN, --language LAN
                        Language code for transcription, e.g. en,de,cs.
  --task {transcribe,translate}
                        Transcribe or translate.
  --start_at START_AT   Start processing audio at this time.
  --backend {faster-whisper,whisper_timestamped}
                        Load only this backend for Whisper processing.
  --offline             Offline mode.
  --vad                 Use VAD = voice activity detection, with the default parameters. 
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



