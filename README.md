# whisper_streaming
Whisper realtime streaming for long speech-to-text transcription and translation

**Turning Whisper into Real-Time Transcription System**

Demonstration paper, by Dominik Macháček, Raj Dabre, Ondřej Bojar, 2023

Abstract:    Whisper is one of the recent state-of-the-art multilingual speech recognition and translation models, however, it is not designed for real time transcription. In this paper, we build on top of Whisper and create Whisper-Streaming, an implementation of real-time speech transcription and translation of Whisper-like models. Whisper-Streaming uses local agreement policy with self-adaptive latency to enable streaming transcription. We show that Whisper-Streaming achieves high quality and 3.3 seconds latency on unsegmented long-form speech transcription test set, and we demonstrate its robustness and practical usability as a component in live transcription service at a multilingual conference. 


Pre-print: https://arxiv.org/abs/2307.14743

Demo video: https://player.vimeo.com/video/840442741

## Installation

1) ``pip install librosa`` -- audio processing library

2) Whisper backend.

Two alternative backends are integrated. The most recommended one is [faster-whisper](https://github.com/guillaumekln/faster-whisper) with GPU support. Follow their instructions for NVIDIA libraries -- we succeeded with CUDNN 8.5.0 and CUDA 11.7. Install with `pip install faster-whisper`.

Alternative, less restrictive, but slower backend is [whisper-timestamped](https://github.com/linto-ai/whisper-timestamped): `pip install git+https://github.com/linto-ai/whisper-timestamped`

The backend is loaded only when chosen. The unused one does not have to be installed.

3) Sentence segmenter (aka sentence tokenizer) 

It splits punctuated text to sentences by full stops, avoiding the dots that are not full stops. The segmenters are language specific.
The unused one does not have to be installed. We integrate the following segmenters, but suggestions for better alternatives are welcome.

- for the languages with codes `as bn ca cs de el en es et fi fr ga gu hi hu is it kn lt lv ml mni mr nl or pa pl pt ro ru sk sl sv ta te yue zh`
- - Apple Silicon: `mosestokenizer` 
- - Otherwise: `pip install opus-fast-mosestokenizer` 

- `pip install tokenize_uk` for Ukrainian -- `uk`

- for other languages, we integrate a good performing multi-lingual model of `wtpslit`. It requires `pip install torch wtpsplit`, and its neural model `wtp-canine-s-12l-no-adapters`. It is downloaded to the default huggingface cache during the first use. 

- we did not find a segmenter for languages `as ba bo br bs fo haw hr ht jw lb ln lo mi nn oc sa sd sn so su sw tk tl tt` that are supported by Whisper and not by wtpsplit. The default fallback option for them is wtpsplit with unspecified language. Alternative suggestions welcome.


## Usage

### Realtime simulation from audio file

```
usage: whisper_online.py [-h] [--min-chunk-size MIN_CHUNK_SIZE] [--model {tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large}] [--model_cache_dir MODEL_CACHE_DIR] [--model_dir MODEL_DIR] [--lan LAN] [--task {transcribe,translate}]
                         [--start_at START_AT] [--backend {faster-whisper,whisper_timestamped}] [--offline] [--comp_unaware] [--vad]
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
  --comp_unaware        Computationally unaware simulation.
  --vad                 Use VAD = voice activity detection, with the default parameters.
```

Example:

It simulates realtime processing from a pre-recorded mono 16k wav file.

```
python3 whisper_online.py en-demo16.wav --language en --min-chunk-size 1 > out.txt
```

Simulation modes:

- default mode, no special option: real-time simulation from file, computationally aware. The chunk size is `MIN_CHUNK_SIZE` or larger, if more audio arrived during last update computation.

- `--comp_unaware` option: computationally unaware simulation. It means that the timer that counts the emission times "stops" when the model is computing. The chunk size is always `MIN_CHUNK_SIZE`. The latency is caused only by the model being unable to confirm the output, e.g. because of language ambiguity etc., and not because of slow hardware or suboptimal implementation. We implement this feature for finding the lower bound for latency.

- `--start_at START_AT`: Start processing audio at this time. The first update receives the whole audio by `START_AT`. It is useful for debugging, e.g. when we observe a bug in a specific time in audio file, and want to reproduce it quickly, without long waiting.

- `--ofline` option: It processes the whole audio file at once, in offline mode. We implement it to find out the lowest possible WER on given audio file.



### Output format

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

### As a module

TL;DR: use OnlineASRProcessor object and its methods insert_audio_chunk and process_iter. 

The code whisper_online.py is nicely commented, read it as the full documentation.


This pseudocode describes the interface that we suggest for your implementation. You can implement e.g. audio from mic or stdin, server-client, etc.

```
from whisper_online import *

src_lan = "en"  # source language
tgt_lan = "en"  # target language  -- same as source for ASR, "en" if translate task is used


asr = FasterWhisperASR(lan, "large-v2")  # loads and wraps Whisper model
# set options:
# asr.set_translate_task()  # it will translate from lan into English
# asr.use_vad()  # set using VAD 


online = OnlineASRProcessor(tgt_lan, asr)  # create processing object


while audio_has_not_ended:   # processing loop:
	a = # receive new audio chunk (and e.g. wait for min_chunk_size seconds first, ...)
	online.insert_audio_chunk(a)
	o = online.process_iter()
	print(o) # do something with current partial output
# at the end of this audio processing
o = online.finish()
print(o)  # do something with the last output


online.init()  # refresh if you're going to re-use the object for the next audio
```

### Server

`whisper_online_server.py` has the same model options as `whisper_online.py`, plus `--host` and `--port` of the TCP connection. See help message (`-h` option).

Client example:

```
arecord -f S16_LE -c1 -r 16000 -t raw -D default | nc localhost 43001
```

- arecord sends realtime audio from a sound device (e.g. mic), in raw audio format -- 16000 sampling rate, mono channel, S16\_LE -- signed 16-bit integer low endian. (use the alternative to arecord that works for you)

- nc is netcat with server's host and port


## Background

Default Whisper is intended for audio chunks of at most 30 seconds that contain
one full sentence. Longer audio files must be split to shorter chunks and
merged with "init prompt". In low latency simultaneous streaming mode, the
simple and naive chunking fixed-sized windows does not work well, it can split
a word in the middle. It is also necessary to know when the transcribt is
stable, should be confirmed ("commited") and followed up, and when the future
content makes the transcript clearer. 

For that, there is LocalAgreement-n policy: if n consecutive updates, each with
a newly available audio stream chunk, agree on a prefix transcript, it is
confirmed. (Reference: CUNI-KIT at IWSLT 2022 etc.)

In this project, we re-use the idea of Peter Polák from this demo:
https://github.com/pe-trik/transformers/blob/online_decode/examples/pytorch/online-decoding/whisper-online-demo.py
However, it doesn't do any sentence segmentation, but Whisper produces
punctuation and the libraries `faster-whisper` and `whisper_transcribed` make
word-level timestamps. In short: we
consecutively process new audio chunks, emit the transcripts that are confirmed
by 2 iterations, and scroll the audio processing buffer on a timestamp of a
confirmed complete sentence. The processing audio buffer is not too long and
the processing is fast.

In more detail: we use the init prompt, we handle the inaccurate timestamps, we
re-process confirmed sentence prefixes and skip them, making sure they don't
overlap, and we limit the processing buffer window. 

Contributions are welcome.

### Tests

[See the results in paper.](https://arxiv.org/abs/2307.14743)

## Contact

Dominik Macháček, machacek@ufal.mff.cuni.cz



