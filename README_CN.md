# whisper_streaming
用于长篇语音转文字转换和翻译的 Whisper 实时流式传输(Whisper realtime streaming)

**将 Whisper 转换为实时转录系统**

Dominik Macháček、Raj Dabre、Ondřej Bojar 撰写的演示论文，2023 年

摘要：Whisper 是最近一种最先进的多语言语音识别和翻译模型之一，然而，它并非设计用于实时转录。在本文中，我们在 Whisper 的基础上构建了 Whisper-Streaming，并实现了类似 Whisper 模型的实时语音转录和翻译。Whisper-Streaming 使用本地协议与自适应延迟来实现流式转录。我们展示了 Whisper-Streaming 在未分段的长篇语音转录测试集上达到了高质量和 3.3 秒的延迟，并演示了它作为多语言会议现场转录服务中的组件的稳健性和实用性。

论文 PDF：https://aclanthology.org/2023.ijcnlp-demo.3.pdf

演示视频：https://player.vimeo.com/video/840442741

[幻灯片](http://ufallab.ms.mff.cuni.cz/~machacek/pre-prints/AACL23-2.11.2023-Turning-Whisper-oral.pdf) -- 2023 年 IJCNLP-AACL 15 分钟口头报告

请引用我们。[ACL 文集](https://aclanthology.org/2023.ijcnlp-demo.3/)，[Bibtex 引用](https://aclanthology.org/2023.ijcnlp-demo.3.bib):

```
@inproceedings{machacek-etal-2023-turning,
    title = "Turning Whisper into Real-Time Transcription System",
    author = "Mach{\'a}{\v{c}}ek, Dominik  and
      Dabre, Raj  and
      Bojar, Ond{\v{r}}ej",
    editor = "Saha, Sriparna  and
      Sujaini, Herry",
    booktitle = "Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics: System Demonstrations",
    month = nov,
    year = "2023",
    address = "Bali, Indonesia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.ijcnlp-demo.3",
    pages = "17--24",
}
```

## 安装

1) `pip install librosa soundfile` -- 音频处理库

2) Whisper 后端。

 集成了几种替代后端。最推荐的是 [faster-whisper](https://github.com/guillaumekln/faster-whisper)，支持 GPU。遵循其关于 NVIDIA 库的说明 -- 我们成功使用了 CUDNN 8.5.0 和 CUDA 11.7。使用 `pip install faster-whisper` 安装。

其次，另一种 less restrictive，但速度较慢的后端是 [whisper-timestamped](https://github.com/linto-ai/whisper-timestamped): `pip install git+https://github.com/linto-ai/whisper-timestamped`

第三，也可以通过 [OpenAI Whisper API](https://platform.openai.com/docs/api-reference/audio/createTranscription) 运行此软件。这种解决方案速度快，不需要 GPU，只需要一个小型 VM 就足够了，但您需要为 api 访问支付 OpenAI 费用。另请注意，由于每个音频片段被多次处理，[价格](https://openai.com/pricing) 将高于定价页面上的明显价格，因此在使用过程中请注意成本。设置更高的块大小将显著降低成本。 
使用 `pip install openai` 安装。

要使用 openai-api 后端运行，请确保您的 [OpenAI api 密钥](https://platform.openai.com/api-keys) 已设置在 `OPENAI_API_KEY` 环境变量中。例如，在运行之前执行：`export OPENAI_API_KEY=sk-xxx`，其中 *sk-xxx* 替换为您的 api 密钥。 

只有在选择时加载后端。未使用的后端不必安装。

3) 可选的，不推荐的: 句子分段器（又称句子分词器）

集成和评估

了两种缓冲区修剪选项。它们对质量和延迟有影响。默认的 "segment" 选项在我们的测试中表现更好，并且不需要安装任何句子分段器。

另一个选项是 "sentence" -- 在已确认的句子末尾修剪，需要安装句子分段器。它通过句号将有标点的文本拆分成句子，避免了不是句号的句号。分段器是语言特定的。未使用的分段器无需安装。我们集成了以下分段器，但欢迎提出更好的替代方案。

- `pip install opus-fast-mosestokenizer` 适用于代码 `as bn ca cs de el en es et fi fr ga gu hi hu is it kn lt lv ml mni mr nl or pa pl pt ro ru sk sl sv ta te yue zh` 的语言

- `pip install tokenize_uk` 适用于乌克兰语 -- `uk`

- 对于其他语言，我们集成了一个表现良好的多语言模型 `wtpslit`。它需要 `pip install torch wtpsplit`，以及它的神经模型 `wtp-canine-s-12l-no-adapters`。第一次使用时，它会下载到默认的 huggingface 缓存中。

- 我们没有找到语言 `as ba bo br bs fo haw hr ht jw lb ln lo mi nn oc sa sd sn so su sw tk tl tt` 的分段器，这些语言由 Whisper 支持，但不受 wtpsplit 支持。对于它们的默认回退选项是未指定语言的 wtpsplit。欢迎提出替代方案。

如果在 Windows 和 Mac 上安装 opus-fast-mosestokenizer 时遇到问题，我们建议仅使用不需要它的 "segment" 选项。

## 使用

### 从音频文件进行实时模拟

```
usage: whisper_online.py [-h] [--min-chunk-size MIN_CHUNK_SIZE] [--model {tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large-v3,large}] [--model_cache_dir MODEL_CACHE_DIR] [--model_dir MODEL_DIR] [--lan LAN] [--task {transcribe,translate}]
                         [--backend {faster-whisper,whisper_timestamped,openai-api}] [--vad] [--buffer_trimming {sentence,segment}] [--buffer_trimming_sec BUFFER_TRIMMING_SEC] [--start_at START_AT] [--offline] [--comp_unaware]
                         audio_path

positional arguments:
  audio_path            Filename of 16kHz mono channel wav, on which live streaming is simulated.

options:
  -h, --help            show this help message and exit
  --min-chunk-size MIN_CHUNK_SIZE
                        Minimum audio chunk size in seconds. It waits up to this time to do processing. If the processing takes shorter time, it waits, otherwise it processes the whole segment that was received by this time.
  --model {tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large-v3,large}
                        Name size of the Whisper model to use (default: large-v2). The model is automatically downloaded from the model hub if not present in model cache dir.
  --model_cache_dir MODEL_CACHE_DIR
                        Overriding the default model cache dir where models downloaded from the hub are saved
  --model_dir MODEL_DIR
                        Dir where Whisper model.bin and other files are saved. This option overrides --model and --model_cache_dir parameter.
  --lan LAN, --language LAN
                        Source language code, e.g. en,de,cs, or 'auto' for language detection.
  --task {transcribe,translate}
                        Transcribe or translate.
  --backend {faster-whisper,whisper_timestamped,openai-api}
                        Load only this backend for Whisper processing.
  --vad                 Use VAD = voice activity detection, with the default parameters.
  --buffer_trimming {sentence,segment}
                        Buffer trimming strategy -- trim completed sentences marked with punctuation mark and detected by sentence segmenter, or the completed segments returned by Whisper. Sentence segmenter must be installed for "sentence" option.
  --buffer_trimming_sec BUFFER_TRIMMING_SEC
                        Buffer trimming length threshold in seconds. If buffer length is longer, trimming sentence/segment is triggered.
  --start_at START_AT   Start processing audio at this time.
  --offline             Offline mode.
  --comp_unaware        Computationally unaware simulation.
```

示例:

它从预先录制的单声道 16k wav 文件模拟实时处理。

```
python3 whisper_online.py en-demo16.wav --language en --min-chunk-size 1 > out.txt
```

模拟模式:

- 默认模式，没有特殊选项: 从文件实时模拟，计算机感知。块大小是 `MIN_CHUNK_SIZE` 或更大，如果在最后一次更新计算期间到达更多音频，则更大。

- `--comp_unaware` 选项: 计算机不知道的模拟。这意味着当模型正在计算时，计时器（计算发射时间的计时器）会“停止”。块大小始终为 `MIN_CHUNK_SIZE`。延迟仅由于模型无法确认输出，例如由于语言歧义等原因造成，并非由于慢硬件或次优实现。我们实现此功能以找到延迟的下限。

- `--start_at START_AT`: 从此时间开始处理音频。第一个更新将接收到 `START_AT` 之前的整个音频。对于调试很有用，例如，当我们观察到音频文件中特定时间的错误时，并希望快速重现它，而不需等待很长时间。

- `--offline` 选项: 它一次性在离线模式下处理整个音频文件。我们实现它是为了找出在给定音频文件上可能的最低 WER。

### 输出格式

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

[在此处查看描述](https://github.com/ufal/whisper_streaming/blob/d915d790a62d7be4e7392dde1480e7981eb142ae/whisper_online.py#L361)

### 作为模块

TL;DR: 使用 OnlineASRProcessor 对象及其方法 insert_audio_chunk 和 process_iter。

代码 whisper_online.py 有良好的注释，请将其阅读为完整文档。

此伪代码描述了我们建议用于您的实现的接口。您可以为应用程序实现任何您需要的功能。

```
from whisper_online import *

src_lan = "en"  # 源语言
tgt_lan = "en"  # 目标语言 -- 对于 ASR 来说与源语言相同，如果使用 translate 任务，则为 "en"

asr = FasterWhisperASR(lan, "large-v2")  # 加载和包装 Whisper 模型
# 设置选项:
# asr.set_translate_task()  # 它将从 lan 翻译为英语
# asr.use_vad()  # 设置使用 VAD

online = OnlineASRProcessor(asr)  # 使用默认缓冲区修剪选项创建处理对象

while audio_has_not_ended:   # 处理循环:
	a = # 接收新的音频块（例如等待 min_chunk_size 秒，...）
	online.insert_audio_chunk(a)
	o = online.process_iter()
	print(o) # 处理当前部分输出
# 在此音频处理结束时
o = online.finish()
print(o)  # 处理最后输出

online.init()  # 如果要重新使用对象进行下一次音频处理，请刷新
```

### 服务器 -- 实时来自麦克风

`whisper_online_server.py` 具有与 `whisper_online.py` 相同的模型选项，以及 TCP 连接的 `--host` 和 `--port`。查看帮助消息（`-h` 选项）。

客户端示例:

```
arecord -f S16_LE -c1 -r 16000 -t raw -D default | nc localhost 43001
```

- arecord 从声音设备（例如麦克风）发送实时音频，以原始音频格式发送 -- 16000 采样率，单声道，S16_LE -- 带符号的 16 位整数低端。 （使用适合您的 arecord 替代品）

- nc 是具有服务器主机和端口的 netcat。

## 背景

默认的 Whisper 适用于最多包含一个完整句子的 30 秒音频块。较长的音频文件必须拆分为较短的块，并与“初始化提示”合并。在低延迟的同时流式模式下，简单且天真的固定大小的窗口切块效果不佳，它可能会将一个单词分割成两部分。还需要知道何时转录稳定，应该被确认（“提交”）和跟进，以及未来的内容何时会使转录更清晰。

为此，有 LocalAgreement-n 策略：如果 n 个连续更新，每个更新都有一个新的可用音频流块，并且它们都同意前缀转录，则它被确认。 （参考：CUNI-KIT 在 IWSLT 2022 等）

在本项目中，我们重用了来自此演示的 Peter Polák 的想法：
https://github.com/pe-trik/transformers/blob/online_decode/examples/pytorch/online-decoding/whisper-online-demo.py
但它不执行任何句子分割，但 Whisper 生成标点符号，而库 `faster-whisper` 和 `whisper_transcribed` 生成了单词级时间戳。简而言之：我们连续处理新的音频块，发出经过 2 次迭代确认的转录，并在确认完整句子的时间戳上滚动音频处理缓冲区。处理音频缓冲区不会太长，处理速度很快。

更详细地说：我们使用初始化提示，处理不准确的时间戳，重新处理已确认的句子前缀并跳过它们，确保它们不重叠，并限制处理缓冲区窗口。

欢迎贡献。

### 性能评估

[查看论文](http://www.afnlp.org/conferences/ijcnlp2023/proceedings/main-demo/cdrom/pdf/2023.ijcnlp-demo.3.pdf)

## 联系方式

Dominik Macháček, machacek@ufal.mff.cuni.cz