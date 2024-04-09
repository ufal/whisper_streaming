#!/usr/bin/env python3
import sys
import numpy as np
import librosa  
from functools import lru_cache
import time
import io
import soundfile as sf
import math

@lru_cache
def load_audio(fname):
    a, _ = librosa.load(fname, sr=16000, dtype=np.float32)
    return a

def load_audio_chunk(fname, beg, end):
    audio = load_audio(fname)
    beg_s = int(beg*16000)
    end_s = int(end*16000)
    return audio[beg_s:end_s]


# Whisper 后端

class ASRBase:

    sep = " "   # 使用该字符连接转录单词（对于 whisper_timestamped 为 " "，对于 faster-whisper 为 ""，因为后者在需要时会插入空格）

    def __init__(self, lan, modelsize=None, cache_dir=None, model_dir=None, logfile=sys.stderr):
        self.logfile = logfile

        self.transcribe_kargs = {}
        if lan == "auto":
            self.original_language = None
        else:
            self.original_language = lan

        self.model = self.load_model(modelsize, cache_dir, model_dir)


    def load_model(self, modelsize, cache_dir):
        raise NotImplemented(f"must be implemented in the child class\n必须在子类中实现")

    def transcribe(self, audio, init_prompt=""):
        raise NotImplemented(f"must be implemented in the child class\n必须在子类中实现")

    def use_vad(self):
        raise NotImplemented(f"must be implemented in the child class\n必须在子类中实现")


class WhisperTimestampedASR(ASRBase):
    """使用 whisper_timestamped 库作为后端。最初我们在此后端上测试了代码。它运行良好，但比 faster-whisper 慢。
    另一方面，GPU 可能更容易安装。
    """

    sep = " "

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        import whisper
        import whisper_timestamped
        from whisper_timestamped import transcribe_timestamped
        self.transcribe_timestamped = transcribe_timestamped
        if model_dir is not None:
            print(f"ignoring model_dir, not implemented\n忽略 model_dir，未实现",file=self.logfile)
        return whisper.load_model(modelsize, download_root=cache_dir)

    def transcribe(self, audio, init_prompt=""):
        result = self.transcribe_timestamped(self.model,
                audio, language=self.original_language,
                initial_prompt=init_prompt, verbose=None,
                condition_on_previous_text=True, **self.transcribe_kargs)
        return result
 
    def ts_words(self,r):
        # return: transcribe result object to [(beg,end,"word1"), ...]
        # 返回：将转录结果对象转换为 [(开始，结束，"单词1"), ...]
        o = []
        for s in r["segments"]:
            for w in s["words"]:
                t = (w["start"],w["end"],w["text"])
                o.append(t)
        return o

    def segments_end_ts(self, res):
        return [s["end"] for s in res["segments"]]

    def use_vad(self):
        self.transcribe_kargs["vad"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"




class FasterWhisperASR(ASRBase):
    """使用 faster-whisper 库作为后端。运行速度要快得多，大约是 4 倍（在离线模式下）。对于 GPU，它需要使用特定的 CUDNN 版本进行安装。
    """

    sep = ""

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        from faster_whisper import WhisperModel
        if model_dir is not None:
            print(f"Loading whisper model from model_dir {model_dir}. modelsize and cache_dir parameters are not used.\n从 model_dir {model_dir} 加载 whisper 模型。不使用 modelsize 和 cache_dir 参数。",file=self.logfile)
            model_size_or_path = model_dir
        elif modelsize is not None:
            model_size_or_path = modelsize
        else:
            raise ValueError("必须设置 modelsize 或 model_dir 参数")


        # 在 NVIDIA L40 上工作得很快且可靠
        model = WhisperModel(model_size_or_path, device="cuda", compute_type="float16", download_root=cache_dir)

        # 或者在 GPU 上使用 INT8 运行
        # tested: the transcripts were different, probably worse than with FP16, and it was slightly (appx 20%) slower
        #model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")

        # 或者在 CPU 上使用 INT8 运行
        # 经测试：能用，但速度慢，大约比 cuda FP16 慢 10 倍
#        model = WhisperModel(modelsize, device="cpu", compute_type="int8") #, download_root="faster-disk-cache-dir/")
        return model

    def transcribe(self, audio, init_prompt=""):

        # 经测试：beam_size=5 比 1 更快且更好（在来自 En ESIC 的一个 200 秒文档上，最小块 0.01）
        segments, info = self.model.transcribe(audio, language=self.original_language, initial_prompt=init_prompt, beam_size=5, word_timestamps=True, condition_on_previous_text=True, **self.transcribe_kargs)
        #print(info)  # info contains language detection result

        return list(segments)

    def ts_words(self, segments):
        o = []
        for segment in segments:
            for word in segment.words:
                # not stripping the spaces -- should not be merged with them!
                # 不去掉空格 -- 不应该与空格合并！
                w = word.word
                t = (word.start, word.end, w)
                o.append(t)
        return o

    def segments_end_ts(self, res):
        return [s.end for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"


class OpenaiApiASR(ASRBase):
    """Uses OpenAI's Whisper API for audio transcription."""

    def __init__(self, lan=None, temperature=0, logfile=sys.stderr):
        self.logfile = logfile

        self.modelname = "whisper-1"  
        self.original_language = None if lan == "auto" else lan # ISO-639-1 language code
        self.response_format = "verbose_json" 
        self.temperature = temperature

        self.load_model()

        self.use_vad_opt = False

        # 在 set_translate_task 函数中重置任务
        self.task = "transcribe"

    def load_model(self, *args, **kwargs):
        from openai import OpenAI
        self.client = OpenAI()

        self.transcribed_seconds = 0  # for logging how many seconds were processed by API, to know the cost|用于记录 API 处理了多少秒的日志，以了解成本
        

    def ts_words(self, segments):
        no_speech_segments = []
        if self.use_vad_opt:
            for segment in segments.segments:
                # TODO：可以从外部设置阈值
                if segment["no_speech_prob"] > 0.8:
                    no_speech_segments.append((segment.get("start"), segment.get("end")))

        o = []
        for word in segments.words:
            start = word.get("start")
            end = word.get("end")
            if any(s[0] <= start <= s[1] for s in no_speech_segments):
                # print("Skipping word", word.get("word"), "because it's in a no-speech segment")
                continue
            o.append((start, end, word.get("word")))
        return o


    def segments_end_ts(self, res):
        return [s["end"] for s in res.words]

    def transcribe(self, audio_data, prompt=None, *args, **kwargs):
        # 将音频数据写入缓冲区
        buffer = io.BytesIO()
        buffer.name = "temp.wav"
        sf.write(buffer, audio_data, samplerate=16000, format='WAV', subtype='PCM_16')
        buffer.seek(0)  # 将缓冲区的位置重置到开头

        self.transcribed_seconds += math.ceil(len(audio_data)/16000)  # 将时间舍入到整秒

        params = {
            "model": self.modelname,
            "file": buffer,
            "response_format": self.response_format,
            "temperature": self.temperature,
            "timestamp_granularities": ["word", "segment"]
        }
        if self.task != "translate" and self.original_language:
            params["language"] = self.original_language
        if prompt:
            params["prompt"] = prompt

        if self.task == "translate":
            proc = self.client.audio.translations
        else:
            proc = self.client.audio.transcriptions

        # 处理转录/翻译
        transcript = proc.create(**params)
        print(f"OpenAI API processed accumulated {self.transcribed_seconds} seconds",file=self.logfile)

        return transcript

    def use_vad(self):
        self.use_vad_opt = True

    def set_translate_task(self):
        self.task = "translate"




class HypothesisBuffer:

    def __init__(self, logfile=sys.stderr):
        self.commited_in_buffer = []
        self.buffer = []
        self.new = []

        self.last_commited_time = 0
        self.last_commited_word = None

        self.logfile = logfile

    def insert(self, new, offset):
        # 比较 self.commited_in_buffer 和 new。仅插入 new 中扩展 commited_in_buffer 的单词，这意味着它们在时间上大致在 last_commited_time 之后，并且在内容上是新的
        # 将新的尾部添加到 self.new 中
        
        new = [(a+offset,b+offset,t) for a,b,t in new]
        self.new = [(a,b,t) for a,b,t in new if a > self.last_commited_time-0.1]

        if len(self.new) >= 1:
            a,b,t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    # 它将搜索在 commited 和 new 中相同的连续的 1、2、...、5 个单词（n-gram）。如果找到了相同的单词组合，它们将被丢弃。
                    cn = len(self.commited_in_buffer)
                    nn = len(self.new)
                    for i in range(1,min(min(cn,nn),5)+1):  # 5 is the maximum 
                        c = " ".join([self.commited_in_buffer[-j][2] for j in range(1,i+1)][::-1])
                        tail = " ".join(self.new[j-1][2] for j in range(1,i+1))
                        if c == tail:
                            print("removing last",i,"words:",file=self.logfile)
                            for j in range(i):
                                print("\t",self.new.pop(0),file=self.logfile)
                            break

    def flush(self):
        # 返回已提交的块 = 最后两次插入的最长公共前缀。

        commit = []
        while self.new:
            na, nb, nt = self.new[0]

            if len(self.buffer) == 0:
                break

            if nt == self.buffer[0][2]:
                commit.append((na,nb,nt))
                self.last_commited_word = nt
                self.last_commited_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit

    def pop_commited(self, time):
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self):
        return self.buffer

class OnlineASRProcessor:

    SAMPLING_RATE = 16000

    def __init__(self, asr, tokenizer=None, buffer_trimming=("segment", 15), logfile=sys.stderr):
        """asr: WhisperASR object
        asr: WhisperASR 对象
        tokenizer: 目标语言的句子分词器对象。必须具有类似于 MosesTokenizer 的 *split* 方法。如果使用了 "segment" 缓冲区修剪选项，则可以为 None，此时不使用分词器。
        buffer_trimming: 一个由 (选项，秒数) 组成的二元组，其中选项是 "sentence" 或 "segment"，秒数是一个数字。如果缓冲区长度超过 "秒数" 阈值，则对其进行修剪。默认情况下，这是最推荐的选项。
        logfile: 日志存储位置。
        """
        self.asr = asr
        self.tokenizer = tokenizer
        self.logfile = logfile

        self.init()

        self.buffer_trimming_way, self.buffer_trimming_sec = buffer_trimming

    def init(self):
        """在开始或重新启动处理时运行此操作。""" 

        self.audio_buffer = np.array([],dtype=np.float32)
        self.buffer_time_offset = 0

        self.transcript_buffer = HypothesisBuffer(logfile=self.logfile)
        self.commited = []

    def insert_audio_chunk(self, audio):
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def prompt(self):
        """
        返回一个元组：(prompt, context)，其中 "prompt" 是在音频缓冲区的滚动部分中的已提交文本的后缀，其长度为200个字符。
        "context" 是在音频缓冲区内的已提交文本。它会再次进行转录并被跳过。仅出于调试和记录目的返回此值。
        """

        k = max(0,len(self.commited)-1)
        while k > 0 and self.commited[k-1][1] > self.buffer_time_offset:
            k -= 1

        p = self.commited[:k]
        p = [t for _,_,t in p]
        prompt = []
        l = 0
        while p and l < 200:  # 200个字符的提示大小
            x = p.pop(-1)
            l += len(x)+1
            prompt.append(x)
        non_prompt = self.commited[k:]
        return self.asr.sep.join(prompt[::-1]), self.asr.sep.join(t for _,_,t in non_prompt)

    def process_iter(self):
        
        """
        在当前音频缓冲区上运行。
        返回：元组(beg_timestamp, end_timestamp, "text")，或(None, None, "")。
        非空文本是已确认（提交）的部分转录。
        """

        prompt, non_prompt = self.prompt()
        print("PROMPT:", prompt, file=self.logfile)
        print("CONTEXT:", non_prompt, file=self.logfile)
        print(f"transcribing {len(self.audio_buffer)/self.SAMPLING_RATE:2.2f} seconds from {self.buffer_time_offset:2.2f}",file=self.logfile)
        res = self.asr.transcribe(self.audio_buffer, init_prompt=prompt)

        # transform to [(beg,end,"word1"), ...]
        tsw = self.asr.ts_words(res)

        self.transcript_buffer.insert(tsw, self.buffer_time_offset)
        o = self.transcript_buffer.flush()
        self.commited.extend(o)
        print(">>>>COMPLETE NOW:",self.to_flush(o),file=self.logfile,flush=True)
        print("INCOMPLETE:",self.to_flush(self.transcript_buffer.complete()),file=self.logfile,flush=True)

        # 存在新确认的文本

        if o and self.buffer_trimming_way == "sentence":  # 切割完成的句子
            if len(self.audio_buffer)/self.SAMPLING_RATE > self.buffer_trimming_sec:  # 超过此长度
                self.chunk_completed_sentence()

        if self.buffer_trimming_way == "segment":
            s = self.buffer_trimming_sec  # 切割超过 s 长度的完成段，
        else:
            s = 30 # 如果音频缓冲区超过 30 秒，对其进行切割

        
        if len(self.audio_buffer)/self.SAMPLING_RATE > s:
            self.chunk_completed_segment(res)

            # 备选方案：在任何单词上
            #l = self.buffer_time_offset + len(self.audio_buffer)/self.SAMPLING_RATE - 10
            # 让我们找到少于 l 的已确认单词
            #k = len(self.commited)-1
            #while k>0 and self.commited[k][1] > l:
            #    k -= 1
            #t = self.commited[k][1] 
            print(f"chunking segment",file=self.logfile)
            #self.chunk_at(t)

        print(f"len of buffer now: {len(self.audio_buffer)/self.SAMPLING_RATE:2.2f}",file=self.logfile)
        return self.to_flush(o)

    def chunk_completed_sentence(self):
        if self.commited == []: return
        print(self.commited,file=self.logfile)
        sents = self.words_to_sentences(self.commited)
        for s in sents:
            print("\t\tSENT:",s,file=self.logfile)
        if len(sents) < 2:
            return
        while len(sents) > 2:
            sents.pop(0)
        # 我们将在此时间戳继续音频处理
        chunk_at = sents[-2][1]

        print(f"--- sentence chunked at {chunk_at:2.2f}",file=self.logfile)
        self.chunk_at(chunk_at)

    def chunk_completed_segment(self, res):
        if self.commited == []: return

        ends = self.asr.segments_end_ts(res)

        t = self.commited[-1][1]

        if len(ends) > 1:

            e = ends[-2]+self.buffer_time_offset
            while len(ends) > 2 and e > t:
                ends.pop(-1)
                e = ends[-2]+self.buffer_time_offset
            if e <= t:
                print(f"--- segment chunked at {e:2.2f}",file=self.logfile)
                self.chunk_at(e)
            else:
                print(f"--- last segment not within commited area",file=self.logfile)
        else:
            print(f"--- not enough segments to chunk",file=self.logfile)





    def chunk_at(self, time):
        """trims the hypothesis and audio buffer at "time"
        """
        self.transcript_buffer.pop_commited(time)
        cut_seconds = time - self.buffer_time_offset
        self.audio_buffer = self.audio_buffer[int(cut_seconds*self.SAMPLING_RATE):]
        self.buffer_time_offset = time

    def words_to_sentences(self, words):
        """Uses self.tokenizer for sentence segmentation of words.
        Returns: [(beg,end,"sentence 1"),...]
        """
        
        cwords = [w for w in words]
        t = " ".join(o[2] for o in cwords)
        s = self.tokenizer.split(t)
        out = []
        while s:
            beg = None
            end = None
            sent = s.pop(0).strip()
            fsent = sent
            while cwords:
                b,e,w = cwords.pop(0)
                w = w.strip()
                if beg is None and sent.startswith(w):
                    beg = b
                elif end is None and sent == w:
                    end = e
                    out.append((beg,end,fsent))
                    break
                sent = sent[len(w):].strip()
        return out

    def finish(self):
        """Flush the incomplete text when the whole processing ends.
        Returns: the same format as self.process_iter()
        """
        o = self.transcript_buffer.complete()
        f = self.to_flush(o)
        print("last, noncommited:",f,file=self.logfile)
        return f


    def to_flush(self, sents, sep=None, offset=0, ):
        # 将时间戳标记的单词或句子连接成一个在一行中刷新的序列
        # sents: [(beg1, end1, "sentence1"), ...] 或空列表 []（如果为空）
        # 返回: 如果为空，则返回 (None, None, "")，否则返回 (beg1, end-of-last-sentence,"concatenation of sentences")
        if sep is None:
            sep = self.asr.sep
        t = sep.join(s[2] for s in sents)
        if len(sents) == 0:
            b = None
            e = None
        else:
            b = offset + sents[0][0]
            e = offset + sents[-1][1]
        return (b,e,t)

WHISPER_LANG_CODES = "af,am,ar,as,az,ba,be,bg,bn,bo,br,bs,ca,cs,cy,da,de,el,en,es,et,eu,fa,fi,fo,fr,gl,gu,ha,haw,he,hi,hr,ht,hu,hy,id,is,it,ja,jw,ka,kk,km,kn,ko,la,lb,ln,lo,lt,lv,mg,mi,mk,ml,mn,mr,ms,mt,my,ne,nl,nn,no,oc,pa,pl,ps,pt,ro,ru,sa,sd,si,sk,sl,sn,so,sq,sr,su,sv,sw,ta,te,tg,th,tk,tl,tr,tt,uk,ur,uz,vi,yi,yo,zh".split(",")

def create_tokenizer(lan):
    """returns an object that has split function that works like the one of MosesTokenizer"""

    assert lan in WHISPER_LANG_CODES, "language must be Whisper's supported lang code: " + " ".join(WHISPER_LANG_CODES)

    if lan == "uk":
        import tokenize_uk
        class UkrainianTokenizer:
            def split(self, text):
                return tokenize_uk.tokenize_sents(text)
        return UkrainianTokenizer()

    # 支持快速的 MosesTokenizer
    if lan in "as bn ca cs de el en es et fi fr ga gu hi hu is it kn lt lv ml mni mr nl or pa pl pt ro ru sk sl sv ta te yue zh".split():
        from mosestokenizer import MosesTokenizer
        return MosesTokenizer(lan)

    # 以下语言在 Whisper 中支持，但不在 wtpsplit 中：
    if lan in "as ba bo br bs fo haw hr ht jw lb ln lo mi nn oc sa sd sn so su sw tk tl tt".split():
        print(f"{lan} 代码不受 wtpsplit 支持。将使用 None lang_code 选项。", file=sys.stderr)
        lan = None

    from wtpsplit import WtP
    # 在第一次使用时从 huggingface 下载模型
    wtp = WtP("wtp-canine-s-12l-no-adapters")
    class WtPtok:
        def split(self, sent):
            return wtp.split(sent, lang_code=lan)
    return WtPtok()


def add_shared_args(parser):
    """为模拟 (此入口点) 和服务器添加共享参数
    parser: argparse.ArgumentParser 对象
    """
    parser.add_argument('--min-chunk-size', type=float, default=1.0, help='最小音频块大小（秒）。它等待最多这么长的时间来进行处理。如果处理时间较短，则等待，否则处理到此时收到的整个段。')
    parser.add_argument('--model', type=str, default='large-v2', choices="tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large-v3,large".split(","),help="要使用的 Whisper 模型的名称大小（默认值：large-v2）。如果模型缓存目录中不存在，则会自动从模型存储库下载模型。")
    parser.add_argument('--model_cache_dir', type=str, default=None, help="覆盖默认模型缓存目录，其中从存储库下载的模型保存")
    parser.add_argument('--model_dir', type=str, default=None, help="保存 Whisper model.bin 和其他文件的目录。此选项将覆盖 --model 和 --model_cache_dir 参数。")
    parser.add_argument('--lan', '--language', type=str, default='auto', help="源语言代码，例如 en、de、cs，或 'auto' 用于语言检测。")
    parser.add_argument('--task', type=str, default='transcribe', choices=["transcribe","translate"],help="转录或翻译。")
    parser.add_argument('--backend', type=str, default="faster-whisper", choices=["faster-whisper", "whisper_timestamped", "openai-api"],help='仅加载此 Whisper 处理的后端。')
    parser.add_argument('--vad', action="store_true", default=False, help='使用 VAD = 声音活动检测，默认参数。')
    parser.add_argument('--buffer_trimming', type=str, default="segment", choices=["sentence", "segment"],help='缓冲区修剪策略 -- 修剪标记有标点符号的已完成句子，并由句子分段器检测到的句子，或 Whisper 返回的已完成段。对于 "sentence" 选项，必须安装句子分段器。')
    parser.add_argument('--buffer_trimming_sec', type=float, default=15, help='缓冲区修剪长度阈值（秒）。如果缓冲区长度较长，则会触发修剪句子/段。')

def asr_factory(args, logfile=sys.stderr):
    """
    根据指定的后端和参数创建和配置 ASR 实例。
    """
    backend = args.backend
    if backend == "openai-api":
        print("使用 OpenAI API。", file=logfile)
        asr = OpenaiApiASR(lan=args.lan)
    else:
        if backend == "faster-whisper":
            asr_cls = FasterWhisperASR
        else:
            asr_cls = WhisperTimestampedASR

        # 仅适用于 FasterWhisperASR 和 WhisperTimestampedASR
        size = args.model
        t = time.time()
        print(f"正在加载 Whisper {size} 模型用于 {args.lan}...", file=logfile, end=" ", flush=True)
        asr = asr_cls(modelsize=size, lan=args.lan, cache_dir=args.model_cache_dir, model_dir=args.model_dir)
        e = time.time()
        print(f"完成。耗时 {round(e-t,2)} 秒。", file=logfile)

    # 应用常见配置
    if getattr(args, 'vad', False):  # 检查是否存在 VAD 参数并且为 True
        print("设置 VAD 过滤器", file=logfile)
        asr.use_vad()

    return asr

## main:
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_path', type=str, help="16kHz 单声道 wav 文件的文件名，用于模拟实时流.")
    add_shared_args(parser)
    parser.add_argument('--start_at', type=float, default=0.0, help='从这个时间开始处理音频.')
    parser.add_argument('--offline', action="store_true", default=False, help='离线模式.')
    parser.add_argument('--comp_unaware', action="store_true", default=False, help='不考虑计算能力的模拟.')
    
    args = parser.parse_args()

    # 重设以将 stderr 存储到不同的文件流中，例如 open(os.devnull,"w")
    logfile = sys.stderr

    if args.offline and args.comp_unaware:
        print("离线和不考虑计算能力的选项中只能选择一个或零个，不能同时选择两个。退出。", file=logfile)
        sys.exit(1)

    audio_path = args.audio_path

    SAMPLING_RATE = 16000
    duration = len(load_audio(audio_path))/SAMPLING_RATE
    print("音频持续时间为：%2.2f 秒" % duration, file=logfile)

    asr = asr_factory(args, logfile=logfile)
    language = args.lan
    if args.task == "translate":
        asr.set_translate_task()
        tgt_language = "en"  # Whisper 翻译成英语
    else:
        tgt_language = language  # Whisper 在该语言中进行转录

    
    min_chunk = args.min_chunk_size
    if args.buffer_trimming == "sentence":
        tokenizer = create_tokenizer(tgt_language)
    else:
        tokenizer = None
    online = OnlineASRProcessor(asr,tokenizer,logfile=logfile,buffer_trimming=(args.buffer_trimming, args.buffer_trimming_sec))


    # 在启动计时器之前将音频加载到 LRU 缓存中
    a = load_audio_chunk(audio_path,0,1)

    # 热身 ASR，因为第一次转录比其他时间要花费更多时间
    asr.transcribe(a)

    beg = args.start_at
    start = time.time()-beg

    def output_transcript(o, now=None):
        # stdout 输出格式如下：
        # 4186.3606 0 1720 Takhle to je
        # - 前三个单词是：
        #    - 从处理开始到现在的发射时间，以毫秒为单位
        #    - Whisper 模型估计的文本段的起始和结束时间戳。时间戳不准确，但仍然有用
        # - 后续单词：段的转录
        if now is None:
            now = time.time()-start
        if o[0] is not None:
            print("%1.4f %1.0f %1.0f %s" % (now*1000, o[0]*1000,o[1]*1000,o[2]),file=logfile,flush=True)
            print("%1.4f %1.0f %1.0f %s" % (now*1000, o[0]*1000,o[1]*1000,o[2]),flush=True)
        else:
            print(o,file=logfile,flush=True)

    if args.offline: ## 离线模式处理（用于测试/调试）
        a = load_audio(audio_path)
        online.insert_audio_chunk(a)
        try:
            o = online.process_iter()
        except AssertionError:
            print("断言错误",file=logfile)
            pass
        else:
            output_transcript(o)
        now = None
    elif args.comp_unaware:  # 不考虑计算能力模式 
        end = beg + min_chunk
        while True:
            a = load_audio_chunk(audio_path,beg,end)
            online.insert_audio_chunk(a)
            try:
                o = online.process_iter()
            except AssertionError:
                print("断言错误",file=logfile)
                pass
            else:
                output_transcript(o, now=end)

            print(f"## 最后处理时间 {end:.2f} 秒",file=logfile,flush=True)

            if end >= duration:
                break
            
            beg = end
            
            if end + min_chunk > duration:
                end = duration
            else:
                end += min_chunk
        now = duration

    else: # 在线 = 同时模式
        end = 0
        while True:
            now = time.time() - start
            if now < end+min_chunk:
                time.sleep(min_chunk+end-now)
            end = time.time() - start
            a = load_audio_chunk(audio_path,beg,end)
            beg = end
            online.insert_audio_chunk(a)

            try:
                o = online.process_iter()
            except AssertionError:
                print("断言错误",file=logfile)
                pass
            else:
                output_transcript(o)
            now = time.time() - start
            print(f"## 最后处理时间 {end:.2f} 秒，当前时间为 {now:.2f} 秒，延迟为 {now-end:.2f} 秒",file=logfile,flush=True)

            if end >= duration:
                break
        now = None

    o = online.finish()
    output_transcript(o, now=now)
