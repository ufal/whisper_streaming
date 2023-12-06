from microphone_stream import MicrophoneStream
from voice_activity_controller import VoiceActivityController
from whisper_online import *
import numpy as np
import librosa  
import io
import soundfile
import sys




class SimpleASRProcessor:

    def __init__(self, asr, sampling_rate = 16000):
        """run this when starting or restarting processing"""
        self.audio_buffer = np.array([],dtype=np.float32)
        self.prompt_buffer = ""
        self.asr = asr
        self.sampling_rate = sampling_rate
        self.init_prompt = ''

    def ts_words(self, segments):
        result = ""
        for segment in segments:
            if segment.no_speech_prob > 0.9:
                continue
            for word in segment.words:
                w = word.word
                t = (word.start, word.end, w)
                result +=w
        return result 

    def stream_process(self, vad_result):
        iter_in_phrase = 0
        for chunk, is_final in vad_result:
            iter_in_phrase += 1

            if chunk is not None:
                sf = soundfile.SoundFile(io.BytesIO(chunk), channels=1,endian="LITTLE",samplerate=SAMPLING_RATE, subtype="PCM_16",format="RAW")
                audio, _ = librosa.load(sf,sr=SAMPLING_RATE)
                out = []
                out.append(audio)
                a = np.concatenate(out)
                self.audio_buffer = np.append(self.audio_buffer, a)

            if is_final and len(self.audio_buffer) > 0:
                res = self.asr.transcribe(self.audio_buffer, init_prompt=self.init_prompt)
                tsw = self.ts_words(res)
                
                self.init_prompt = self.init_prompt + tsw
                self.init_prompt  = self.init_prompt [-100:]
                self.audio_buffer.resize(0)
                iter_in_phrase =0
                
                yield True, tsw
            # show progress evry 50 chunks
            elif iter_in_phrase % 50 == 0 and len(self.audio_buffer) > 0:
                res = self.asr.transcribe(self.audio_buffer, init_prompt=self.init_prompt)
                # use custom ts_words
                tsw = self.ts_words(res)
                yield False, tsw
            
        





SAMPLING_RATE = 16000

model = "large-v2"
src_lan = "en"  # source language
tgt_lan = "en"  # target language  -- same as source for ASR, "en" if translate task is used
use_vad_result = True
min_sample_length = 1 * SAMPLING_RATE



vad = VoiceActivityController(use_vad_result = use_vad_result)
asr = FasterWhisperASR(src_lan, "large-v2")  # loads and wraps Whisper model

tokenizer = create_tokenizer(tgt_lan)
online = SimpleASRProcessor(asr)


stream = MicrophoneStream()
stream = vad.detect_user_speech(stream, audio_in_int16 = False) 
stream = online.stream_process(stream)

for isFinal, text in stream:
    if isFinal:
        print( text,  end="\r\n")
    else:
        print( text,  end="\r")
