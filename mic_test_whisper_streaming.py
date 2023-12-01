from microphone_stream import MicrophoneStream
from voice_activity_controller import VoiceActivityController
from whisper_online import *
import numpy as np
import librosa  
import io
import soundfile
import sys


SAMPLING_RATE = 16000
model = "large-v2"
src_lan = "en"  # source language
tgt_lan = "en"  # target language  -- same as source for ASR, "en" if translate task is used
use_vad_result = True
min_sample_length = 1 * SAMPLING_RATE



asr = FasterWhisperASR(src_lan, model)  # loads and wraps Whisper model
tokenizer = create_tokenizer(tgt_lan)  # sentence segmenter for the target language
online = OnlineASRProcessor(asr, tokenizer)  # create processing object

microphone_stream = MicrophoneStream() 
vad = VoiceActivityController(use_vad_result = use_vad_result)

complete_text = ''
final_processing_pending = False
out = []
out_len = 0
for iter in vad.detect_user_speech(microphone_stream):   # processing loop:
    raw_bytes=  iter[0]
    is_final =  iter[1]

    if  raw_bytes:
        sf = soundfile.SoundFile(io.BytesIO(raw_bytes), channels=1,endian="LITTLE",samplerate=SAMPLING_RATE, subtype="PCM_16",format="RAW")
        audio, _ = librosa.load(sf,sr=SAMPLING_RATE)
        out.append(audio)
        out_len += len(audio)

    
    if (is_final or out_len >= min_sample_length) and out_len>0:
        a = np.concatenate(out)
        online.insert_audio_chunk(a)    
        
    if out_len > min_sample_length:
        o = online.process_iter()
        print('-----'*10)
        complete_text = complete_text + o[2]
        print('PARTIAL - '+ complete_text) # do something with current partial output
        print('-----'*10)     
        out = []
        out_len = 0   

    if is_final:
        o = online.finish()
        online.init()   
        # final_processing_pending = False         
        print('-----'*10)
        complete_text = complete_text + o[2]
        print('FINAL - '+ complete_text) # do something with current partial output
        print('-----'*10)   
        out = []
        out_len = 0    
        






