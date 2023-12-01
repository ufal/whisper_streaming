import torch
import numpy as np
# import sounddevice as sd
import torch
import numpy as np


class VoiceActivityController:
    def __init__(
            self, 
            sampling_rate = 16000,
            second_ofSilence = 0.5,
            second_ofSpeech = 0.25,
            second_ofMinRecording = 10,
            use_vad_result = True,
            activity_detected_callback=None,
        ):
        self.activity_detected_callback=activity_detected_callback
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad'
        )
        (self.get_speech_timestamps,
        save_audio,
        read_audio,
        VADIterator,
        collect_chunks) = self.utils

        self.sampling_rate = sampling_rate  
        self.silence_limit = second_ofSilence * self.sampling_rate 
        self.speech_limit = second_ofSpeech *self.sampling_rate 
        self.MIN_RECORDING_LENGTH =  second_ofMinRecording * self.sampling_rate 

        self.use_vad_result = use_vad_result
        self.vad_iterator = VADIterator(
            model =self.model,
            threshold = 0.3,
            sampling_rate= 16000,
            min_silence_duration_ms = 500, #100
            speech_pad_ms = 400 #30
        )
        self.last_marked_chunk = None
        
    
    def int2float(self, sound):
        abs_max = np.abs(sound).max()
        sound = sound.astype('float32')
        if abs_max > 0:
            sound *= 1/32768
        sound = sound.squeeze()  # depends on the use case
        return sound

    def apply_vad(self, audio):
        audio_float32 = self.int2float(audio)
        chunk = self.vad_iterator(audio_float32, return_seconds=False)

        if chunk is not None:        
            if "start" in chunk:
                start = chunk["start"]
                self.last_marked_chunk = chunk
                return audio[start:] if self.use_vad_result else audio, (len(audio) - start), 0
            
            if "end" in chunk:
                #todo: pending get the padding from the next chunk
                end = chunk["end"] if chunk["end"] < len(audio) else len(audio)
                self.last_marked_chunk = chunk
                return audio[:end] if self.use_vad_result else audio, end ,len(audio) - end

        if self.last_marked_chunk is not None:
            if "start" in self.last_marked_chunk:
                return audio, len(audio)  ,0

            if "end" in self.last_marked_chunk:
                return  np.array([], dtype=np.float16) if self.use_vad_result else audio, 0 ,len(audio) 

        return  np.array([], dtype=np.float16) if self.use_vad_result else audio, 0 , 0 



    def detect_user_speech(self, audio_stream, audio_in_int16 = False):
        silence_len= 0
        speech_len = 0

        for data in audio_stream:  # replace with your condition of choice
            # if isinstance(data, EndOfTransmission):
            #     raise EndOfTransmission("End of transmission detected")
            
            
            audio_block = np.frombuffer(data, dtype=np.int16) if not audio_in_int16 else data
            wav = audio_block
            

            is_final = False
            voice_audio, speech_in_wav, last_silent_duration_in_wav = self.apply_vad(wav)
            # print(f'----r> speech_in_wav: {speech_in_wav} last_silent_duration_in_wav: {last_silent_duration_in_wav}')

            if speech_in_wav > 0 :
                silence_len= 0                
                speech_len += speech_in_wav
                if self.activity_detected_callback is not None:
                    self.activity_detected_callback()

            silence_len = silence_len + last_silent_duration_in_wav
            if silence_len>= self.silence_limit and speech_len >= self.speech_limit:
                is_final = True
                silence_len= 0
                speech_len = 0
            

            yield voice_audio.tobytes(), is_final







