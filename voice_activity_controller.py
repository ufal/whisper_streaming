import torch
import numpy as np

class VoiceActivityController:
    def __init__(
            self, 
            sampling_rate = 16000,
            min_silence_to_final_ms = 500,
            min_speech_to_final_ms = 100,
            min_silence_duration_ms = 100,
            use_vad_result = True,
#            activity_detected_callback=None,
            threshold =0.3
        ):
#        self.activity_detected_callback=activity_detected_callback
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad'
        )
        # (self.get_speech_timestamps,
        # save_audio,
        # read_audio,
        # VADIterator,
        # collect_chunks) = self.utils

        self.sampling_rate = sampling_rate  
        self.final_silence_limit = min_silence_to_final_ms * self.sampling_rate / 1000 
        self.final_speech_limit = min_speech_to_final_ms *self.sampling_rate / 1000
        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000

        self.use_vad_result = use_vad_result
        self.threshold = threshold
        self.reset_states()

    def reset_states(self):
        self.model.reset_states()
        self.temp_end = 0
        self.current_sample = 0

        self.last_silence_len= 0
        self.speech_len = 0

    def apply_vad(self, audio):
        """
        returns: triple
            (voice_audio,
            speech_in_wav,
            silence_in_wav)

        """
        x = audio
        if not torch.is_tensor(x):
            try:
                x = torch.Tensor(x)
            except:
                raise TypeError("Audio cannot be casted to tensor. Cast it manually")

        speech_prob = self.model(x, self.sampling_rate).item()
        print("speech_prob",speech_prob)
        
        window_size_samples = len(x[0]) if x.dim() == 2 else len(x)
        self.current_sample += window_size_samples 

        if speech_prob >= self.threshold:  # speech is detected
            self.temp_end = 0
            return audio, window_size_samples, 0

        else:  # silence detected, counting w
            if not self.temp_end:
                self.temp_end = self.current_sample

            if self.current_sample - self.temp_end < self.min_silence_samples:
                return audio, 0, window_size_samples
            else:
                return np.array([], dtype=np.float16) if self.use_vad_result else audio, 0, window_size_samples


    def detect_speech_iter(self, data, audio_in_int16 = False):
        audio_block = data
        wav = audio_block

        is_final = False
        voice_audio, speech_in_wav, last_silent_in_wav = self.apply_vad(wav)
        print("speech, last silence",speech_in_wav, last_silent_in_wav)


        if speech_in_wav > 0 :
            self.last_silence_len= 0                
            self.speech_len += speech_in_wav
#            if self.activity_detected_callback is not None:
#                self.activity_detected_callback()

        self.last_silence_len +=  last_silent_in_wav
        print("self.last_silence_len",self.last_silence_len, self.final_silence_limit,self.last_silence_len>= self.final_silence_limit)
        print("self.speech_len, final_speech_limit",self.speech_len , self.final_speech_limit,self.speech_len >= self.final_speech_limit)
        if self.last_silence_len>= self.final_silence_limit and self.speech_len >= self.final_speech_limit:
            for i in range(10): print("TADY!!!")

            is_final = True
            self.last_silence_len= 0
            self.speech_len = 0                

        return voice_audio, is_final

    def detect_user_speech(self, audio_stream, audio_in_int16 = False):
        self.last_silence_len= 0
        self.speech_len = 0

        for data in audio_stream:  # replace with your condition of choice
            yield self.detect_speech_iter(data, audio_in_int16)
