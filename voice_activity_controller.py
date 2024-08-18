import torch
from silero_vad import VADIterator
import time

class VoiceActivityController:
    SAMPLING_RATE = 16000
    def __init__(self):
        self.model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad'
        )
        # we use the default options: 500ms silence, etc.
        self.iterator = VADIterator(self.model)

    def reset(self):
        self.iterator.reset_states()

    def __call__(self, audio):
        '''
        audio: audio chunk in the current np.array format
        returns: 
        - { 'start': time_frame } ... when voice start was detected. time_frame is number of frame (can be converted to seconds)
        - { 'end': time_frame }   ... when voice end is detected
        - None                    ... when no change detected by current chunk 
        '''
        x = audio
#        if not torch.is_tensor(x):
#            try:
#                x = torch.Tensor(x)
#            except:
#                raise TypeError("Audio cannot be casted to tensor. Cast it manually")
        t = time.time()
        a = self.iterator(x)
        print("VAD took ",time.time()-t,"seconds")
        return a
