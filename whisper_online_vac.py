from whisper_online import *
from voice_activity_controller import *
import soundfile
import io

SAMPLING_RATE = 16000

class VACOnlineASRProcessor(OnlineASRProcessor):

    def __init__(self, online_chunk_size, *a, **kw):
        self.online_chunk_size = online_chunk_size

        self.online = OnlineASRProcessor(*a, **kw)
        self.vac = VoiceActivityController(use_vad_result = False)

        self.logfile = self.online.logfile

        self.init()

    def init(self):
        self.online.init()
        self.vac.reset_states()
        self.current_online_chunk_buffer_size = 0
        self.is_currently_final = False


    def insert_audio_chunk(self, audio):
        r = self.vac.detect_speech_iter(audio,audio_in_int16=False)
        audio, is_final = r
        print(is_final)
        self.is_currently_final = is_final
        self.online.insert_audio_chunk(audio)
        self.current_online_chunk_buffer_size += len(audio)

    def process_iter(self):
        if self.is_currently_final:
            return self.finish()
        elif self.current_online_chunk_buffer_size > SAMPLING_RATE*self.online_chunk_size:
            self.current_online_chunk_buffer_size = 0
            ret = self.online.process_iter()
            return ret
        else:
            print("no online update, only VAD", file=self.logfile)
            return (None, None, "")

    def finish(self):
        ret = self.online.finish()
        self.online.init()
        self.current_online_chunk_buffer_size = 0
        return ret




if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_path', type=str, help="Filename of 16kHz mono channel wav, on which live streaming is simulated.")
    add_shared_args(parser)
    parser.add_argument('--start_at', type=float, default=0.0, help='Start processing audio at this time.')
    parser.add_argument('--offline', action="store_true", default=False, help='Offline mode.')
    parser.add_argument('--comp_unaware', action="store_true", default=False, help='Computationally unaware simulation.')
    parser.add_argument('--vac-chunk-size', type=float, default=0.04, help='VAC sample size in seconds.') 
    args = parser.parse_args()

    # reset to store stderr to different file stream, e.g. open(os.devnull,"w")
    logfile = sys.stderr

    if args.offline and args.comp_unaware:
        print("No or one option from --offline and --comp_unaware are available, not both. Exiting.",file=logfile)
        sys.exit(1)

    audio_path = args.audio_path

    SAMPLING_RATE = 16000
    duration = len(load_audio(audio_path))/SAMPLING_RATE
    print("Audio duration is: %2.2f seconds" % duration, file=logfile)

    size = args.model
    language = args.lan

    t = time.time()
    print(f"Loading Whisper {size} model for {language}...",file=logfile,end=" ",flush=True)

    if args.backend == "faster-whisper":
        asr_cls = FasterWhisperASR
    else:
        asr_cls = WhisperTimestampedASR

    asr = asr_cls(modelsize=size, lan=language, cache_dir=args.model_cache_dir, model_dir=args.model_dir)

    if args.task == "translate":
        asr.set_translate_task()
        tgt_language = "en"  # Whisper translates into English
    else:
        tgt_language = language  # Whisper transcribes in this language


    e = time.time()
    print(f"done. It took {round(e-t,2)} seconds.",file=logfile)

    if args.vad:
        print("setting VAD filter",file=logfile)
        asr.use_vad()

    
    min_chunk = args.vac_chunk_size
    if args.buffer_trimming == "sentence":
        tokenizer = create_tokenizer(tgt_language)
    else:
        tokenizer = None
    online = VACOnlineASRProcessor(args.min_chunk_size, asr,tokenizer,logfile=logfile,buffer_trimming=(args.buffer_trimming, args.buffer_trimming_sec))


    # load the audio into the LRU cache before we start the timer
    a = load_audio_chunk(audio_path,0,1)

    # warm up the ASR, because the very first transcribe takes much more time than the other
    asr.transcribe(a)

    beg = args.start_at
    start = time.time()-beg

    def output_transcript(o, now=None):
        # output format in stdout is like:
        # 4186.3606 0 1720 Takhle to je
        # - the first three words are:
        #    - emission time from beginning of processing, in milliseconds
        #    - beg and end timestamp of the text segment, as estimated by Whisper model. The timestamps are not accurate, but they're useful anyway
        # - the next words: segment transcript
        if now is None:
            now = time.time()-start
        if o[0] is not None:
            print("%1.4f %1.0f %1.0f %s" % (now*1000, o[0]*1000,o[1]*1000,o[2]),file=logfile,flush=True)
            print("%1.4f %1.0f %1.0f %s" % (now*1000, o[0]*1000,o[1]*1000,o[2]),flush=True)
        else:
            print(o,file=logfile,flush=True)

    if args.offline: ## offline mode processing (for testing/debugging)
        a = load_audio(audio_path)
        online.insert_audio_chunk(a)
        try:
            o = online.process_iter()
        except AssertionError:
            print("assertion error",file=logfile)
            pass
        else:
            output_transcript(o)
        now = None
    elif args.comp_unaware:  # computational unaware mode 
        end = beg + min_chunk
        while True:
            a = load_audio_chunk(audio_path,beg,end)
            online.insert_audio_chunk(a)
            try:
                o = online.process_iter()
            except AssertionError:
                print("assertion error",file=logfile)
                pass
            else:
                output_transcript(o, now=end)

            print(f"## last processed {end:.2f}s",file=logfile,flush=True)

            if end >= duration:
                break
            
            beg = end
            
            if end + min_chunk > duration:
                end = duration
            else:
                end += min_chunk
        now = duration

    else: # online = simultaneous mode
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
                print("assertion error",file=logfile)
                pass
            else:
                output_transcript(o)
            now = time.time() - start
            print(f"## last processed {end:.2f} s, now is {now:.2f}, the latency is {now-end:.2f}",file=logfile,flush=True)

            if end >= duration:
                break
        now = None

    o = online.finish()
    output_transcript(o, now=now)
