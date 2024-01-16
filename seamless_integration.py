#!/usr/bin/env python3

import sys
import numpy as np

# code extracted from https://github.com/facebookresearch/seamless_communication/blob/main/Seamless_Tutorial.ipynb :

from simuleval.data.segments import SpeechSegment, EmptySegment
from simuleval.utils.arguments import cli_argument_list
from simuleval import options


from typing import Union, List
from simuleval.data.segments import Segment, TextSegment
from simuleval.agents.pipeline import TreeAgentPipeline
from simuleval.agents.states import AgentStates


SAMPLE_RATE = 16000

def reset_states(system, states):
    if isinstance(system, TreeAgentPipeline):
        states_iter = states.values()
    else:
        states_iter = states
    for state in states_iter:
        state.reset()


def get_states_root(system, states) -> AgentStates:
    if isinstance(system, TreeAgentPipeline):
        # self.states is a dict
        return states[system.source_module]
    else:
        # self.states is a list
        return system.states[0]


def build_streaming_system(model_configs, agent_class):
    parser = options.general_parser()
    parser.add_argument("-f", "--f", help="a dummy argument to fool ipython", default="1")

    agent_class.add_args(parser)
    args, _ = parser.parse_known_args(cli_argument_list(model_configs))
    system = agent_class.from_args(args)
    return system

class OutputSegments:
    def __init__(self, segments: Union[List[Segment], Segment]):
        if isinstance(segments, Segment):
            segments = [segments]
        self.segments: List[Segment] = [s for s in segments]

    @property
    def is_empty(self):
        return all(segment.is_empty for segment in self.segments)

    @property
    def finished(self):
        return all(segment.finished for segment in self.segments)


######################
# fixing DetokenizerAgent -- it strips output segment.content last space, but sometimes a word is split into more segments. Simple joining with spaces would be wrong.
from seamless_communication.streaming.agents.detokenizer import DetokenizerAgent
from seamless_communication.streaming.agents.offline_w2v_bert_encoder import (
    OfflineWav2VecBertEncoderAgent,
)
from seamless_communication.streaming.agents.online_feature_extractor import (
    OnlineFeatureExtractorAgent,
)
from seamless_communication.streaming.agents.online_text_decoder import (
    MMASpeechToTextDecoderAgent,
)
from seamless_communication.streaming.agents.silero_vad import SileroVADAgent
from seamless_communication.streaming.agents.unity_pipeline import UnitYAgentPipeline
class FixDetokenizerAgent(DetokenizerAgent):
    def decode(self, x: str) -> str:
        return x.replace(" ", "").replace("\u2581", " ")

class FixSeamlessStreamingS2TVADAgent(UnitYAgentPipeline):
    pipeline = [
        SileroVADAgent,
        OnlineFeatureExtractorAgent,
        OfflineWav2VecBertEncoderAgent,
        MMASpeechToTextDecoderAgent,
        FixDetokenizerAgent,
    ]
##################################

#class SeamlessProcessor(OnlineASRProcessorBase):  # TODO: there should be a common base class
class SeamlessProcessor:
    def __init__(self, tgt_lan, logfile=sys.stderr):
        self.logfile = logfile

        agent_class = FixSeamlessStreamingS2TVADAgent

        model_configs = dict(
            source_segment_size=320,
            device="cuda:0",
            dtype="fp16",
            min_starting_wait_w2vbert=192,
            decision_threshold=0.5,
            min_unit_chunk_size=50,
            no_early_stop=True,
            max_len_a=0,
            max_len_b=100,
            task="s2tt",
            tgt_lang=tgt_lan,
            block_ngrams=True,
            detokenize_only=True,
        )
        self.tgt_lan = tgt_lan

        self.system = build_streaming_system(model_configs, agent_class)

        self.system_states = self.system.build_states()

        self.init()

    def init(self):
        reset_states(self.system, self.system_states)
        self.audio_buffer = np.array([],dtype=np.float32)
        self.beg, self.end = 0, 0

    def insert_audio_chunk(self, audio):
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def process_segment(self, input_segment):
        output_segments = OutputSegments(self.system.pushpop(input_segment, self.system_states))
        out = []
        for segment in output_segments.segments:
            if not segment.is_empty:
                out.append(segment.content)
        if output_segments.finished:
            print("End of VAD segment",file=self.logfile)
            reset_states(self.system, self.system_states)
        if out:
            b = self.beg
            self.beg = self.end
            o = "".join(out)
            return (b, self.end, "".join(out))
        return (None, None, "")


    def process_iter(self):
        input_segment = SpeechSegment(
                content=self.audio_buffer,
                sample_rate=SAMPLE_RATE,
                finished=False,
        )
        self.audio_buffer = np.array([],dtype=np.float32)
        input_segment.tgt_lang = self.tgt_lan
        self.end += (len(input_segment.content)/SAMPLE_RATE)
        return self.process_segment(input_segment)

    def finish(self):
        segment = EmptySegment(
                finished=True,
            )
        return self.process_segment(segment)
