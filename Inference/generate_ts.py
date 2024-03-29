import math
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torchaudio.models import Hypothesis, RNNTBeamSearch

from syllabe_vocab import VOCAB_TOKENS, syllbe_vocab_size, BLANK_ID

class SampleConfig:
    def __init__(self):
        self.resamplers = {}
        self.mel_spectrograms = {}
        self.resample_rate = 16000
        self.lowpass_filter_width = 64
        self.rolloff = 0.9475937167399596
        self.resampling_method = "kaiser_window"
        self.beta = 14.769656459379492
        self.n_fft = 1024
        self.n_mels = 80

class KinspeakEmformerRNNT(torch.nn.Module):
    def __init__(self, target_vocab_size,
                 target_blank_id):
        super(KinspeakEmformerRNNT, self).__init__()
        self.target_vocab_size = target_vocab_size
        self.target_blank_id = target_blank_id
        self.rnnt = torchaudio.models.emformer_rnnt_base(self.target_vocab_size)
        self.loss = torchaudio.transforms.RNNTLoss(reduction="sum", clamp=1.0)

    def forward(self, log_mel_spectrograms: torch.Tensor, #log_mel_spectrograms: (N,F,L)
                log_mel_spectrogram_lengths: List[int],
                target_syllabe_ids:torch.Tensor, target_syllabe_id_lengths:List[int],
                target_syllabe_ids_with_eos=True,
                target_syllabe_gpt_output = None):
        sources = log_mel_spectrograms.transpose(1,2)
        source_lengths = torch.tensor(log_mel_spectrogram_lengths).to(sources.device, dtype=torch.int32)
        target_syllabe_ids = target_syllabe_ids.to(dtype=torch.int32)
        targets = target_syllabe_ids.split(target_syllabe_id_lengths)
        targets = pad_sequence(targets, batch_first=True)
        target_lengths = torch.tensor(target_syllabe_id_lengths).to(targets.device, dtype=torch.int32)
        prepended_targets = targets.new_empty([targets.size(0), targets.size(1) + 1])
        prepended_targets[:, 1:] = targets
        prepended_targets[:, 0] = self.target_blank_id
        prepended_target_lengths = target_lengths + 1
        (output, src_lengths, _, __) = self.rnnt(sources, source_lengths, prepended_targets, prepended_target_lengths)
        # print('\noutput:', output.shape)
        # print('targets:',targets.shape)
        # print('src_lengths:',src_lengths, '==>', src_lengths.tolist(), 'MAX:', max(src_lengths.tolist()))
        # print('source_lengths:',source_lengths, '==>', source_lengths.tolist(), 'MAX:', max(source_lengths.tolist()))
        # print('target_lengths:',target_lengths,flush=True)
        loss = torchaudio.functional.rnnt_loss(output, targets, src_lengths-1, target_lengths, blank=self.target_blank_id, reduction = 'mean', clamp=1.0)
        # print('loss:', loss,'\n',flush=True)
        return loss

def get_hypo_tokens(hypo: Hypothesis) -> List[int]:
    return hypo[0]


def get_hypo_score(hypo: Hypothesis) -> float:
    return hypo[3]


def to_string(input: List[int], tgt_dict: List[str], separator: str = "",) -> str:
    # torchscript dislikes sets
    extra_symbols_to_ignore: Dict[int, int] = {}
    extra_symbols_to_ignore[0] = 1 # <pad>
    extra_symbols_to_ignore[1] = 1 # <unk>
    extra_symbols_to_ignore[2] = 1 # <mask>
    extra_symbols_to_ignore[3] = 1 # <s>
    extra_symbols_to_ignore[4] = 1 # </s>
    extra_symbols_to_ignore[6] = 1 # ~

    # it also dislikes comprehensions with conditionals
    filtered_idx: List[int] = []
    for idx in input:
        if idx not in extra_symbols_to_ignore:
            filtered_idx.append(idx)

    return separator.join([tgt_dict[idx] for idx in filtered_idx]).replace('|', ' ')


def post_process_hypos(
    hypos: List[Hypothesis], tgt_dict: List[str],
) -> List[Tuple[str, List[float], List[int]]]:
    post_process_remove_list = [0,1,2,3,4,6]
    hypos_str: List[str] = []
    for h in hypos:
        filtered_tokens: List[int] = []
        for token_index in get_hypo_tokens(h)[1:]:
            if token_index not in post_process_remove_list:
                filtered_tokens.append(token_index)
        string = to_string(filtered_tokens, tgt_dict)
        hypos_str.append(string)

    hypos_ids = [get_hypo_tokens(h)[1:] for h in hypos]
    hypos_score = [[math.exp(get_hypo_score(h))] for h in hypos]

    nbest_batch = list(zip(hypos_str, hypos_score, hypos_ids))

    return nbest_batch


def _piecewise_linear_log(x):
    x[x > math.e] = torch.log(x[x > math.e])
    x[x <= math.e] = x[x <= math.e] / math.e
    return x


class ModelWrapper(torch.nn.Module):
    def __init__(self, tgt_dict: List[str]):
        super().__init__()
        # self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, n_mels=80, hop_length=160)
        cfg = SampleConfig()
        win_length = cfg.resample_rate * 25 // 1000  # 25ms
        hop_length = cfg.resample_rate * 10 // 1000  # 10ms
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=cfg.resample_rate, n_fft=cfg.n_fft,
                                           win_length=win_length,
                                           hop_length=hop_length, center=True, pad_mode="reflect", power=2.0,
                                           norm="slaney", onesided=True, n_mels=cfg.n_mels,
                                           mel_scale="htk", )

        rnnt = torchaudio.models.emformer_rnnt_base(syllbe_vocab_size())
        model = KinspeakEmformerRNNT(syllbe_vocab_size(), BLANK_ID)
        state_dict = torch.load('/home/nzeyi/KINLP/data/kinspeak_asr_syllabe_emformer_rnnt_base_2024-03-07.pt_best_valid_loss.pt', map_location='cpu')
        model.load_state_dict(state_dict['model_state_dict'])
        rnnt.load_state_dict(model.rnnt.state_dict())
        del state_dict
        del model

        self.decoder = RNNTBeamSearch(rnnt, BLANK_ID)
        self.tgt_dict = tgt_dict

    def forward(
        self, input: torch.Tensor, prev_hypo: Optional[List[Hypothesis]], prev_state: Optional[List[List[torch.Tensor]]]
    ) -> Tuple[str, Optional[List[Hypothesis]], Optional[List[List[torch.Tensor]]]]:
        log_eps = 1e-36
        spectrogram = self.mel_spectrogram(input).transpose(1, 0)
        features = torch.log(spectrogram + log_eps)#.unsqueeze(0)[:, :-1]

        length = torch.tensor([features.shape[1]])

        hypotheses, state = self.decoder.infer(features, length, 10, state=prev_state, hypothesis=prev_hypo)
        transcript = post_process_hypos(hypotheses[:1], self.tgt_dict)[0][0]
        return transcript, hypotheses, state


wrapper = ModelWrapper(VOCAB_TOKENS)
wrapper = torch.jit.script(wrapper)
wrapper.save("scripted_wrapper_tuple.pt")
