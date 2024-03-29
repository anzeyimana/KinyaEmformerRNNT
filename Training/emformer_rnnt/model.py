from typing import List

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence


class KinspeakEmformerRNNT(torch.nn.Module):
    def __init__(self, target_vocab_size,
                 target_blank_id):
        super(KinspeakEmformerRNNT, self).__init__()
        self.target_vocab_size = target_vocab_size
        self.target_blank_id = target_blank_id
        self.rnnt = torchaudio.models.emformer_rnnt_base(self.target_vocab_size)
        self.loss = torchaudio.transforms.RNNTLoss(reduction="sum", clamp=1.0)

    def forward(self, log_mel_spectrograms: torch.Tensor, # (N,F,L)
                log_mel_spectrogram_lengths: List[int],
                target_syllabe_ids:torch.Tensor, target_syllabe_id_lengths:List[int]):
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
        loss = torchaudio.functional.rnnt_loss(output, targets, src_lengths-1, target_lengths, blank=self.target_blank_id, reduction = 'mean', clamp=1.0)
        return loss
