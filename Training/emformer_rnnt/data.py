import math
import os

from datetime import datetime
import random
from typing import List

import torch
import torchaudio
import torchaudio.transforms as T
import progressbar
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from mutagen.mp3 import MP3

cv_audio_clips_dir = "/home/user/cv-corpus-12.0-2022-12-07/rw/clips/"

SAMPLE_RATE = 16000

audio_min_duration = 2.0
audio_max_duration = 25.0
labels_min_length = 4
labels_max_length = 1024


def time_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def data_read_common_voice(cv_list_file, text_to_id_sequence_fun, bos_id, eos_id):

    f = open(cv_list_file, 'r', encoding="utf-8")
    cv_lines = [line.rstrip('\n') for line in f if len(line.rstrip('\n')) > 0]
    f.close()
    cv_lines = cv_lines[1:]
    cv_data = []
    print(time_now(), 'Reading {} common voice audio files listed in {} ...'.format(len(cv_lines), cv_list_file), flush=True)
    with progressbar.ProgressBar(initial_value=0,
                                 max_value=len(cv_lines),
                                 redirect_stdout=True) as bar:
        for itr,line in enumerate(cv_lines):
            if (itr % 100000) == 0:
                bar.update(itr)
            toks = line.split('\t')
            if len(toks) == 10:
                TEXT = toks[2]
                labels = [bos_id] + text_to_id_sequence_fun(TEXT) + [eos_id]
                audio_file = os.path.join(cv_audio_clips_dir, toks[1])
                tags = (toks[5].strip(), toks[6].strip())
                if (len(labels) >= labels_min_length) and (len(labels) <= labels_max_length):
                    try:
                        secs = MP3(audio_file).info.length
                        if ((secs >= audio_min_duration) and (secs <= audio_max_duration)):
                            cv_data.append((audio_file, secs, TEXT))
                    except:
                        print('Can\'t read file {}'.format(audio_file))
    tot = sum([s for f, s, txt in cv_data])
    hr = int(tot / 3600)
    mn = int(tot / 60) % 60
    sc = int(tot) % 60
    print(time_now(), '==> Read total {} CV files: {}h{}m{}s from {}'.format(len(cv_data),hr,mn,sc,cv_list_file))
    return cv_data

def data_read_common_voice_with_syllabe_vocab(file):
    from syllabe_vocab import text_to_id_sequence, BOS_ID, EOS_ID
    return data_read_common_voice(file, text_to_id_sequence, BOS_ID, EOS_ID)

class KinSpeakDataset(Dataset):
    def __init__(self, data_read_list, include_labels=False,
                 num_shuffle_buckets=100,
                 max_batch_seconds=80,
                 debug=False,
                 batch_amplification_factor = 1.2):
        self.resample_rate = 16000
        self.lowpass_filter_width = 64
        self.rolloff = 0.9475937167399596
        self.resampling_method = "sinc_interp_kaiser"
        self.beta = 14.769656459379492
        self.resamplers = dict()
        self.include_labels = include_labels
        self.num_shuffle_buckets = num_shuffle_buckets
        self.max_batch_seconds = max_batch_seconds
        self.debug=debug
        self.batch_amplification_factor = batch_amplification_factor
        self.data_items = []
        for (file,file_read_fn) in data_read_list:
            self.data_items.extend(file_read_fn(file)) # format: List[(audio_filename,audio_length,labels_seq)]
        self.index = [i for i in range(len(self.data_items))]
        self.index.sort(key=lambda x: self.data_items[x][1], reverse=True)
        if self.debug:
            tot = sum([s for f,s,txt in self.data_items])
            hr = int(tot/3600)
            mn = int(tot/60) % 60
            sc = int(tot) % 60
            print(time_now(), 'Read {} audio files:> Total: {}h{}m{}s ==> {:.0f} -> {:.0f} secs'.format(len(self.data_items), hr, mn, sc, self.data_items[self.index[0]][1], self.data_items[self.index[-1]][1]), flush=True)
        total_length = int(sum([x[1] for x in self.data_items]))+1
        bucket_size = (total_length // self.num_shuffle_buckets) + 1
        self.buckets = []
        start = 0
        num_seconds = 0.0
        for end in range(len(self.index)):
            length = self.data_items[self.index[end]][1]
            if (num_seconds + length) > bucket_size:
                if end > start:
                    self.buckets.append((start, end))
                num_seconds = 0.0
                start = end
            num_seconds += length
        self.shuffle_buckets_and_mark_batches()

    def shuffle_buckets_and_mark_batches(self):
        if self.debug:
            print(time_now(), 'Dataset shuffling ...', flush=True)
        seed_val = datetime.now().microsecond + (13468 * os.getpid())
        seed_val = int(seed_val) % ((2 ** 32) - 1)
        np.random.seed(seed_val)
        random.seed(seed_val)
        torch.random.manual_seed(seed_val)
        # Shuffle within buckets
        for (start, end) in self.buckets:
            copy = self.index[start:end]
            random.shuffle(copy)
            self.index[start:end] = copy
        # Form batches
        new_batches = []
        start = 0
        num_seconds = 0.0
        excl = 0
        for end, idx in enumerate(self.index):
            length = self.data_items[idx][1]
            if (num_seconds + length) > self.max_batch_seconds:
                if end > start:
                    new_batches.append((start, end))
                num_seconds = 0.0
                excl = 0
                start = end
            num_seconds += length
            excl += 1
        random.shuffle(new_batches)
        self.batches = new_batches
        if self.debug:
            print(time_now(), 'Batching DONE: got {} batches; discarded {} examples ({:.1f} seconds)'.format(len(self.batches), excl,
                                                                                         num_seconds), flush=True)
    def __len__(self):
        return int(len(self.batches) * self.batch_amplification_factor) # x 1.z to ensure we will always reach len(batches)-1 to re-shuffle buckets --> Num epochs adjusted to 200 to to reflect this!

    def __getitem__(self, batch_idx):
        batch_idx = batch_idx % len(self.batches)
        (start, end) = self.batches[batch_idx]
        items = []
        # total_samples = 0
        for i in range(start,end):
            idx = self.index[i]
            (audio_filename, audio_length, transcript) = self.data_items[idx]
            waveform, sample_rate = torchaudio.load(audio_filename)
            if (sample_rate != SAMPLE_RATE):
                sr_kwd = f'{sample_rate}'
                if sr_kwd in self.resamplers:
                    resampler = self.resamplers[sr_kwd]
                else:
                    resampler = T.Resample(sample_rate, self.resample_rate,
                                           lowpass_filter_width=self.lowpass_filter_width,
                                           rolloff=self.rolloff, resampling_method=self.resampling_method,
                                           dtype=waveform.dtype,
                                           beta=self.beta, )
                    self.resamplers[sr_kwd] = resampler
                waveform = resampler(waveform)
            waveform = torch.mean(waveform, 0, keepdim=False)
            items.append((waveform,transcript))
        if batch_idx == (len(self.batches) - 1):
            self.shuffle_buckets_and_mark_batches()
        # print('=====================> Batch transcript length:', sum([len(t) for w,t in items]), flush=True)
        return items


def KINSPEAK_TRAIN_DATASET(path_to_cv='/home/user/cv-corpus-12.0-2022-12-07'):
    read_common_voice = data_read_common_voice_with_syllabe_vocab
    train_data_list = [(f'{path_to_cv}/rw/train.tsv', read_common_voice),
                       ]
    return KinSpeakDataset(train_data_list, include_labels=True,
                                    num_shuffle_buckets=100,
                                    max_batch_seconds=80,
                                    debug=True,
                                    batch_amplification_factor=1.1)

def KINSPEAK_VALID_DATASET(path_to_cv='/home/user/cv-corpus-12.0-2022-12-07'):
    read_common_voice = data_read_common_voice_with_syllabe_vocab
    jw_dev_data_list = [(f'{path_to_cv}/rw/dev.tsv', read_common_voice),
                        ]
    return KinSpeakDataset(jw_dev_data_list, include_labels=True,
                                     num_shuffle_buckets=100,
                                     max_batch_seconds=80,
                                     debug=True,
                                     batch_amplification_factor=1.1)

def time_to_spec_samples(secs):
    return int(secs * 1000) // 10 # 10ms hop length

def adaptive_spec_augment(log_mel_spectrogram, # (N,F,L)
                          log_mel_spec_lengths: List[int],
                          frequency_mask_param=27,
                          num_frequency_masks=2,
                          time_mask_ratio_ps = 0.05,
                          num_time_masks=10):
    cloned_spec = log_mel_spectrogram.clone()
    mask_value = cloned_spec.min().item()
    for i,length in enumerate(log_mel_spec_lengths):
        F = frequency_mask_param
        T = int(math.floor(length * time_mask_ratio_ps))
        for _ in range(num_frequency_masks):
            cloned_spec[i:(i+1),:,:length] = torchaudio.functional.mask_along_axis(cloned_spec[i:(i+1),:,:length],
                                                                                     F, mask_value, 1)
        for _ in range(num_time_masks):
            cloned_spec[i:(i+1),:,:length] = torchaudio.functional.mask_along_axis(cloned_spec[i:(i+1),:,:length],
                                                                                     T, mask_value, 2)
    return cloned_spec

def kinspeak_collate(batch_items):
    items = batch_items[0]
    input_data = []
    target_data = []
    input_lengths = []
    target_lengths = []
    for (x,y) in items:
        x_len = x.size(1)
        y_len = len(y)
        input_data.append(x.transpose(0, 1))  # padding dimension must be 0
        target_data.extend(y)
        input_lengths.append(x_len)
        target_lengths.append(y_len)
    with torch.no_grad():
        input = pad_sequence(input_data, batch_first=True).transpose(1, 2)  # (N,F,L)
        target = torch.tensor(target_data, dtype=torch.long)  # (N,S)
    return (input, input_lengths, target, target_lengths)

def kinspeak_collate_with_spec_augment(batch_items):
    items = batch_items[0]
    input_data = []
    target_data = []
    input_lengths = []
    target_lengths = []
    for (x,y) in items:
        x_len = x.size(1)
        y_len = len(y)
        input_data.append(x.transpose(0, 1))  # padding dimension must be 0
        target_data.extend(y)
        input_lengths.append(x_len)
        target_lengths.append(y_len)
    with torch.no_grad():
        input = pad_sequence(input_data, batch_first=True).transpose(1, 2)  # (N,F,L)
        input = adaptive_spec_augment(input,  # (N,F,L)
                              input_lengths)
        target = torch.tensor(target_data, dtype=torch.long)  # (N,S)
    return (input, input_lengths, target, target_lengths)

def kinspeak_collate_without_labels(batch_items):
    items = batch_items[0]
    input_data = []
    input_lengths = []
    for (x,_) in items:
        x_len = x.size(1)
        input_data.append(x.transpose(0, 1))  # padding dimension must be 0
        input_lengths.append(x_len)
    with torch.no_grad():
        input = pad_sequence(input_data, batch_first=True).transpose(1, 2)  # (N,F,L)
    return (input, input_lengths, None, None)
