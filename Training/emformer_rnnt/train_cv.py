from __future__ import print_function, division

import gc
import math
import os
import os.path
import os.path
import sys
import time
from shutil import copyfile

import apex
import progressbar
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from arguments import py_trainer_args
from data import KINSPEAK_TRAIN_DATASET, KINSPEAK_VALID_DATASET, kinspeak_collate_with_spec_augment, kinspeak_collate
from learning_rates import InverseSQRT_LRScheduler
from misc_functions import time_now
from model import KinspeakEmformerRNNT


def save_model_state(model, scaler, optimizer, lr_scheduler, best_valid_loss, epoch, num_epochs, filename):
    model.eval()
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        torch.save({'model_state_dict': (model.state_dict() if (dist.get_world_size() > 1) else model.state_dict()),
                    'scaler_state_dict': scaler.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    'best_valid_loss': best_valid_loss,
                    'epoch': epoch,
                    'num_epochs': num_epochs}, filename)
    model.train()
def load_model_state(model, scaler, optimizer, lr_scheduler, map_location, filename):
    if (dist.get_rank() == 0):
        print(time_now(), 'Loading model state...', flush=True)
    kb_state_dict = torch.load(filename, map_location=map_location)
    model.load_state_dict(kb_state_dict['model_state_dict'])
    del kb_state_dict
    gc.collect()
    kb_state_dict = torch.load(filename, map_location=torch.device('cpu'))
    lr_scheduler.load_state_dict(kb_state_dict['lr_scheduler_state_dict'])
    scaler.load_state_dict(kb_state_dict['scaler_state_dict'])
    optimizer.load_state_dict(kb_state_dict['optimizer_state_dict'])
    best_valid_loss = kb_state_dict['best_valid_loss']
    epoch = kb_state_dict['epoch']
    del kb_state_dict
    gc.collect()
    return epoch, best_valid_loss

def validation_loop(keyword, model, device, data_loader, lr_scheduler, epoch, num_epochs):
    world_size = dist.get_world_size()
    model.eval()
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    rnnt_loss_aggr = torch.tensor(0.0, device=device)
    syllabe_loss_avg_aggr = torch.tensor(0.0, device=device)
    syllabe_nll_loss_avg_aggr = torch.tensor(0.0, device=device)

    num_data_items = torch.tensor(len(data_loader), device=device)
    dist.all_reduce(num_data_items, op=dist.ReduceOp.MIN)
    total_data_items = int(num_data_items.cpu().item())
    print(dist.get_rank(),'Evaluating on {} items'.format(total_data_items))

    count_items = 0
    print(time_now(), 'Evaluating on dev ...', flush=True)
    with progressbar.ProgressBar(initial_value=0,
                                 max_value=len(data_loader),
                                 redirect_stdout=True) as bar:
        for batch_idx, batch_data_item in enumerate(data_loader):
            bar.update(batch_idx)
            target_syllabe_gpt_output = None
            (log_mel_spectrograms, log_mel_spectrogram_lengths,
             target_syllabe_ids, target_syllabe_id_lengths) = batch_data_item
            log_mel_spectrograms = log_mel_spectrograms.to(device)
            target_syllabe_ids = target_syllabe_ids.to(device)
            rnnt_loss = model(log_mel_spectrograms, log_mel_spectrogram_lengths,
                                           target_syllabe_ids, target_syllabe_id_lengths,
                                           target_syllabe_ids_with_eos=True,
                                           target_syllabe_gpt_output=target_syllabe_gpt_output)
            rnnt_loss_aggr += (rnnt_loss.detach().clone().squeeze())
            count_items += 1
            if batch_idx == (total_data_items - 1):
                break

    loss_Z = count_items * world_size
    # Aggregate losses
    dist.all_reduce(rnnt_loss_aggr)

    # Logging & Checkpointing
    if dist.get_rank() == 0:
        print(time_now(),
              'After Iter:', "{}/{}".format(lr_scheduler.num_iters, lr_scheduler.end_iter),
              keyword,
              'VALIDATION LOSS:',
              'RNNT:', "{:.6f}".format(rnnt_loss_aggr.item() / loss_Z),
              'LR:', "{:.8f}/{}".format(lr_scheduler.get_lr(), lr_scheduler.start_lr),
              'Warmup:', "{}".format(lr_scheduler.warmup_iter),
              'Epochs:', '{}/{}'.format(epoch + 1, num_epochs), flush=True)
        sys.stdout.flush()

    return (rnnt_loss_aggr.item() / loss_Z) # + (syllabe_nll_loss_avg_aggr.item() / loss_Z)

def train_loop(model, device, scaler, optimizer,
               lr_scheduler, train_data_loader,
               cv_validation_data_loader,
               save_file_path, accumulation_steps, epoch, num_epochs, total_steps, bar, best_valid_loss):
    world_size = dist.get_world_size()
    model.train()
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    rnnt_loss_aggr = torch.tensor(0.0, device=device)

    num_data_items = torch.tensor(len(train_data_loader), device=device)
    dist.all_reduce(num_data_items, op=dist.ReduceOp.MIN)
    total_data_items = int(num_data_items.cpu().item())
    print(dist.get_rank(),f'Training on {total_data_items} batches out of {len(train_data_loader)}')

    # Train
    start_steps = total_steps
    start_time = time.time()
    count_items = 0
    for batch_idx, batch_data_item in enumerate(train_data_loader):
        total_steps += 1
        count_items += 1
        target_syllabe_gpt_output = None
        (log_mel_spectrograms, log_mel_spectrogram_lengths,
         target_syllabe_ids, target_syllabe_id_lengths) = batch_data_item
        log_mel_spectrograms = log_mel_spectrograms.to(device)
        target_syllabe_ids = target_syllabe_ids.to(device)
        with torch.cuda.amp.autocast():
            if (int(total_steps % (accumulation_steps // world_size)) != 0) and (world_size > 1):
                with model.no_sync():
                    rnnt_loss = model(log_mel_spectrograms, log_mel_spectrogram_lengths,
                                                   target_syllabe_ids, target_syllabe_id_lengths,
                                                   target_syllabe_ids_with_eos=True,
                                                   target_syllabe_gpt_output = target_syllabe_gpt_output)
                    scaler.scale(rnnt_loss/accumulation_steps).backward()
            else:
                rnnt_loss = model(log_mel_spectrograms, log_mel_spectrogram_lengths,
                                 target_syllabe_ids, target_syllabe_id_lengths,
                                 target_syllabe_ids_with_eos=True,
                                 target_syllabe_gpt_output=target_syllabe_gpt_output)
                scaler.scale(rnnt_loss / accumulation_steps).backward()
        rnnt_loss_aggr += (rnnt_loss.detach().clone().squeeze())

        if int(total_steps % (accumulation_steps // world_size)) == 0:
            lr_scheduler.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            current_time = time.time()
            if (lr_scheduler.num_iters % 100) == 0:
                torch.cuda.empty_cache()
            if (dist.get_rank() == 0):
                print(time_now(),
                      'Iter:', "{}/{}".format(lr_scheduler.num_iters, lr_scheduler.end_iter),
                      'OBJ:',
                      'RNNT:', "{:.6f}".format(rnnt_loss_aggr.item() / count_items),
                      'LR:', "{:.8f}/{}".format(lr_scheduler.get_lr(), lr_scheduler.start_lr),
                      'Warmup:', "{}".format(lr_scheduler.warmup_iter),
                      'Milli_Steps_Per_Second (MSS): ', "{:.3f}".format(
                        1000.0 * ((total_steps - start_steps) / (accumulation_steps // world_size)) / (
                                    current_time - start_time)),
                      'Epochs:', '{}/{}'.format(epoch + 1, num_epochs), flush=True)
                bar.update(epoch)
                bar.fd.flush()
                sys.stdout.flush()
                sys.stderr.flush()
                if (lr_scheduler.num_iters % 5000) == 0:
                    if (math.isfinite((rnnt_loss_aggr.item()))):
                        save_model_state(model, scaler, optimizer, lr_scheduler, best_valid_loss,
                                         epoch, num_epochs, save_file_path+"_safe_checkpoint.pt")
                        print(time_now(), 'Safe model checkpointed!', flush=True)
        if batch_idx == (total_data_items - 1):
            break

    torch.cuda.empty_cache()

    loss_Z = count_items * world_size
    # Aggregate losses
    dist.all_reduce(rnnt_loss_aggr)

    # Logging & Checkpointing
    if dist.get_rank() == 0:
        print(time_now(),
              'After Iter:', "{}/{}".format(lr_scheduler.num_iters, lr_scheduler.end_iter),
              'LOSS:',
              'CTC:', "{:.6f}".format(rnnt_loss_aggr.item() / loss_Z),
              'LR:', "{:.8f}/{:.5f}".format(lr_scheduler.get_lr(), lr_scheduler.start_lr),
              'Warmup:', "{}".format(lr_scheduler.warmup_iter),
              'Epochs:', '{}/{}'.format(epoch + 1, num_epochs), flush=True)
        sys.stdout.flush()

        if os.path.exists(save_file_path):
            copyfile(save_file_path, save_file_path + "_prev_checkpoint.pt")
            print(time_now(), 'Prev model file checkpointed!', flush=True)

        save_model_state(model, scaler, optimizer, lr_scheduler, best_valid_loss,
                         epoch+1, num_epochs, save_file_path)

    torch.cuda.empty_cache()
    with torch.no_grad():
        cv_valid_loss = validation_loop('COMMON_VOICE', model, device, cv_validation_data_loader, lr_scheduler, epoch, num_epochs)
        if cv_valid_loss < best_valid_loss:
            best_valid_loss = cv_valid_loss
            if dist.get_rank() == 0:
                save_model_state(model, scaler, optimizer, lr_scheduler, best_valid_loss,
                                 epoch+1, num_epochs, save_file_path+'_best_valid_loss.pt')

    return total_steps, best_valid_loss

def syllabe_vocab_targets():
    from syllabe_vocab import BLANK_ID, syllbe_vocab_size
    return syllbe_vocab_size(), BLANK_ID

def train_model_one_stage(args, model, home_path, device, map_location):
    scaler = torch.cuda.amp.GradScaler()
    curr_save_file_path = (home_path + 'data/emformer_rnnt_base_model.pt')
    if (dist.get_rank() == 0):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('---------------------------------- Kinspeak ASR Emformer RNNT Model Size ----------------------------------------')
        print(time_now(), 'Total params:', total_params, 'Trainable params:', trainable_params, flush=True)
        print('Saving model in:', curr_save_file_path)
        print('---------------------------------------------------------------------------------------')


    best_valid_loss = 9999.0

    if (dist.get_rank() == 0):
        print(time_now(), 'Reading parallel data ...', flush=True)

    train_dataset = KINSPEAK_TRAIN_DATASET()
    cv_dev_dataset = KINSPEAK_VALID_DATASET()

    train_data_loader = DataLoader(train_dataset, batch_size=1,
                                   collate_fn=kinspeak_collate_with_spec_augment,
                                   drop_last=False, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)

    cv_validation_data_loader = DataLoader(cv_dev_dataset, batch_size=1,
                                        collate_fn=kinspeak_collate,
                                        pin_memory=True,
                                        drop_last=False, shuffle=False, num_workers=2, persistent_workers=False)

    if (dist.get_rank() == 0):
        print(time_now(), 'Done reading parallel data!', flush=True)

    iters_per_epoch = len(train_data_loader) // (args.asr_accumulation_steps // dist.get_world_size())

    peak_lr = 2e-4

    warmup_steps = 5000

    optimizer = apex.optimizers.FusedAdam(model.parameters(), lr=peak_lr, betas=(0.9, 0.98), eps=1e-08)
    lr_scheduler = InverseSQRT_LRScheduler(optimizer,
                                                   start_lr=peak_lr,
                                                   warmup_iter=warmup_steps,
                                                   num_iters=(iters_per_epoch * args.asr_num_train_epochs),
                                                   last_iter=-1)

    if (not args.load_saved_model) and (dist.get_world_size() > 1):
        if (dist.get_rank() == 0):
            save_model_state(model, scaler, optimizer, lr_scheduler, best_valid_loss,
                             0, args.asr_num_train_epochs, curr_save_file_path)
        dist.barrier()
        args.load_saved_model = True

    epoch = 0
    if args.load_saved_model:
        (epoch,
         best_valid_loss) = load_model_state(model, scaler, optimizer,
                                             lr_scheduler,
                                             map_location, curr_save_file_path)

    lr_scheduler.end_iter = (iters_per_epoch * args.asr_num_train_epochs)

    epoch = int(round(args.asr_num_train_epochs * (lr_scheduler.num_iters / lr_scheduler.end_iter)))

    if (dist.get_rank() == 0):
        print('---------------------------------------------------------------------------------------')
        print('Model Arguments:', args)
        print('------------------ Train Config --------------------')
        print('epoch:', epoch)
        print('num_epochs:', args.asr_num_train_epochs)
        print('iters_per_epoch:', iters_per_epoch)
        print('Iters:', lr_scheduler.num_iters)
        print('End_Iter:', lr_scheduler.end_iter)
        print('Warmup steps:', lr_scheduler.warmup_iter)
        print('number_of_load_batches:', len(train_data_loader))
        print('accumulation_steps:', args.asr_accumulation_steps)
        print('batch_size:', args.asr_batch_max_seconds, 'secs')
        print('effective_batch_size:', args.asr_batch_max_seconds * args.asr_accumulation_steps, 'secs')
        print('peak_lr: {}'.format(peak_lr))
        print('-----------------------------------------------------')

    total_steps = int(lr_scheduler.num_iters * args.asr_accumulation_steps)

    if (dist.get_rank() == 0):
        print(time_now(), 'Start training (total steps: {}) ....'.format(total_steps), flush=True)

    # epoch = 0
    with progressbar.ProgressBar(initial_value=epoch,
                                 max_value=args.asr_num_train_epochs,
                                 redirect_stdout=True) as bar:
        if (dist.get_rank() == 0):
            bar.update(epoch)
            sys.stdout.flush()
        for ep in range(epoch, args.asr_num_train_epochs):
            (total_steps,
             best_valid_loss) = train_loop(model, device, scaler, optimizer,
                                           lr_scheduler,
                                           train_data_loader, cv_validation_data_loader,
                                           curr_save_file_path, args.asr_accumulation_steps,
                                           ep, args.asr_num_train_epochs, total_steps, bar, best_valid_loss)
            if (dist.get_rank() == 0):
                print(time_now(), (ep + 1), 'TRAINING EPOCHS COMPLETE!', flush=True)
                bar.update(ep + 1)
                sys.stdout.flush()
            dist.barrier()

def train_fn(rank, args):
    print(time_now(), 'Called train_fn()', flush=True)
    device = torch.device('cuda:%d' % rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    torch.backends.cudnn.benchmark = True
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True
    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(rank)
    # torch.autograd.set_detect_anomaly(True)

    print('Waiting for device: ', device, "from", dist.get_world_size(), 'processes', flush=True)
    dist.barrier()
    print('Using device: ', device, "from", dist.get_world_size(), 'processes', flush=True)

    args.asr_num_train_epochs = args.asr_num_train_epochs // dist.get_world_size()

    home_path = args.home_path

    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    (target_vocab_size, target_blank_id) = syllabe_vocab_targets()
    model = KinspeakEmformerRNNT(target_vocab_size, target_blank_id).to(device)
    model.float()
    print(time_now(), 'Using DDP for training!')
    # model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    model.float()

    train_model_one_stage(args, model, home_path, device, map_location)
    dist.barrier()

def kinspeak_asr_trainer_main():
    args = py_trainer_args()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8181'
    if args.gpus == 0:
        args.world_size = 1

    mp.spawn(train_fn, nprocs=args.world_size, args=(args,))


if __name__ == '__main__':
    kinspeak_asr_trainer_main()
