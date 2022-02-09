import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import numpy as np
import time
import sys
import os
from os.path import join
from models import utils, caption
from datasets import coco
from configuration import *
from engine import train_one_epoch, evaluate


def main(config):
    available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    for idx, dev in enumerate(available_gpus):
        print("Available GPU-{} name: {}".format(idx, dev))
    device = torch.device(config.device)
    print(f'Initializing Device: {device}')

    seed = config.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    ### Select Model ###
    #model, _ = caption.build_model(config)
    # New Model
    model, _ = caption.build_model_bs(config)
    #lst = [n for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
    #exit()
    # Multi-GPU
    #model = torch.nn.DataParallel(model)

    model.to(device)

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")

    param_dicts = [
        {"params": [p for n, p in model.named_parameters(
        ) if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": config.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(
        param_dicts, lr=config.lr, weight_decay=config.weight_decay)
    ### lr_scheduler with / without warmup ###
    if config.warmup_steps == 0:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_drop)
    else:
        #lr_scheduler = get_cosine_schedule_with_warmup(
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.batch_size * config.epochs - config.warmup_steps,
        )

    if config.modality == 'image':
        dataset_train = coco.build_dataset(config, mode='training')
        dataset_val = coco.build_dataset(config, mode='validation')
    elif config.modality == 'video':
        dataset_train = coco.build_dataset_msvd(config, mode='training')
        dataset_val = coco.build_dataset_msvd(config, mode='validation')
    else:
        raise TypeError("Input Modality not supported!")
    print(f"Train: {len(dataset_train)}")
    print(f"Valid: {len(dataset_val)}")

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, config.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train, batch_sampler=batch_sampler_train, num_workers=config.num_workers)
    data_loader_val = DataLoader(dataset_val, config.batch_size,
                                 sampler=sampler_val, drop_last=False, num_workers=config.num_workers)

    # Redefine criterion
    print("Ignored index: ", dataset_val.tokenizer.convert_tokens_to_ids(dataset_val.tokenizer._pad_token))
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    # Free GPU memory n allow growth
    torch.cuda.empty_cache()

    min_loss_val = 100
    save_dir = 'epoch_checks'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    '''
    if os.path.exists(config.checkpoint):
        print("Loading Checkpoint...")
        checkpoint = torch.load(config.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.start_epoch = checkpoint['epoch'] + 1
        print("Current checkpoint epoch = %d" % checkpoint['epoch'])
    '''

    print("Start Training..")
    for epoch in range(config.start_epoch, config.epochs):
        print(f"Epoch: {epoch}")
        epoch_loss = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, config.clip_max_norm)
        lr_scheduler.step()
        print(f"Training Loss: {epoch_loss}")

        validation_loss = evaluate(model, criterion, data_loader_val, device)
        print(f"Validation Loss: {validation_loss}")

        if validation_loss <= min_loss_val:
            min_loss_val = validation_loss
            best_model_name = config.checkpoint[:-4] + '-best_epoch{}_loss{}.pth'.format(epoch, round(validation_loss*100))
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
            }, join(save_dir, best_model_name))
        elif (epoch + 1) % 5 == 0:
            model_name = config.checkpoint[:-4] + '-epoch{}_loss{}.pth'.format(epoch, round(validation_loss*100))
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
            }, join(save_dir, model_name))
        print()


if __name__ == "__main__":
    config = Config2()
    main(config)
