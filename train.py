import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import os
import json
import valid
from utils import utils
from utils import sam
from utils import option
from data import dataset
from model import htr_convtext
from functools import partial
import random
import numpy as np
import re
import importlib
from model.tcm_head import TCMHead, build_tcm_vocab, make_context_batch
import wandb


def compute_losses(
    args,
    model,
    tcm_head,
    image,
    texts,
    batch_size,
    criterion_ctc,
    converter,
    nb_iter,
    ctc_lambda,
    tcm_lambda,
    stoi,
    mask_mode='span',
    mask_ratio=0.30,
    block_span=4,
    max_span_length=8,
    pre_tcm_ctx=None,
    use_masking=True,
):
    if tcm_head is None or nb_iter < args.tcm_warmup_iters:
        preds = model(image, use_masking=use_masking, mask_mode=mask_mode,
                      mask_ratio=mask_ratio, max_span_length=max_span_length)
        feats = None
    else:
        preds, feats, vis_mask = model(
            image,
            use_masking=use_masking,
            return_features=True,
            return_mask=True,
            mask_mode=mask_mode,
            mask_ratio=mask_ratio,
            block_span=block_span,
            max_span_length=max_span_length
        )
    text_ctc, length_ctc = converter.encode(texts)
    text_ctc = text_ctc.to(preds.device)
    length_ctc = length_ctc.to(preds.device)
    preds_sz = torch.full((batch_size,), preds.size(
        1), dtype=torch.int32, device=preds.device)
    loss_ctc = criterion_ctc(preds.permute(1, 0, 2).log_softmax(2),
                             text_ctc, preds_sz, length_ctc).mean()

    loss_tcm = torch.zeros((), device=preds.device)
    if tcm_head is not None and feats is not None:
        left_ctx, right_ctx, tgt_ids, tgt_mask = pre_tcm_ctx if pre_tcm_ctx is not None else make_context_batch(
            texts, stoi, sub_str_len=args.tcm_sub_len, device=image.device)
        if vis_mask is not None:
            B_v, N_v = vis_mask.shape
            B_t, L_t = tgt_mask.shape
            if N_v != L_t:
                idx = torch.linspace(0, N_v - 1, steps=L_t,
                                     device=vis_mask.device).long()
                focus_mask = vis_mask[:, idx]
            else:
                focus_mask = vis_mask
        else:
            focus_mask = None

        out = tcm_head(
            feats,
            left_ctx, right_ctx,
            tgt_ids, tgt_mask,
            focus_mask=focus_mask
        )
        loss_tcm = out['loss_tcm']

    total = ctc_lambda * loss_ctc + tcm_lambda * loss_tcm
    return total, loss_ctc.detach(), loss_tcm.detach()


def tri_masked_loss(args, model, tcm_head, image, labels, batch_size,
                    criterion, converter, nb_iter, ctc_lambda, tcm_lambda, stoi,
                    r_rand=0.6, r_block=0.6, block_span=4, r_span=0.4, max_span=8):
    total = 0.0
    total_ctc = 0.0
    total_tcm = 0.0
    plans = [("random", r_rand), ("block", r_block), ("span", r_span)]

    if tcm_head is not None and nb_iter >= args.tcm_warmup_iters:
        pre_tcm_ctx = make_context_batch(
            labels, stoi, sub_str_len=args.tcm_sub_len, device=image.device)

    for mode, ratio in plans:
        loss, loss_ctc, loss_tcm = compute_losses(
            args, model, tcm_head, image, labels, batch_size, criterion, converter,
            nb_iter, ctc_lambda, tcm_lambda, stoi,
            mask_mode=mode, mask_ratio=ratio, block_span=block_span, max_span_length=max_span,
            pre_tcm_ctx=pre_tcm_ctx
        )
        total += loss
        total_ctc += loss_ctc
        total_tcm += loss_tcm

    denom = 3.0
    return total/denom, total_ctc/denom, total_tcm/denom


def main():

    args = option.get_args_parser()
    torch.manual_seed(args.seed)

    args.save_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.save_dir, exist_ok=True)

    logger = utils.get_logger(args.save_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    writer = SummaryWriter(args.save_dir)

    if getattr(args, 'use_wandb', False):
        try:
            wandb = importlib.import_module('wandb')
            wandb.init(project=getattr(args, 'wandb_project', 'None'), name=args.exp_name,
                       config=vars(args), dir=args.save_dir)
            logger.info("Weights & Biases logging enabled")
        except Exception as e:
            logger.warning(
                f"Failed to initialize wandb: {e}. Continuing without wandb.")
            wandb = None
    else:
        wandb = None

    torch.backends.cudnn.benchmark = True

    model = htr_convtext.create_model(
        nb_cls=args.nb_cls, img_size=args.img_size[::-1])

    total_param = sum(p.numel() for p in model.parameters())
    logger.info('total_param is {}'.format(total_param))

    model.train()
    model = model.cuda()
    ema_decay = args.ema_decay
    logger.info(f"Using EMA decay: {ema_decay}")
    model_ema = utils.ModelEma(model, ema_decay)
    model.zero_grad()

    resume_path = args.resume
    best_cer, best_wer, start_iter, optimizer_state, train_loss, train_loss_count = utils.load_checkpoint(
        model, model_ema, None, resume_path, logger)

    logger.info('Loading train loader...')
    train_dataset = dataset.myLoadDS(
        args.train_data_list, args.data_path, args.img_size, dataset=args.dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.train_bs,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=args.num_workers,
                                               collate_fn=partial(dataset.SameTrCollate, args=args))
    train_iter = dataset.cycle_data(train_loader)

    logger.info('Loading val loader...')
    val_dataset = dataset.myLoadDS(
        args.val_data_list, args.data_path, args.img_size, ralph=train_dataset.ralph, dataset=args.dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.val_bs,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=args.num_workers)

    criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True)
    converter = utils.CTCLabelConverter(train_dataset.ralph.values())

    stoi, itos, pad_id = build_tcm_vocab(converter)
    vocab_size_tcm = len(itos)
    d_vis = model.embed_dim

    if args.tcm_enable:
        tcm_head = TCMHead(d_vis=d_vis, vocab_size_tcm=vocab_size_tcm, pad_id=pad_id,
                           sub_str_len=args.tcm_sub_len).cuda()
        tcm_head.train()
    else:
        tcm_head = None

    param_groups = list(model.parameters())
    if args.tcm_enable and tcm_head is not None:
        param_groups += list(tcm_head.parameters())
        logger.info(
            f"Optimizing {sum(p.numel() for p in tcm_head.parameters())} tcm params in addition to model params")
    optimizer = sam.SAM(param_groups, torch.optim.AdamW,
                        lr=1e-7, betas=(0.9, 0.99), weight_decay=args.weight_decay)

    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
            logger.info("Successfully loaded optimizer state")
        except Exception as e:
            logger.warning(f"Failed to load optimizer state: {e}")
            logger.info(
                "Continuing training without optimizer state (will restart from initial lr/momentum)")
    elif resume_path and os.path.isfile(resume_path):
        try:
            ckpt = torch.load(resume_path, map_location='cpu',
                              weights_only=False)
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
                logger.info("Loaded optimizer state from checkpoint directly")
        except Exception as e:
            logger.warning(
                f"Could not load optimizer state from checkpoint: {e}")

    if resume_path and os.path.isfile(resume_path) and tcm_head is not None:
        try:
            ckpt = torch.load(resume_path, map_location='cpu',
                              weights_only=False)
            if 'tcm_head' in ckpt:
                tcm_head.load_state_dict(ckpt['tcm_head'], strict=False)
                logger.info("Restored tcm head state from checkpoint")
            else:
                logger.info(
                    "No tcm head state found in checkpoint; training tcm from scratch")
        except Exception as e:
            logger.warning(f"Failed to restore tcm head from checkpoint: {e}")

    best_cer, best_wer = best_cer, best_wer
    train_loss = train_loss
    train_loss_count = train_loss_count

    #### ---- train & eval ---- ####
    logger.info('Start training...')
    accum_steps = max(1, int(getattr(args, 'accum_steps', 1)))
    micro_step = 0
    avg_loss_ctc = 0.0
    avg_loss_tcm = 0.0

    for nb_iter in range(start_iter, args.total_iter):
        optimizer, current_lr = utils.update_lr_cos(
            nb_iter, args.warm_up_iter, args.total_iter, args.max_lr, optimizer)
        optimizer.zero_grad()
        total_loss_this_macro = 0.0
        avg_loss_ctc = 0.0
        avg_loss_tcm = 0.0
        cached_batches = []
        for micro_step in range(accum_steps):
            batch = next(train_iter)
            cached_batches.append(batch)
            image = batch[0].cuda(non_blocking=True)
            text, length = converter.encode(batch[1])
            batch_size = image.size(0)
            if args.use_masking:
                # loss, loss_ctc, loss_tcm = tri_masked_loss(
                #     args, model, tcm_head, image, batch[1], batch_size, criterion, converter,
                #     nb_iter, args.ctc_lambda, args.tcm_lambda, stoi,
                #     r_rand=args.r_rand,
                #     r_block=args.r_block,
                #     block_span=args.block_span,
                #     r_span=args.r_span,
                #     max_span=args.max_span
                # )
                loss, loss_ctc, loss_tcm = compute_losses(
                    args, model, tcm_head, image, batch[1], batch_size, criterion, converter,
                    nb_iter, args.ctc_lambda, args.tcm_lambda, stoi,
                    mask_mode='span', mask_ratio=0.4, max_span_length=8, use_masking=True
                )
            else:
                loss, loss_ctc, loss_tcm = compute_losses(
                    args, model, tcm_head, image, batch[1], batch_size, criterion, converter,
                    nb_iter, args.ctc_lambda, args.tcm_lambda, stoi, use_masking=False
                )
            (loss / accum_steps).backward()
            total_loss_this_macro += loss.item()
            avg_loss_ctc += loss_ctc.mean().item()
            avg_loss_tcm += loss_tcm.mean().item()

        optimizer.first_step(zero_grad=True)

        # Recompute with perturbed weights and accumulate again for the second step
        for micro_step in range(accum_steps):
            batch = cached_batches[micro_step]
            image = batch[0].cuda(non_blocking=True)
            text, length = converter.encode(batch[1])
            batch_size = image.size(0)
            if args.use_masking:
                # loss2, loss_ctc, loss_tcm = tri_masked_loss(
                #     args, model, tcm_head, image, batch[1], batch_size, criterion, converter,
                #     nb_iter, args.ctc_lambda, args.tcm_lambda, stoi,
                #     r_rand=args.r_rand,
                #     r_block=args.r_block,
                #     block_span=args.block_span,
                #     r_span=args.r_span,
                #     max_span=args.max_span
                # )
                loss2, loss_ctc, loss_tcm = compute_losses(
                    args, model, tcm_head, image, batch[1], batch_size, criterion, converter,
                    nb_iter, args.ctc_lambda, args.tcm_lambda, stoi,
                    mask_mode='span', mask_ratio=0.4, max_span_length=8, use_masking=True
                )
            else:
                loss2, loss_ctc, loss_tcm = compute_losses(
                    args, model, tcm_head, image, batch[1], batch_size, criterion, converter,
                    nb_iter, args.ctc_lambda, args.tcm_lambda, stoi, use_masking=False
                )
            (loss2 / accum_steps).backward()
        optimizer.second_step(zero_grad=True)
        model.zero_grad()
        model_ema.update(model, num_updates=nb_iter / 2)

        train_loss += total_loss_this_macro / accum_steps
        train_loss_count += 1

        if nb_iter % args.print_iter == 0:
            train_loss_avg = train_loss / train_loss_count if train_loss_count > 0 else 0.0

            logger.info(
                f'Iter : {nb_iter} \t LR : {current_lr:0.5f} \t total : {train_loss_avg:0.5f} \t CTC : {(avg_loss_ctc/accum_steps):0.5f} \t tcm : {(avg_loss_tcm/accum_steps):0.5f} \t ')

            writer.add_scalar('./Train/lr', current_lr, nb_iter)
            writer.add_scalar('./Train/train_loss', train_loss_avg, nb_iter)
            if wandb is not None:
                wandb.log({
                    'train/lr': current_lr,
                    'train/loss': train_loss_avg,
                    'train/CTC': (avg_loss_ctc/accum_steps),
                    'train/tcm': (avg_loss_tcm/accum_steps),
                    'iter': nb_iter,
                }, step=nb_iter)
            train_loss = 0.0
            train_loss_count = 0

        if nb_iter % args.eval_iter == 0:
            model.eval()
            with torch.no_grad():
                val_loss, val_cer, val_wer, preds, labels = valid.validation(model_ema.ema,
                                                                             criterion,
                                                                             val_loader,
                                                                             converter)
                if nb_iter % args.eval_iter*5 == 0:
                    ckpt_name = f"checkpoint_{best_cer:.4f}_{best_wer:.4f}_{nb_iter}.pth"
                    checkpoint = {
                        'model': model.state_dict(),
                        'state_dict_ema': model_ema.ema.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'nb_iter': nb_iter,
                        'best_cer': best_cer,
                        'best_wer': best_wer,
                        'args': vars(args),
                        'random_state': random.getstate(),
                        'numpy_state': np.random.get_state(),
                        'torch_state': torch.get_rng_state(),
                        'torch_cuda_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                        'train_loss': train_loss,
                        'train_loss_count': train_loss_count,
                    }
                    if tcm_head is not None:
                        checkpoint['tcm_head'] = tcm_head.state_dict()
                    torch.save(checkpoint, os.path.join(
                        args.save_dir, ckpt_name))
                if val_cer < best_cer:
                    logger.info(
                        f'CER improved from {best_cer:.4f} to {val_cer:.4f}!!!')
                    best_cer = val_cer
                    checkpoint = {
                        'model': model.state_dict(),
                        'state_dict_ema': model_ema.ema.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'nb_iter': nb_iter,
                        'best_cer': best_cer,
                        'best_wer': best_wer,
                        'args': vars(args),
                        'random_state': random.getstate(),
                        'numpy_state': np.random.get_state(),
                        'torch_state': torch.get_rng_state(),
                        'torch_cuda_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                        'train_loss': train_loss,
                        'train_loss_count': train_loss_count,
                    }
                    if tcm_head is not None:
                        checkpoint['tcm_head'] = tcm_head.state_dict()
                    torch.save(checkpoint, os.path.join(
                        args.save_dir, 'best_CER.pth'))

                if val_wer < best_wer:
                    logger.info(
                        f'WER improved from {best_wer:.4f} to {val_wer:.4f}!!!')
                    best_wer = val_wer
                    checkpoint = {
                        'model': model.state_dict(),
                        'state_dict_ema': model_ema.ema.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'nb_iter': nb_iter,
                        'best_cer': best_cer,
                        'best_wer': best_wer,
                        'args': vars(args),
                        'random_state': random.getstate(),
                        'numpy_state': np.random.get_state(),
                        'torch_state': torch.get_rng_state(),
                        'torch_cuda_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                        'train_loss': train_loss,
                        'train_loss_count': train_loss_count,
                    }
                    if tcm_head is not None:
                        checkpoint['tcm_head'] = tcm_head.state_dict()
                    torch.save(checkpoint, os.path.join(
                        args.save_dir, 'best_WER.pth'))

                logger.info(
                    f'Val. loss : {val_loss:0.3f} \t CER : {val_cer:0.4f} \t WER : {val_wer:0.4f} \t ')

                writer.add_scalar('./VAL/CER', val_cer, nb_iter)
                writer.add_scalar('./VAL/WER', val_wer, nb_iter)
                writer.add_scalar('./VAL/bestCER', best_cer, nb_iter)
                writer.add_scalar('./VAL/bestWER', best_wer, nb_iter)
                writer.add_scalar('./VAL/val_loss', val_loss, nb_iter)
                if wandb is not None:
                    wandb.log({
                        'val/loss': val_loss,
                        'val/CER': val_cer,
                        'val/WER': val_wer,
                        'val/best_CER': best_cer,
                        'val/best_WER': best_wer,
                        'iter': nb_iter,
                    }, step=nb_iter)
                model.train()


if __name__ == '__main__':
    main()
