import torch

import os
import re
import json
import valid
from utils import utils
from utils import option
from data import dataset
from model import htr_convtext
from collections import OrderedDict


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    args.save_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.save_dir, exist_ok=True)
    logger = utils.get_logger(args.save_dir)

    model = htr_convtext.create_model(
        nb_cls=args.nb_cls, img_size=args.img_size[::-1])

    pth_path = args.resume
    logger.info('loading HWR checkpoint from {}'.format(pth_path))

    ckpt = torch.load(pth_path, map_location='cpu', weights_only=False)
    model_dict = OrderedDict()
    pattern = re.compile('module.')

    for k, v in ckpt['state_dict_ema'].items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            model_dict[k] = v

    model.load_state_dict(model_dict, strict=True)
    model = model.cuda()

    logger.info('Loading test loader...')
    train_dataset = dataset.myLoadDS(
        args.train_data_list, args.data_path, args.img_size, dataset=args.dataset)

    test_dataset = dataset.myLoadDS(
        args.test_data_list, args.data_path, args.img_size, ralph=train_dataset.ralph, dataset=args.dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.val_bs,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=args.num_workers)

    converter = utils.CTCLabelConverter(train_dataset.ralph.values())
    criterion = torch.nn.CTCLoss(
        reduction='none', zero_infinity=True).to(device)

    model.eval()
    with torch.no_grad():
        val_loss, val_cer, val_wer, preds, labels = valid.validation(
            model,
            criterion,
            test_loader,
            converter,
        )

    logger.info(
        f'Test. loss : {val_loss:0.3f} \t CER : {val_cer:0.4f} \t WER : {val_wer:0.4f} ')

    # Save predictions as JSON
    results = {
        "test_metrics": {
            "loss": float(val_loss),
            "cer": float(val_cer),
            "wer": float(val_wer)
        },
        "predictions": []
    }

    def _levenshtein(pred_tokens, gt_tokens):
        if pred_tokens == gt_tokens:
            return 0
        lp, lg = len(pred_tokens), len(gt_tokens)
        if lp == 0:
            return lg
        if lg == 0:
            return lp
        prev = list(range(lg + 1))
        for i in range(1, lp + 1):
            cur = [i]
            pi = pred_tokens[i - 1]
            for j in range(1, lg + 1):
                gj = gt_tokens[j - 1]
                cost = 0 if pi == gj else 1
                cur.append(
                    min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
            prev = cur
        return prev[-1]

    def _levenshtein_str(a: str, b: str):
        return _levenshtein(list(a), list(b))

    def _cer(pred: str, gt: str):
        if len(gt) == 0:
            return 0.0 if len(pred) == 0 else 1.0
        return _levenshtein_str(pred, gt) / len(gt)

    def _wer(pred: str, gt: str):
        gt_words = gt.split()
        pred_words = pred.split()
        if len(gt_words) == 0:
            return 0.0 if len(pred_words) == 0 else 1.0
        return _levenshtein(pred_words, gt_words) / len(gt_words)

    for i, (pred, label) in enumerate(zip(preds, labels)):
        if i < len(test_dataset.fns):
            img_path = test_dataset.fns[i]
            img_name = os.path.basename(img_path)
        else:
            img_path = None
            img_name = None
        results["predictions"].append({
            "sample_id": i + 1,
            "image_filename": img_name,
            "image_path": img_path,
            "prediction": pred,
            "ground_truth": label,
            "match": pred == label,
            "cer": round(float(_cer(pred, label)), 6),
            "wer": round(float(_wer(pred, label)), 6)
        })

    pred_file = os.path.join(args.save_dir, 'predictions.json')
    with open(pred_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    args = option.get_args_parser()
    main()
