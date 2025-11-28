import argparse


def get_args_parser() -> argparse.Namespace:
    """Create and parse command-line options for HTR-ConvText.

    This keeps all option names and defaults intact, but organizes them into
    logical groups with clearer help messages.
    """
    parser = argparse.ArgumentParser(
        description='HTR-ConvText: Leveraging Convolution and Textual Context with Mixed Masking for Handwritten Text Recognition',
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---------------------------------------------------------------------
    # Experiment & Logging
    # ---------------------------------------------------------------------
    exp = parser.add_argument_group('Experiment & Logging')
    exp.add_argument('--out-dir', type=str, default='./output',
                     help='Root directory to save logs, checkpoints, and outputs')
    exp.add_argument('--exp-name', type=str, default='IAM_HTR_ORIGAMI_NET',
                     help='Experiment name; results go to <out-dir>/<exp-name>')
    exp.add_argument('--seed', default=123, type=int,
                     help='Random seed for reproducibility')
    exp.add_argument('--use-wandb', action='store_true', default=False,
                     help='Log to Weights & Biases; otherwise use TensorBoard')
    exp.add_argument('--wandb-project', type=str, default='None',
                     help='W&B project name (used only if --use-wandb)')
    exp.add_argument('--print-iter', default=100, type=int,
                     help='Iterations between training status prints')
    exp.add_argument('--eval-iter', default=1000, type=int,
                     help='Iterations between validation runs')

    # ---------------------------------------------------------------------
    # Data & Dataloading
    # ---------------------------------------------------------------------
    data = parser.add_argument_group('Data & Dataloading')
    data.add_argument('--dataset', type=str, choices=['iam', 'read2016', 'lam', 'vnondb'],
                      help='Dataset choice')
    data.add_argument('--data-path', type=str, default='./data/iam/lines/',
                      help='Root directory containing image/line data')
    data.add_argument('--train-data-list', type=str, default='./data/iam/train.ln',
                      help='Path to training list file (e.g., .ln)')
    data.add_argument('--val-data-list', type=str, default='./data/iam/val.ln',
                      help='Path to validation list file (e.g., .ln)')
    data.add_argument('--test-data-list', type=str, default='./data/iam/test.ln',
                      help='Path to test list file (e.g., .ln)')
    data.add_argument('--nb-cls', default=80, type=int,
                      help='Number of classes. IAM=79+1, READ2016=89+1, LAM=90+1, VNOnDB=161+1')
    data.add_argument('--num-workers', default=0, type=int,
                      help='Dataloader worker processes')
    data.add_argument('--img-size', default=[512, 64], type=int, nargs='+',
                      help='Input image size [W, H]')
    data.add_argument('--patch-size', default=[4, 32], type=int, nargs='+',
                      help='Patch size [W, H] for patch embedding')

    # ---------------------------------------------------------------------
    # Training Schedule & Optimization
    # ---------------------------------------------------------------------
    train = parser.add_argument_group('Training Schedule & Optimization')
    train.add_argument('--train-bs', default=8, type=int,
                       help='Training batch size per iteration')
    train.add_argument('--accum-steps', default=1, type=int,
                       help='Gradient accumulation steps; effective batch = train-bs * accum-steps')
    train.add_argument('--val-bs', default=1, type=int,
                       help='Validation/test batch size')
    train.add_argument('--total-iter', default=100000, type=int,
                       help='Total training iterations')
    train.add_argument('--warm-up-iter', default=1000, type=int,
                       help='Warm-up iterations for the optimizer/scheduler')
    train.add_argument('--max-lr', default=1e-3, type=float,
                       help='Peak learning rate')
    train.add_argument('--weight-decay', default=5e-1, type=float,
                       help='Weight decay (L2) regularization')
    train.add_argument('--ema-decay', default=0.9999, type=float,
                       help='Exponential Moving Average (EMA) decay factor for model weights')
    train.add_argument('--alpha', default=0, type=float,
                       help='KL-divergence loss ratio (if applicable)')

    # ---------------------------------------------------------------------
    # Model & Encoder
    # ---------------------------------------------------------------------
    model = parser.add_argument_group('Model & Encoder')
    model.add_argument('--model-type', default='ctc', type=str, choices=['ctc', 'encoder_decoder'],
                      help='Model family to train/use')
    model.add_argument('--cos-temp', default=8, type=int,
                      help='Cosine-similarity classifier temperature')
    model.add_argument('--proj', default=8, type=float,
                      help='Projection dimension or scaling for classifier head')
    model.add_argument('--attn-mask-ratio', default=0., type=float,
                      help='Attention drop-key mask ratio')

    # ---------------------------------------------------------------------
    # Masking Strategy
    # ---------------------------------------------------------------------
    mask = parser.add_argument_group('Masking Strategy')
    mask.add_argument('--use-masking', action='store_true', default=False,
                      help='Enable masking strategy during training')
    mask.add_argument('--mask-ratio', default=0.3, type=float,
                      help='Overall proportion of tokens/patches to mask')
    mask.add_argument('--max-span-length', default=4, type=int,
                      help='Max length for individual span masks')
    mask.add_argument('--spacing', default=0, type=int,
                      help='Minimum spacing between two span masks')
    # Tri-masking schedule ratios
    mask.add_argument('--r-rand', dest='r_rand', default=0.6, type=float,
                      help='Ratio for random masking in tri-masking schedule')
    mask.add_argument('--r-block', dest='r_block', default=0.6, type=float,
                      help='Ratio for block masking in tri-masking schedule')
    mask.add_argument('--block-span', dest='block_span', default=4, type=int,
                      help='Block span length for block masking')
    mask.add_argument('--r-span', dest='r_span', default=0.4, type=float,
                      help='Ratio for span masking in tri-masking schedule')
    mask.add_argument('--max-span', dest='max_span', default=8, type=int,
                      help='Max span length for span masking')

    # ---------------------------------------------------------------------
    # Data Augmentations
    # ---------------------------------------------------------------------
    aug = parser.add_argument_group('Data Augmentations')
    aug.add_argument('--dpi-min-factor', default=0.5, type=float,
                     help='Minimum scaling factor for DPI-based resize')
    aug.add_argument('--dpi-max-factor', default=1.5, type=float,
                     help='Maximum scaling factor for DPI-based resize')
    aug.add_argument('--perspective-low', default=0., type=float,
                     help='Lower bound for perspective transform magnitude')
    aug.add_argument('--perspective-high', default=0.4, type=float,
                     help='Upper bound for perspective transform magnitude')
    aug.add_argument('--elastic-distortion-min-kernel-size', default=3, type=int,
                     help='Minimum kernel size for elastic distortion grid')
    aug.add_argument('--elastic-distortion-max-kernel-size', default=3, type=int,
                     help='Maximum kernel size for elastic distortion grid')
    aug.add_argument('--elastic_distortion-max-magnitude', default=20, type=int,
                     help='Maximum distortion magnitude for elastic transforms')
    aug.add_argument('--elastic-distortion-min-alpha', default=0.5, type=float,
                     help='Minimum alpha for elastic distortion')
    aug.add_argument('--elastic-distortion-max-alpha', default=1, type=float,
                     help='Maximum alpha for elastic distortion')
    aug.add_argument('--elastic-distortion-min-sigma', default=1, type=int,
                     help='Minimum sigma for Gaussian in elastic distortion')
    aug.add_argument('--elastic-distortion-max-sigma', default=10, type=int,
                     help='Maximum sigma for Gaussian in elastic distortion')
    aug.add_argument('--dila-ero-max-kernel', default=3, type=int,
                     help='Max kernel size for dilation/erosion ops')
    aug.add_argument('--dila-ero-iter', default=1, type=int,
                     help='Iterations for dilation/erosion')
    aug.add_argument('--jitter-contrast', default=0.4, type=float,
                     help='ColorJitter: contrast range')
    aug.add_argument('--jitter-brightness', default=0.4, type=float,
                     help='ColorJitter: brightness range')
    aug.add_argument('--jitter-saturation', default=0.4, type=float,
                     help='ColorJitter: saturation range')
    aug.add_argument('--jitter-hue', default=0.2, type=float,
                     help='ColorJitter: hue range')
    aug.add_argument('--blur-min-kernel', default=3, type=int,
                     help='Minimum kernel size for Gaussian blur')
    aug.add_argument('--blur-max-kernel', default=5, type=int,
                     help='Maximum kernel size for Gaussian blur')
    aug.add_argument('--blur-min-sigma', default=3, type=int,
                     help='Minimum sigma for Gaussian blur')
    aug.add_argument('--blur-max-sigma', default=5, type=int,
                     help='Maximum sigma for Gaussian blur')
    aug.add_argument('--sharpen-min-alpha', default=0, type=int,
                     help='Minimum alpha/mix for sharpening')
    aug.add_argument('--sharpen-max-alpha', default=1, type=int,
                     help='Maximum alpha/mix for sharpening')
    aug.add_argument('--sharpen-min-strength', default=0, type=int,
                     help='Minimum sharpening strength')
    aug.add_argument('--sharpen-max-strength', default=1, type=int,
                     help='Maximum sharpening strength')
    aug.add_argument('--zoom-min-h', default=0.8, type=float,
                     help='Minimum vertical zoom factor')
    aug.add_argument('--zoom-max-h', default=1, type=float,
                     help='Maximum vertical zoom factor')
    aug.add_argument('--zoom-min-w', default=0.99, type=float,
                     help='Minimum horizontal zoom factor')
    aug.add_argument('--zoom-max-w', default=1, type=float,
                     help='Maximum horizontal zoom factor')
    aug.add_argument('--proba', default=0.5, type=float,
                     help='Default probability for applying stochastic augmentations')

    # ---------------------------------------------------------------------
    # Decoder & Inference (for encoder-decoder mode)
    # ---------------------------------------------------------------------
    dec = parser.add_argument_group('Decoder & Inference')
    dec.add_argument('--decoder-layers', default=6, type=int,
                     help='Number of Transformer decoder layers')
    dec.add_argument('--decoder-heads', default=8, type=int,
                     help='Number of attention heads in decoder')
    dec.add_argument('--max-seq-len', default=256, type=int,
                     help='Maximum output sequence length')
    dec.add_argument('--label-smoothing', default=0.1, type=float,
                     help='Label-smoothing factor for cross-entropy loss')
    dec.add_argument('--beam-size', default=5, type=int,
                     help='Beam size for beam-search decoding')
    dec.add_argument('--generation-method', default='nucleus', type=str,
                     choices=['greedy', 'nucleus', 'beam_search'],
                     help='Token generation method for inference')
    dec.add_argument('--generation-temperature', default=0.7, type=float,
                     help='Sampling temperature (used by nucleus/greedy sampling)')
    dec.add_argument('--repetition-penalty', default=1.3, type=float,
                     help='Penalty to discourage token repetition during generation')
    dec.add_argument('--top-p', default=0.9, type=float,
                     help='Top-p threshold for nucleus sampling')

    # ---------------------------------------------------------------------
    # TCM (Textual Context Module)
    # ---------------------------------------------------------------------
    tcm = parser.add_argument_group('TCM (Textual Context Module)')
    tcm.add_argument('--tcm-enable', action='store_true', default=False,
                    help='Enable Textual Context Module (TCM)')
    tcm.add_argument('--tcm-lambda', default=1.0, type=float,
                    help='TCM loss weight (λ2 in the paper)')
    tcm.add_argument('--ctc-lambda', default=0.1, type=float,
                    help='CTC loss weight (λ1 in the paper)')
    tcm.add_argument('--tcm-sub-len', default=5, type=int,
                    help='TCM context sub-string length')
    tcm.add_argument('--tcm-warmup-iters', default=0, type=int,
                    help='Warm-up iterations before activating TCM (0 = start immediately)')

    # ---------------------------------------------------------------------
    # Checkpointing & Pretrained Weights
    # ---------------------------------------------------------------------
    ckpt = parser.add_argument_group('Checkpointing & Pretrained Weights')
    ckpt.add_argument('--resume', type=str, default=None,
                      help='Resume training from a checkpoint (alias)')
    ckpt.add_argument('--load-model', type=str, default=None,
                      help='Load a full pretrained model for fine-tuning')
    ckpt.add_argument('--load-encoder-only', action='store_true', default=False,
                      help='Load only encoder weights (transfer learning)')
    ckpt.add_argument('--strict-loading', action='store_true', default=True,
                      help='Use strict key matching when loading weights')

    return parser.parse_args()