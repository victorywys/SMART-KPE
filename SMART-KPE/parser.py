import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    # IO
    parser.add_argument(
        "--read_from_cached_features",
        action="store_true",
        help="Whether to read saved features")

    parser.add_argument(
        "--cached_features_dir",
        default="./features",
        help="The input data dir(cached)")

    parser.add_argument(
        "--data_dir",
        default="./data",
        help="The input data dir(.jsonl)")

    parser.add_argument(
        "--from_checkpoint",
        default=None,
        help="The directory to load the model checkpoint")

    parser.add_argument(
        "--output_dir",
        default="./output",
        help="The output dir")

    parser.add_argument(
        "--print_dir",
        default="./output",
        help="The printing dir")

    parser.add_argument(
        "--meta_dir",
        default="./meta_data",
        help="The metadata dir.")

    parser.add_argument(
        "--use_snapshot",
        action="store_true",
        help="Whether to use snapshot feature")

    parser.add_argument(
        "--include_title",
        action="store_true",
        help="Whether to use titles")

    parser.add_argument(
        "--snapshot_dim",
        type=int, default=512,
        help="The dimension of snapshot vectors")

    '''
    parser.add_argument(
        "--elmo_option_file",
        default="None",
        help="ELMO option file")

    parser.add_argument(
        "--elmo_weight_file",
        default="None",
        help="ELMO weight file")

    parser.add_argument(
        "--elmo_finetune",
        action="store_true",
        help="ELMO weights")

    parser.add_argument(
        "--elmo_layernorm",
        action="store_true",
        help="ELMO")
    '''
    parser.add_argument(
        "--train",
        action='store_true',
        help="whether to train")
    parser.set_defaults(train=False)

    parser.add_argument(
        "--dev",
        action='store_true',
        help="whether to evaluate")
    parser.set_defaults(dev=False)

    parser.add_argument(
        "--test",
        action='store_true',
        help="whether to test")
    parser.set_defaults(test=False)

    parser.add_argument(
        "--only-predictions",
        dest='only_pred',
        action='store_true',
        help="only present the prediction results and do not evaluate precisions and recalls according to the golden labels.")
    parser.set_defaults(only_pred=False)

    # Model
    parser.add_argument(
        "--device",
        default=None,
        help="Whether to use cuda")

    parser.add_argument(
        "--learning_rate",
        type=float, default=1e-3,
        help="Learning rate")

    parser.add_argument(
        "--max_grad_norm",
        type=float, default=1.,
        help="max gradient norm used for clipping")

    parser.add_argument(
        "--num_train_epochs",
        type=int, default=1,
        help="Number of epochs")

    parser.add_argument(
        "--batch_size",
        type=int, default=32,
        help="Batch size")

    parser.add_argument(
        "--max_text_length",
        type=int, default=256,
        help="Sequence length")

    parser.add_argument(
        "--logging_steps",
        type=int, default=2000,
        help="Steps for warmup")

    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="whether to evaluate model during training")
    parser.set_defaults(evaluate_during_training=False)

    parser.add_argument(
        "--filter_predicted_kp",
        action="store_true",
        help="whether to filter kps")

    parser.add_argument(
        "--save_steps",
        type=int, default=2000,
        help="Steps to save model")

    parser.add_argument(
        "--save_best",
        action='store_true',
        help="store the best model checkpoint according to main_metric")
    parser.set_defaults(save_best=False)

    parser.add_argument(
        "--main_metric",
        type=str, default="P@3",
        help="the main metric to compare for best model saving")

    parser.add_argument(
        "--max_steps",
        type=int, default=-1,
        help="Max steps per epoch")

    parser.add_argument(
        "--evaluate_num",
        type=int, default=-1,
        help="when set with positive integer, use limited samples to evalute during training. this will save much time for debugging and logging.")

    parser.add_argument("--positional_size",type=int, default=256,help="Positional encoding")
    #parser.add_argument("--elmo_size", type=int, default=1024, help="ELMo encoding")
    parser.add_argument("--tag_num", type=int, default=3, help="Tags.")
    parser.add_argument("--bert_size", type=int, default=768, help="ELMo encoding")
    parser.add_argument("--visual_size",type=int, default=18,help="Positional encoding")
    parser.add_argument("--gradient_accumulation_steps",type=int, default=1,help="Steps for gradient calculation")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--num_trans", type=int, default=2, help="number of parallel transformers")

    args = parser.parse_args()
    return args
