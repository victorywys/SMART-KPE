import collections
import csv
import os, sys
import numpy as np
import logging
import random
import json
import re, string
import argparse
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
from transformers import BertTokenizer, BertConfig

from parser import get_parser
from data_processor import load_and_cache_examples, get_predicted_keyphrase
from model import BLING_KPE
from evaluation import evaluate_kps
logger = logging.getLogger(__name__)


def set_seed():
    random.seed(11747)
    np.random.seed(11747)
    torch.manual_seed(11747)
    torch.cuda.manual_seed_all(11747)


def train(args, model, dataloaders, examples, loss_func, optimizer):
    dataloader = dataloaders['train']

    tb_writer = SummaryWriter()
    batch_size = args.batch_size

    t_total = len(dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    dev_loader = dataloaders['dev']
    dev_example = examples['dev']

    # Training
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(dataloader))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    #train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch")
    #train_iterator = range(int(args.num_train_epochs))

    best_result = -1.
    accumulated_loss = 0.
    for epoch in range(epochs_trained, int(args.num_train_epochs)):
        #epoch_iterator = tqdm(dataloader, desc="Iteration")

        #for step, batch in enumerate(epoch_iterator):
        for step, batch in enumerate(dataloader):
            model.train()
            text_id = batch[0].to(args.device,dtype=torch.long) # [BS, SENT_LEN]
            visual_input = batch[1].to(args.device,dtype=torch.float) # [BS, SENT_LEN, 18]
            labels = batch[2].to(args.device,dtype=torch.long) # [BS, SENT_LEN]
            input_mask = batch[3].to(args.device,dtype=torch.bool) # [BS, SENT_LEN]
            valid_id = batch[4].to(args.device,dtype=torch.bool) # [BS, SENT_LEN]
            meta = batch[5].to(args.device, dtype=torch.float) # [BS, META_DIM]

            pred_outputs = model(text_id, visual_input, input_mask, meta)
            valid_id = valid_id.view(-1)
            pred_outputs = pred_outputs.view(-1,args.tag_num)[valid_id]
            labels = labels.view(-1)[valid_id]
            loss = loss_func(pred_outputs,labels)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            tr_loss += loss.item()
            accumulated_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                print(f"[{epoch}, {step + 1}] loss: {(accumulated_loss):.6f}")
                accumulated_loss = 0
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.evaluate_during_training: # Only evaluate when single GPU otherwise metrics may not average well
                        print(f"evaluating at global_step {global_step}:")
                        results = evaluate(args, model, dev_loader, dev_example, num_to_evaluate=args.evaluate_num, print_name="Trying")
                        for key, value in results.items():
                            print(f"\t{key}: {value:.6f}")
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                        if results[args.main_metric] > best_result and args.save_best:
                            print("Saving best model ...")
                            best_result = results[args.main_metric]
                            output_dir = os.path.join(args.output_dir, "checkpoint-best")
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            torch.save(args, os.path.join(output_dir, "training_args.bin"))
                            torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    # TODO: add learning decay scheduler
                    # tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss


                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # TODO: should save current epoch and step for restoring
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    print(output_dir, os.path.exists(output_dir))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))

            if args.max_steps > 0 and global_step > args.max_steps:
                break

        # print(f"evaluating on training set for epoch {epoch}:")
        # results, _ = evaluate(args, model, dataloader, num_to_evaluate=args.evaluate_num)
        # for key, value in results.items():
        #     print(f"\t{key}: {value}")

        if args.max_steps > 0 and global_step > args.max_steps:
            break

    tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, dataloader, examples, calc_f1=True, num_to_evaluate=-1, print_name=None):
    true_ans_ls, pred_ans_ls = [], []
    batch_size = args.batch_size
    if (print_name is not None) and (args.print_dir is not None):
        if not os.path.exists(args.print_dir):
            os.makedirs(args.print_dir)
        f = open(os.path.join(args.print_dir,print_name+"_predict.json"),'w')
    model.eval()
    loss = 0
    all_outputs = None
    all_valid_ids = None
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            text_id = batch[0].to(args.device, dtype=torch.long)
            visual_input = batch[1].to(args.device, dtype=torch.float)
            labels = batch[2].to(args.device,dtype=torch.long)
            input_mask = batch[3].to(args.device,dtype=torch.bool)
            valid_id = batch[4].to(args.device,dtype=torch.bool) # [BS, SENT_LEN]
            meta = batch[5].to(args.device, dtype=torch.float)
            pred_outputs = model(text_id, visual_input, input_mask, meta)

            valid_id_for_loss = valid_id.view(-1)
            pred_outputs_for_loss = pred_outputs.view(-1,args.tag_num)[valid_id_for_loss]
            labels = labels.view(-1)[valid_id_for_loss]
            #loss += F.cross_entropy(pred_outputs_for_loss,labels).item()
            loss += F.nll_loss(pred_outputs_for_loss, labels).item()
            #pred_outputs = F.softmax(pred_outputs, dim=-1)
            pred_outputs = torch.exp(pred_outputs)
            if all_outputs is None:
                all_outputs = pred_outputs
                all_valid_ids = valid_id
            else:
                all_outputs = torch.cat([all_outputs, pred_outputs], 0)
                all_valid_ids = torch.cat([all_valid_ids, valid_id], 0)

            if num_to_evaluate>0 and all_outputs.shape[0]>num_to_evaluate:
                break
    print(f"Evaluation loss: {loss/all_outputs.shape[0]}")

    all_outputs_arr = all_outputs.to('cpu').data.numpy()
    all_valid_ids_arr = all_valid_ids.to('cpu').data.numpy()
    for idx in range(all_outputs_arr.shape[0]):
        example = examples[idx]
        pred_logits = all_outputs_arr[idx].reshape(args.max_text_length,args.tag_num)
        valid_id = all_valid_ids_arr[idx]
        single_pred_dic = {}
        single_pred_dic['url'] = example.url
        single_pred_dic['text'] = example.text[:args.max_text_length]
        single_pred_dic['true_kp'] = example.keyphrase
        single_pred_dic['KeyPhrases'] = get_predicted_keyphrase(args, single_pred_dic, pred_logits, valid_id)

        if (print_name is not None) and (args.print_dir is not None):
            j_dict = {'KeyPhrases':[], 'url':example.url}
            for phrase in single_pred_dic['KeyPhrases']:
                j_dict['KeyPhrases'].append(phrase.split())
            json.dump(j_dict,f)
            f.write("\n")
        true_ans_ls.append(example.keyphrase)
        pred_ans_ls.append(single_pred_dic['KeyPhrases'])

    result = evaluate_kps(true_ans_ls, pred_ans_ls)
    print(result)
    return result

def main():
    args = get_parser()
    print(args)
    args.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    args.tokenizer.add_special_tokens({'additional_special_tokens':['[TITLE]']})
    if not (args.train or args.test or args.dev):
        raise NotImplementedError("Need to define task!")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

#    set_seed()

    # Data Prep #
    dataset_ls, example_ls = load_and_cache_examples(args)
    data_loader_dict, example_dict = dict(), dict()
    train_loader = DataLoader(dataset_ls[0],
                              sampler=RandomSampler(dataset_ls[0]),
                              batch_size=args.batch_size)
    dev_loader = DataLoader(dataset_ls[1],
                            sampler=SequentialSampler(dataset_ls[1]),
                            batch_size=args.batch_size)
    test_loader = DataLoader(dataset_ls[2],
                            sampler=SequentialSampler(dataset_ls[2]),
                            batch_size=args.batch_size)
    data_loader_dict['train'] = train_loader
    data_loader_dict['dev'] = dev_loader
    data_loader_dict['test'] = test_loader
    example_dict['dev'] = example_ls[1]
    example_dict['test'] = example_ls[2]

    # Model prep #
    device = torch.device("cuda" if torch.cuda.is_available() and not (args.device=="cpu") else "cpu")
    args.device = device
    if args.from_checkpoint is not None:
        old_args = torch.load(os.path.join(args.from_checkpoint, "training_args.bin"))
        check_diff = ["max_text_length", "visual_size"]
        for prop in check_diff:
            if args.__dict__[prop] != old_args.__dict__[prop]:
                print(f"Argument {prop} should be consistent:")
                print(f"Loaded {prop}: {old_args.__dict__[prop]}")
                print(f"New {prop}: {args.__dict__[prop]}")
                raise NotImplementedError

    model = BLING_KPE(args)
    model = torch.nn.DataParallel(model)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    # TODO: 150 should be a hyperparameter
    # loss_func = nn.CrossEntropyLoss(weight=torch.tensor([1.,1.,1.,1.,1.]).to(device))
    loss_func = nn.NLLLoss()

    if args.from_checkpoint is not None:
        # Load part of the state dict:
        pretrained_model = torch.load(os.path.join(args.from_checkpoint, "model.pt"))
        tot_mat_num = len(pretrained_model)
        model_dict = model.state_dict()
        pretrained_model = {k: v for k, v in pretrained_model.items() if k in model_dict and model_dict[k].size() == v.size()}
        load_mat_num = len(pretrained_model)
        model_dict.update(pretrained_model)
        model.load_state_dict(model_dict)
        print(f"Successfully loaded pretrained parameters. Total: {tot_mat_num}, Loaded: {load_mat_num}")
        # Load part of the state dict where size matches.
        if tot_mat_num == load_mat_num:
            pretrained_optimizer = torch.load(os.path.join(args.from_checkpoint, "optimizer.pt"))
            optimizer.load_state_dict(pretrained_optimizer)

    # Find total parameters and trainable parameters
#    total_params = sum(p.numel() for p in model.parameters())
#    print(f'{total_params:,} total parameters.')
#    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#    print(f'{total_trainable_params:,} training parameters.')

    # print(model)

    if args.train:
        global_step, tr_loss = train(args, model, data_loader_dict, example_dict, loss_func, optimizer)
        print(global_step, tr_loss)

    if args.dev:
        results = evaluate(args, model, data_loader_dict['dev'], example_dict['dev'], calc_f1=True, print_name='dev')
        if not args.only_pred:
            print("Evaluation results on validation set:")
            for key in results:
                print(f"{key}: {results[key]}")
        # TODO if anyone wants: present or save the predicted keyphrases

    if args.test:
        results = evaluate(args, model, data_loader_dict['test'], example_dict['test'], calc_f1=False, print_name='test')
        # TODO if anyone wants: present or save the predicted keyphrases


if __name__ == "__main__":
    main()
