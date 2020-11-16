import collections
import csv
import os, sys
import numpy as np

import logging
import json
import argparse
import pickle as pkl
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import unicodedata
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

TAG = {'O':0, 'B':1, 'I':2, 'E':3, 'U':4 }

class InputExample(object):
    def __init__(self, url, text, VDOM, kp):
        self.url = url # URL
        self.text = text # List of tokens
        self.VDOM = VDOM # List of dics
        self.keyphrase = kp # List of KPs

class BlingFeature(object):
    def __init__(self, text, vis_arr, meta, label_arr):
        self.text = text
        self.vis_arr = vis_arr
        self.meta = meta
        self.label_arr = label_arr

    def convert_to_bert_features(self, tokenizer, max_length):
        #### REMEMBER TO CHECK THIS FOR SERVER!!!! ####
        self.text_id = tokenizer.tokenize(" ".join(self.text))
        self.valid_id = []
        new_vis_arr, new_label_arr = [],[]
        orig_token_ptr = 0
        cur_string = ""
        for idx, token in enumerate(self.text_id):
            if cur_string == "":
                self.valid_id.append(1)
            else:
                self.valid_id.append(0)
            new_vis_arr.append(self.vis_arr[orig_token_ptr])
            new_label_arr.append(self.label_arr[orig_token_ptr])
            if token.startswith("##"):
                token = token[2:]
            cur_string += token
            if cur_string == self.text[orig_token_ptr]:
                orig_token_ptr += 1
                cur_string = ""

        l = len(self.text_id)
        if l > max_length-2:
            l = max_length-2

        ''' The format is [CLS] A B C D E [SEP] [PAD] ... '''

        self.text_id = ["[CLS]"] +self.text_id[:l]+ ["[SEP]"] + ["[PAD]" for i in range(max_length-l-2)]
        self.valid_id = [0] + self.valid_id[:l] + [0] + [0 for i in range(max_length-l-2)]
        self.input_mask = [1] + [1 for i in range(l)] + [1] +[0 for i in range(max_length-l-2)]
        S = new_vis_arr[0].shape
        self.vis_arr = [np.zeros(S)] + new_vis_arr[:l] + [np.zeros(S)] + [np.zeros(S) for i in range(max_length-l-2)]
        self.label_arr = [0] + new_label_arr[:l] + [0] + [0 for i in range(max_length-l-2)]

        self.text_id = torch.tensor(tokenizer.convert_tokens_to_ids(self.text_id))
        self.valid_id = torch.tensor(self.valid_id)
        self.input_mask = torch.tensor(self.input_mask)
        self.vis_arr = torch.tensor(self.vis_arr)
        self.label_arr = torch.tensor(self.label_arr)
        self.meta = torch.tensor(self.meta)

def convert_input_to_features(args, data_dir, split="train", is_test=False, return_examples=False):
    def transform_dic_to_example(dic):
        url = dic['url']
        text = dic['text'].strip().lower().split(" ")[:args.max_text_length]
        VDOM = []
        if args.include_title:
            title = dic['title'].strip().lower().split(" ") + ["[title]"]
            l_add = len(title)
        else:
            title = []
            l_add = 0
        text = title + text
        text = text[:args.max_text_length]
        for vdom_dic in json.loads(dic['VDOM']):
            VDOM.append({'feature':vdom_dic['feature'],
                         'start_idx':vdom_dic['start_idx']+l_add,
                         'end_idx':vdom_dic['end_idx']+l_add})
        if is_test:
            kp = ["_"]
            return InputExample(url, text, VDOM, kp)
        else:
            kp = dic['KeyPhrases']
            kp = [[t.lower() for t in phrase] for phrase in kp]
            final_kp = [" ".join(phrase) for phrase in kp if len(phrase)>0]
            return InputExample(url,text,VDOM,final_kp)

    def get_visual_features(VDOM):
        ret_arr = np.zeros((args.max_text_length,18), dtype=np.float)
        for vdom_dic in VDOM:
            st, ed = vdom_dic['start_idx'], vdom_dic['end_idx']
            if st>args.max_text_length:
                break
            if ed>args.max_text_length:
                ed=args.max_text_length
            feature = vdom_dic['feature']
            ret_arr[st:ed,:] = np.array(feature[0:6]+feature[7:16]+feature[17:20])
        return ret_arr

    def get_answer_features(kp_ls, text):
        if len(text) > args.max_text_length:
            text = text[:args.max_text_length]
        l = len(text)
        kplabel_arr = [TAG['O'] for i in range(l)]
        for kp in kp_ls:
            kplen = len(kp.strip().split(" "))
            if kplen == 0:
                continue
            for idx in range(l - kplen + 1):
                if " ".join(text[idx:idx+kplen]) == kp:
                    # TAGGING #
                    if kplen == 1:
                        kplabel_arr[idx] = TAG['U']
                    else:
                        kplabel_arr[idx] = TAG['B']
                        kplabel_arr[idx+kplen-1] = TAG['E']
                        for i in range(idx+1, idx+kplen-1):
                            kplabel_arr[i] = TAG['I']
        return kplabel_arr

    def get_meta_features(args, split, line_id):
        ret = None
        if args.use_snapshot:
            f_name = os.path.join(args.meta_dir, "snapshot", f"{split}_res", f"{line_id}.npy")
            if os.path.isfile(f_name):
                snapshot_vec = np.load(f_name)
            else:
                snapshot_vec = np.zeros((args.snapshot_dim))
            ret = snapshot_vec
        else:
            ret = np.array([])
        return ret


    feature_ls = []
    example_ls = []
    with open(os.path.join(data_dir, f"{split}.jsonl")) as f:
        for line_id, line in enumerate(f):
            ### get input example ###
            dic = json.loads(line)
            example = transform_dic_to_example(dic)
            if return_examples:
                example_ls.append(example)
            if len(example.text)<5:
                continue
            if len(example.keyphrase)==0:
                continue
            ### transform to feature ###
            visual_arr = get_visual_features(example.VDOM) #[Sent_Len, 18]
            answer_arr = get_answer_features(example.keyphrase, example.text)
            meta = get_meta_features(args, split, line_id)
            bling_feat = BlingFeature(example.text,visual_arr,meta,answer_arr)
            bling_feat.convert_to_bert_features(args.tokenizer, args.max_text_length)
            feature_ls.append(bling_feat)
            if(len(feature_ls)%3000 == 0):
                print(f"Processed {len(feature_ls)} features.")
    if return_examples:
        return feature_ls, example_ls
    else:
        return feature_ls

def load_and_cache_examples(args):

    # Load data features from cache or dataset file
    cached_features_dir = args.cached_features_dir
    if args.read_from_cached_features:
        print("Loading features from cache dir:", cached_features_dir)
        if args.train:
            train_features = pkl.load(open(os.path.join(cached_features_dir,"train.pkl"),"rb"))
        if args.dev or args.evaluate_during_training:
            dev_features = pkl.load(open(os.path.join(cached_features_dir,"dev.pkl"),"rb"))
            dev_examples = pkl.load(open(os.path.join(cached_features_dir,"dev-examples.pkl"),"rb"))
        if args.test:
            test_features = pkl.load(open(os.path.join(cached_features_dir,"test.pkl"),"rb"))
            test_examples = pkl.load(open(os.path.join(cached_features_dir,"test-examples.pkl"),"rb"))
    else:
        print("Creating features from dataset dir at:", args.data_dir)
        print("Saving features into cache dir:", cached_features_dir)
        if args.train:
            print("Processing training set.")
            train_features = convert_input_to_features(
                                            args,
                                            args.data_dir,
                                            split="train",
                                            is_test=False,
                                            return_examples=False)
            pkl.dump(train_features, open(os.path.join(cached_features_dir,"train.pkl"),'wb'))

        if args.dev or args.evaluate_during_training:
            print("Processing validation set.")
            dev_features, dev_examples = convert_input_to_features(
                                            args,
                                            args.data_dir,
                                            split="dev",
                                            is_test=False,
                                            return_examples=True)
            pkl.dump(dev_features, open(os.path.join(cached_features_dir,"dev.pkl"),'wb'))
            pkl.dump(dev_examples, open(os.path.join(cached_features_dir,"dev-examples.pkl"),'wb'))

        if args.test:
            print("Processing test set.")
            test_features, test_examples = convert_input_to_features(
                                            args,
                                            args.data_dir,
                                            split="test",
                                            is_test=True,
                                            return_examples=True)
            pkl.dump(test_features, open(os.path.join(cached_features_dir,"test.pkl"),'wb'))
            pkl.dump(test_examples, open(os.path.join(cached_features_dir,"test-examples.pkl"),'wb'))
    if args.train:
        for i in range(3):
            feat = train_features[i]
            print("TEXT:\t", feat.text[:100])
            print("LABELS:\t", feat.label_arr[:100])
            print("INPUT_MASK:\t", feat.input_mask[:100])
            print("VALID_ID:\t", feat.valid_id[:100])

    # Convert to Tensors and build dataset
    dataset_ret = []
    example_ret = []
    if args.train:
        train_dataset = SentenceDataset(train_features,args)
        dataset_ret.append(train_dataset)
        example_ret.append("_")
    else:
        dataset_ret.append("_")
        example_ret.append("_")
    if args.dev or args.evaluate_during_training:
        dev_dataset = SentenceDataset(dev_features,args)
        dataset_ret.append(dev_dataset)
        example_ret.append(dev_examples)
    else:
        dataset_ret.append("_")
        example_ret.append("_")
    if args.test:
        test_dataset = SentenceDataset(test_features,args)
        dataset_ret.append(test_dataset)
        example_ret.append(test_examples)
    else:
        dataset_ret.append("_")
        example_ret.append("_")

    return dataset_ret, example_ret

class SentenceDataset(Dataset):
    def __init__(self, feature_ls,args):
        self.max_text_length = args.max_text_length
        self.feature_ls = feature_ls
    def __len__(self):
        return len(self.feature_ls)
    def __getitem__(self, index):
        feature = self.feature_ls[index]
        return feature.text_id, feature.vis_arr, feature.label_arr, feature.input_mask, feature.valid_id, feature.meta

def get_predicted_keyphrase(args, dic, pred_logits, valid_id):
    ### pred_logits: [SENT_LEN, TAG_NUM]
    ### valid_id: [SENT_LEN]

    def overlap(a, b):
        K = 0.9
        a = set(a.strip().split(" "))
        b = set(b.strip().split(" "))
        overlap_set = a.intersection(b)
        if len(overlap_set) / len(b) > K:
            return True
        return False

    text = dic['text']
    arr = pred_logits[valid_id==1]
    tag = []
    L = arr.shape[0]
    for pos in range(L):
        tag.append(np.argmax(arr[pos]))
    candidate_phrase_set = set()
    candidate_phrase_scoredic = {}
    for p_len in range(1,6):
        for start_pos in range(L-p_len):
            end_pos = start_pos + p_len
            candidate_phrase = " ".join(text[start_pos:end_pos])
            if len(candidate_phrase.strip()) == 0:
                continue
            flag = True
            if p_len == 1:
                phrase_score = arr[start_pos,TAG['U']]
                flag = flag and (tag[start_pos] == TAG['U'])
            else:
                flag = flag and (tag[start_pos] == TAG['B'])
                flag = flag and (tag[start_pos + p_len - 1] == TAG['E'])
                for mid_pos in range(start_pos + 1, end_pos - 1):
                    flag = flag and (tag[mid_pos] == TAG['I'])
                candidate_score_ls = [arr[start_pos,TAG['B']]] + [arr[start_pos+j,TAG['I']] for j in range(1,p_len-1)] + [arr[start_pos+p_len-1,TAG['E']]]
                phrase_score = min(candidate_score_ls)
            if not flag:
                continue
            if candidate_phrase in candidate_phrase_set:
                if candidate_phrase_scoredic[candidate_phrase] < phrase_score:
                    candidate_phrase_scoredic[candidate_phrase] = phrase_score
            else:
                candidate_phrase_set.add(candidate_phrase)
                candidate_phrase_scoredic[candidate_phrase] = phrase_score
    final_ls = sorted([(p, candidate_phrase_scoredic[p]) for p in candidate_phrase_scoredic],key=lambda x:x[1], reverse=True)

    if args.filter_predicted_kp:
        ret_ls = []
        kp_ptr = 0
        while(len(ret_ls)<5 and kp_ptr<len(final_ls)):
            cur_kp = final_ls[kp_ptr][0]
            flag = True
            for kp in ret_ls:
                if overlap(kp, cur_kp):
                    flag = False
                    break
            if flag:
                ret_ls.append(cur_kp)
            kp_ptr += 1
    else:
        ret_ls = [tup[0] for tup in final_ls[:5]]
    return ret_ls


