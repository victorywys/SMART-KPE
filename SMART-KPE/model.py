import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import *

def positional_encoding(max_len, dim):
    a = np.arange(max_len)[:, np.newaxis]
    b = np.arange(dim)[np.newaxis, :]
    angle = a / np.power(10000., (b // 2 * 2) / 10)
    sines = np.sin(angle[:, 0::2])
    cosines = np.cos(angle[:, 1::2])
    return np.concatenate([sines, cosines], -1) # max_len * dim

class BLING_KPE(nn.Module):
    def __init__(self, args):
        super(BLING_KPE, self).__init__()
        # TODO: initializing all the parameters
        self.args = args
        self.BERT = BertModel.from_pretrained('bert-base-uncased')
        self.BERT.resize_token_embeddings(len(args.tokenizer))
        embed_size = args.bert_size + args.visual_size
        visual_trans_layer = nn.TransformerEncoderLayer(d_model=args.visual_size, nhead=3)
        self.visual_trans = nn.TransformerEncoder(visual_trans_layer, num_layers=2)

        self.meta_dim = 0
        self.meta_dim += args.bert_size
        if args.use_snapshot:
            self.meta_dim += args.snapshot_dim
        assert self.meta_dim > 0, "At least one of the meta data should be used"

        self.meta_selector = nn.Sequential(
                nn.Linear(self.meta_dim, 256),
                nn.ReLU(),
                nn.Linear(256, args.num_trans),
            )

        phrase_trans_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=6)
        self.phrase_trans = nn.TransformerEncoder(phrase_trans_layer, num_layers=2)

        #self.pos_embed = torch.from_numpy(positional_encoding(args.max_text_length, args.positional_size)).to(args.device,dtype=torch.float)

        self.tag_prediction = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_size, 128),
                nn.ReLU(),
                nn.Linear(128, args.tag_num),
            ) for i in range(args.num_trans)])

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, text_id, visual_input, input_mask, meta, valid_id=None):
        """
        text_id: batch * len
        position_input: batch * len * pos_size
        visual_input: batch * len * visual_size
        meta: batch * meta_dim
        """
        bert_embedding,_ = self.BERT(text_id, attention_mask = input_mask)
        batch, length, _ = bert_embedding.size()

        bert_cls = bert_embedding[:, 0, :].squeeze(1) # batch * bert_size

        visual_t = visual_input.transpose(0, 1) # len * batch * visual_size
        visual_embedding = self.visual_trans(visual_t, src_key_padding_mask=(~input_mask)).transpose(0, 1) # batch * len * visual_size

        embedding = torch.cat([bert_embedding, visual_embedding], -1).transpose(0, 1) #  len * batch * embed_size

        phrase_embedding = self.phrase_trans(embedding, src_key_padding_mask=(~input_mask)).transpose(0, 1)
        '''
        batch_size = phrase_embedding.size(0)
        for i in range(batch_size):
            valid_num = sum(valid_id[i]).item()
            vectors = phrase_embedding[i][valid_id[i]==1]
            phrase_embedding[i,:valid_num].copy_(vectors)
            phrase_embedding[i,valid_num:] = 0
        '''
        pred_before_softmax = torch.cat([self.tag_prediction[i](phrase_embedding).unsqueeze(1) for i in range(self.args.num_trans)], 1) # batch * num_trans * len * tag_num

        meta = torch.mean(meta.view(batch, 4, -1), 1) # batch * 512
        meta_cat = torch.cat([bert_cls, meta], -1) # batch * meta_size
#        meta_cat = bert_cls
        pred_mask_before_softmax = self.meta_selector(meta_cat) # batch * num_trans
        pred_mask = F.softmax(pred_mask_before_softmax, -1).unsqueeze(-1).unsqueeze(-1) # batch * num_trans * 1 * 1
        pred = F.softmax(pred_before_softmax, -1) # batch * num_trans * len * tag_num
        pred = torch.log(torch.sum(pred * pred_mask, 1)) # batch * len * tag_num
        return pred
