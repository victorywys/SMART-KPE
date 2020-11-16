import torch
import logging
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, NLLLoss
from ..transformers import RobertaForTokenClassification

logger = logging.getLogger()


class RobertaForSeqTagging(RobertaForTokenClassification):

    def forward(self, visual_input, meta_input, input_ids, attention_mask, valid_ids, active_mask, valid_output, labels=None):
        
        # --------------------------------------------------------------------------------
        # Bert Embedding Outputs

        outputs = self.roberta(input_ids=input_ids,
                               attention_mask=attention_mask)

        attention_mask = attention_mask.to(torch.bool)

        sequence_output = outputs[0]  # batch * len * bert_size

        # --------------------------------------------------------------------------------
        # Meta-feature Predictor
        bert_cls = sequence_output[:, 0, :].squeeze(1)  # batch * bert_size
        meta_cat = torch.cat([bert_cls, meta_input], -1)
        pred_mask_before_softmax = self.meta_selector(meta_cat)
        pred_mask = F.softmax(pred_mask_before_softmax, -1).unsqueeze(-1).unsqueeze(-1)

        # --------------------------------------------------------------------------------
        # Visual Embedding Outputs
        visual_t = visual_input.transpose(0, 1)
        visual_embedding = self.visual_trans(visual_t,
                                             src_key_padding_mask=(~attention_mask)).transpose(0, 1)

        embedding = torch.cat([sequence_output, visual_embedding], -1).transpose(0, 1)  # len * batch * embed_size

        phrase_embedding = self.phrase_trans(embedding,
                                             src_key_padding_mask=(~attention_mask)).transpose(0, 1)

        # --------------------------------------------------------------------------------
        # Valid Outputs : get first token vector  
        batch_size = phrase_embedding.size(0)
        for i in range(batch_size):
            valid_num = sum(valid_ids[i]).item()

            vectors = phrase_embedding[i][valid_ids[i] == 1]
            valid_output[i, :valid_num].copy_(vectors)

        # --------------------------------------------------------------------------------
        # Dropout
        phrase_embedding = self.dropout(valid_output)
        logits = torch.cat(
            [self.tag_prediction[i](phrase_embedding).unsqueeze(1)
             for i in range(4)],
            1
        )
        pred = F.softmax(logits, -1)
        pred = torch.log(torch.sum(pred * pred_mask, 1))

        
        # --------------------------------------------------------------------------------
        # Active Logits
        active_loss = active_mask.view(-1) == 1 # [False, True, ...]
        active_logits = pred.view(-1, self.num_labels)[active_loss] # False

        if labels is not None:
            loss_fct = NLLLoss()
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            return loss
        else:
            return active_logits