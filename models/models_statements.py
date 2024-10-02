import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Dict
from .gnn_encoder import HypeTKGEncoder, HypeTKGBase
from utils.utils_gcn import get_param
from models.time_encoder import TimeEncode



class HypeTKG(HypeTKGEncoder):
    model_name = 'HypeTKG'

    def __init__(self, kg_graph_repr: Dict[str, np.ndarray], config: dict, id2e: tuple = None):
        if id2e is not None:
            super(self.__class__, self).__init__(kg_graph_repr, config, id2e[1])
        else:
            super(self.__class__, self).__init__(kg_graph_repr, config)

        self.model_name = 'HypeTKG'

        self.hid_drop2 = config['STAREARGS']['HID_DROP2']
        self.feat_drop = config['STAREARGS']['FEAT_DROP']
        self.num_transformer_layers = config['STAREARGS']['T_LAYERS']
        self.num_heads = config['STAREARGS']['T_N_HEADS']
        self.num_hidden = config['STAREARGS']['T_HIDDEN']
        self.d_model = config['EMBEDDING_DIM']
        self.positional = config['STAREARGS']['POSITIONAL']
        self.p_option = config['STAREARGS']['POS_OPTION']
        self.pooling = config['STAREARGS']['POOLING']  # min / avg / concat
        self.embed_statics = config['SAMPLER_W_STATICS']
        self.embed_qualifiers = config['SAMPLER_W_QUALIFIERS']

        self.hidden_drop = torch.nn.Dropout(self.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.feat_drop)

        encoder_layers_qualifier = TransformerEncoderLayer(self.d_model, self.num_heads, self.num_hidden, config['STAREARGS']['HID_DROP2'])
        encoder_layers_static = TransformerEncoderLayer(self.d_model, self.num_heads, self.num_hidden, config['STAREARGS']['HID_DROP2'])
        encoder_layers = TransformerEncoderLayer(self.d_model, self.num_heads, self.num_hidden, config['STAREARGS']['HID_DROP2'])
        self.encoder = TransformerEncoder(encoder_layers, config['STAREARGS']['T_LAYERS'])
        self.encoder_qualifier = TransformerEncoder(encoder_layers_qualifier, config['STAREARGS']['T_LAYERS'])
        self.encoder_static = TransformerEncoder(encoder_layers_static, config['STAREARGS']['T_LAYERS'])
        # self.position_embeddings = nn.Embedding(config['MAX_QPAIRS'] - 1, self.d_model)
        self.qual_position_emb = nn.Embedding(config['len_qualifier']+1, self.d_model)
        self.sta_position_emb = nn.Embedding(config['len_static']+1, self.d_model)

        self.layer_norm = torch.nn.LayerNorm(self.emb_dim)

        self.qual_cls = get_param((1, self.emb_dim))
        self.sta_cls = get_param((1, self.emb_dim))
        self.cls = get_param((1, self.emb_dim))

        self.w_srware = get_param((2 * self.emb_dim, self.emb_dim))
        self.w_feature = get_param((self.emb_dim, 2 * self.emb_dim))
        self.w_feature_no_static = get_param((self.emb_dim, self.emb_dim))
        self.linear_layer = torch.nn.Linear(2 * self.emb_dim, self.emb_dim)
        self.w_qual_pairs = get_param((2 * self.emb_dim, self.emb_dim))


        if self.pooling == "concat":
            self.flat_sz = self.emb_dim * (config['MAX_QPAIRS'] - 1)
            self.fc = torch.nn.Linear(self.flat_sz, self.emb_dim)
        else:
            self.fc = torch.nn.Linear(self.emb_dim, self.emb_dim)

    def concat(self, e1_embed, rel_embed, quals=None, statics=None):
        e1_embed = e1_embed.view(-1, 1, self.emb_dim)
        rel_embed = rel_embed.view(-1, 1, self.emb_dim)
        """
            arrange quals in the conve format with shape [bs, num_qual_pairs, emb_dim]
            num_qual_pairs is 2 * (any qual tensor shape[1])
            for each datum in bs the order will be 
                rel1, emb
                en1, emb
                rel2, emb
                en2, emb
        """
        if statics is not None:
            if quals is not None:
                stack_inp = torch.cat([e1_embed, rel_embed, quals, statics], 1).transpose(1, 0)  # [2 + num_qual_pairs, bs, emb_dim]
            else:
                stack_inp = torch.cat([e1_embed, rel_embed, statics], 1).transpose(1, 0)  # [2 + num_qual_pairs, bs, emb_dim]
        else:
            if quals is not None:
                stack_inp = torch.cat([e1_embed, rel_embed, quals], 1).transpose(1, 0)  # [2 + num_qual_pairs, bs, emb_dim]
            else:
                stack_inp = torch.cat([e1_embed, rel_embed], 1).transpose(1, 0)  # [2 + num_qual_pairs, bs, emb_dim]
        return stack_inp

    def forward(self, sub, rel, time, quals=None):
        '''


        :param sub: bs
        :param rel: bs
        :param quals: bs*(sl-2) # bs*14
        :return:


        '''
        sub_emb, rel_emb, qual_emb, static_emb, all_ent, qual_mask, static_mask, all_qual_emb, qual_sub_mask, time_emb = \
            self.forward_base(sub, rel, time, self.hidden_drop, self.feature_drop, quals, True)

        # bs*emb_dim , ......, bs*6*emb_dim
        # stk_inp = self.concat(sub_emb, rel_emb, qual_emb, static_emb)

        if self.embed_qualifiers:
            qual_cls = self.qual_cls.expand_as(sub_emb)
            # qual_cls = self.qual_cls.repeat(qual_emb.shape[0], 1)
            qual_cls = qual_cls.unsqueeze(1)
            qual_emb = torch.cat((qual_cls, qual_emb), dim=1).transpose(0, 1)

            if self.positional:
                positions = torch.arange(qual_emb.shape[0], dtype=torch.long, device=self.device).repeat(qual_emb.shape[1], 1)
                pos_embeddings = self.qual_position_emb(positions).transpose(1, 0)
                qual_emb = qual_emb + pos_embeddings

            qualifier = self.encoder_qualifier(qual_emb, src_key_padding_mask=qual_mask)[0] # [B,D]

        if self.embed_statics:
            sta_cls = self.sta_cls.expand_as(static_emb[:,1,:])
            # sta_cls = self.sta_cls.repeat(static_emb.shape[0], 1)
            sta_cls = sta_cls.unsqueeze(1)
            static_emb = torch.cat((sta_cls, static_emb), dim=1).transpose(0, 1)

            if self.positional:
                positions = torch.arange(static_emb.shape[0], dtype=torch.long, device=self.device).repeat(
                    static_emb.shape[1], 1)
                pos_embeddings = self.sta_position_emb(positions).transpose(1, 0)
                static_emb = static_emb + pos_embeddings

            static_all = self.encoder_static(static_emb, src_key_padding_mask=static_mask)[0] # [B,D]
            static = torch.index_select(static_all, 0, sub)

        sub_emb = sub_emb.unsqueeze(1)  # [B, 1, D]
        rel_emb = rel_emb.unsqueeze(1)  # [B, 1, D]
        # sr_emb = torch.cat((sub_emb, rel_emb), dim=1) # [B,2,D]
        cls = self.cls.expand_as(sub_emb.squeeze(1))
        cls = cls.unsqueeze(1)
        if self.embed_statics and self.embed_qualifiers:
            qualifier = qualifier.unsqueeze(1)  # [B,1,D]
            static = static.unsqueeze(1)  # [B,1,D]
            # sr_qual_attention
            # sr_aware

            sr_emb = torch.cat((sub_emb.squeeze(1), rel_emb.squeeze(1)), dim=1)  # [B, 2D]
            sr_aware = torch.mm(sr_emb, self.w_srware)  # [B, 2xD] X [2D, D] ==> [B, D]
            qual_aware_emb = torch.mm(all_qual_emb, self.w_qual_pairs)  # [N_qual_pairs, 2xD] x [2D, D] ==> [N_qual_pairs, D]
            attention_score = torch.matmul(sr_aware, qual_aware_emb.t())  # [B, N_qual_pairs]

            # Obtain the attention_score with mask and do the softmax operation.
            masked_attention_score = attention_score.masked_fill(qual_sub_mask.bool(), float('-inf'))
            attention_weights = F.softmax(masked_attention_score, dim=1)
            attention_weights = torch.where(torch.isnan(attention_weights),
                                            torch.tensor(0.0, device=attention_weights.device), attention_weights)
            # ### print attention ###
            # nonzero_indices = attention_weights.nonzero()
            # nonzero_values = attention_weights[nonzero_indices[:, 0], nonzero_indices[:, 1]]
            # ### print end ###

            # [B, 1, N]  [1, N, D]
            sr_aware_qual_pairs_emb = torch.matmul(attention_weights.unsqueeze(1),
                                                   qual_aware_emb.unsqueeze(0))  # [B,1,D]
            # aggregate all info to transformer
            feature_aware = torch.cat((cls, sub_emb, rel_emb, qualifier, static, sr_aware_qual_pairs_emb), dim=1)  # [B, 6, D]
            feature_mask =  torch.zeros((feature_aware.shape[0], feature_aware.shape[1])).bool().to(self.device)
            feature_mask = torch.all(feature_aware == 0, dim=2)
            ## transformer
            feature_aware = self.encoder(feature_aware.transpose(0, 1), src_key_padding_mask=feature_mask)[0]  # [B, D]
        elif self.embed_qualifiers:
            qualifier = qualifier.unsqueeze(1)  # [B,1,D]
            # sr_qual_attention

            # sr_aware
            sr_emb = torch.cat((sub_emb.squeeze(1), rel_emb.squeeze(1)), dim=1)  # [B, 2D]
            sr_aware = torch.mm(sr_emb, self.w_srware)  # [B, 2xD] X [2D, D] ==> [B, D]
            qual_aware_emb = torch.mm(all_qual_emb, self.w_qual_pairs)  # [N_qual_pairs, D]
            attention_score = torch.matmul(sr_aware, qual_aware_emb.t())  # [B, N_qual_pairs]

            # Obtain the attention_score with mask and do the softmax operation.
            masked_attention_score = attention_score.masked_fill(qual_sub_mask.bool(), float('-inf'))
            attention_weights = F.softmax(masked_attention_score, dim=1)
            attention_weights = torch.where(torch.isnan(attention_weights),
                                            torch.tensor(0.0, device=attention_weights.device), attention_weights)

            # Obtain the attention_score with mask and do the softmax operation.
            attention_weights = attention_weights * qual_sub_mask.float()
            # [B, 1, N]  [1, N, D]
            sr_aware_qual_pairs_emb = torch.matmul(attention_weights.unsqueeze(1),
                                                   qual_aware_emb.unsqueeze(0))  # [B,1,D]
            ## decode_matcher
            feature_aware = torch.cat((cls, sub_emb, rel_emb, qualifier, sr_aware_qual_pairs_emb),
                                      dim=1)  # [B, 6, D]
            feature_mask = torch.zeros((feature_aware.shape[0], feature_aware.shape[1])).bool().to(self.device)
            feature_mask = torch.all(feature_aware == 0, dim=2)
            feature_aware = self.encoder(feature_aware.transpose(0, 1), src_key_padding_mask=feature_mask)[0]  # [B, D]
        elif self.embed_statics:
            static = static.unsqueeze(1)  # [B,1,D]
            # aggregate all info to transformer
            feature_aware = torch.cat((cls, sub_emb, rel_emb, static),dim=1)  # [B, 6, D]
            feature_mask = torch.zeros((feature_aware.shape[0], feature_aware.shape[1])).bool().to(self.device)
            feature_mask = torch.all(feature_aware == 0, dim=2)
            ## transformer
            feature_aware = self.encoder(feature_aware.transpose(0, 1), src_key_padding_mask=feature_mask)[0]  # [B, D]
        else:
            feature_aware = torch.cat((cls, sub_emb, rel_emb),dim=1)  # [B, 3, D]
            feature_aware = self.encoder(feature_aware.transpose(0, 1))[0]  # [B, D]

        # Time
        feature_aware = feature_aware * time_emb

        x = torch.mm(feature_aware, self.w_feature_no_static) # (d, d)
        x = torch.mm(x, all_ent.transpose(1, 0))

        score = torch.sigmoid(x)
        return score
