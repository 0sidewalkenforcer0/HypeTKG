import torch
import numpy as np

from typing import Dict
from torch import nn
from torch.nn import Parameter
from utils.utils_gcn import get_param
from .gnn_layer import HypeTKGConvLayer
from models.time_encoder import TimeEncode


class HypeTKGBase(torch.nn.Module):
    def __init__(self, config):
        super(HypeTKGBase, self).__init__()
        """ Not saving the config dict bc model saving can get a little hairy. """

        self.act = torch.tanh if 'ACT' not in config['STAREARGS'].keys() \
            else config['STAREARGS']['ACT']
        self.bceloss = torch.nn.BCELoss()

        self.emb_dim = config['EMBEDDING_DIM']
        self.num_rel = config['NUM_RELATIONS']
        self.num_ent = config['NUM_ENTITIES']
        self.raw_num_ent = config['NUM_RAW_ENTITIES']
        self.n_bases = config['STAREARGS']['N_BASES']
        self.n_layer = config['STAREARGS']['LAYERS']
        self.gcn_dim = config['STAREARGS']['GCN_DIM']
        self.hid_drop = config['STAREARGS']['HID_DROP']
        # self.bias = config['STAREARGS']['BIAS']
        self.model_nm = config['MODEL_NAME'].lower()
        self.triple_mode = config['STATEMENT_LEN'] == 3
        self.qual_mode = config['STAREARGS']['QUAL_REPR']

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)


class HypeTKGEncoder(HypeTKGBase):
    def __init__(self, graph_repr: Dict[str, np.ndarray], config: dict, timestamps: dict = None):
        super().__init__(config)

        self.device = config['DEVICE']

        self.static_all = torch.tensor(config["STATIC_ALL"], dtype=torch.long, device= config['DEVICE'])
        self.embed_statics = config['SAMPLER_W_STATICS']
        self.embed_qualifiers = config['SAMPLER_W_QUALIFIERS']


        # Storing the KG
        self.edge_index = torch.tensor(graph_repr['edge_index'], dtype=torch.long, device=self.device)
        self.edge_type = torch.tensor(graph_repr['edge_type'], dtype=torch.long, device=self.device)
        self.edge_time = torch.tensor(graph_repr['edge_time'], dtype=torch.long, device=self.device)
        self.static_index = torch.tensor(graph_repr['static_index'], dtype=torch.long, device=self.device)
        self.static_type = torch.tensor(graph_repr['static_type'], dtype=torch.long, device=self.device)

        if not self.triple_mode:
            if self.qual_mode == "full":
                self.qual_rel = torch.tensor(graph_repr['qual_rel'], dtype=torch.long, device=self.device)
                self.qual_ent = torch.tensor(graph_repr['qual_ent'], dtype=torch.long, device=self.device)
            elif self.qual_mode == "sparse":
                self.quals = torch.tensor(graph_repr['quals'], dtype=torch.long, device=self.device)
                # self.quals_sub = torch.tensor(graph_repr['quals_sub'], dtype=torch.long, device=self.device)
                self.quals_pairs = self.get_unique_qualifier_pairs(self.quals[:2])
                # self.sub_qual_mask_dict= {
                #     torch.tensor(key, dtype=torch.long, device=self.device): torch.tensor(value, dtype=torch.long,
                #                 device=self.device) for key, value in graph_repr['quals_sub_mask'].items()
                # }
                self.sub_qual_mask_dict = graph_repr['quals_sub_mask']

        self.gcn_dim = self.emb_dim if self.n_layer == 1 else self.gcn_dim

        if timestamps is None:
            self.init_embed = get_param((self.num_ent, self.emb_dim))
            self.init_embed.data[0] = 0  # padding

        self.proj = nn.Linear(2 * self.gcn_dim, self.gcn_dim)
        self.time_encoder = TimeEncode(expand_dim=self.gcn_dim, entity_specific=False,
                                       num_entities=None, device=self.device)
        self.reset_parameters()

        if self.model_nm.endswith('transe'):
            self.init_rel = get_param((self.num_rel, self.emb_dim))
        elif config['STAREARGS']['OPN'] == 'rotate' or config['STAREARGS']['QUAL_OPN'] == 'rotate' or config['STAREARGS']['OPN'] == 'con_add_mul' or config['STAREARGS']['QUAL_OPN'] == 'con_add_mul':
            phases = 2 * np.pi * torch.rand(self.num_rel, self.emb_dim // 2)
            self.init_rel = nn.Parameter(torch.cat([
                torch.cat([torch.cos(phases), torch.sin(phases)], dim=-1),
                torch.cat([torch.cos(phases), -torch.sin(phases)], dim=-1)
            ], dim=0))
        else:
            self.init_rel = get_param((self.num_rel * 2, self.emb_dim))

        self.init_rel.data[0] = 0 # padding

        self.conv1 = HypeTKGConvLayer(self.emb_dim, self.gcn_dim, self.num_rel, act=self.act,
                                       config=config)
        self.conv2 = HypeTKGConvLayer(self.gcn_dim, self.emb_dim, self.num_rel, act=self.act,
                                       config=config) if self.n_layer == 2 else None

        if self.conv1: self.conv1.to(self.device)
        if self.conv2: self.conv2.to(self.device)

        self.register_parameter('bias', Parameter(torch.zeros(self.num_ent)))

    def get_unique_qualifier_pairs(self, quali_pairs):
        # Transfer tensor to List of the column, each col is a tuple
        columns = list(map(tuple, quali_pairs.t().cpu().numpy()))

        # A set for storing the observed columns and create a list for saving the results.
        seen = set()
        result = []

        for col in columns:
            if col not in seen:
                result.append(col)
                seen.add(col)

        # Transfer the result to tensor back.
        quali_pairs_unique = torch.tensor(result, dtype=torch.long, device=self.device).t()
        return quali_pairs_unique

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj.weight)

    def forward_base(self, sub, rel, time, drop1, drop2,
                     quals=None, return_mask: bool = False):
        """"
        :param sub:
        :param rel:
        :param drop1:
        :param drop2:
        :param quals: (optional) (bs, maxqpairs*2) Each row is [qp, qe, qp, qe, ...]
        :param return_mask: if True, returns a True/False mask of [bs, total_len] that says which positions were padded
        :return:
        """
        r = self.init_rel

        x, r = self.conv1(x=self.init_embed, edge_index=self.edge_index,
                          edge_type=self.edge_type, edge_time=self.edge_time,
                          rel_embed=r,
                          quals=self.quals,
                          static_index=self.static_index,
                          static_type=self.static_type)

        x = drop1(x)
        x, r = self.conv2(x=x, edge_index=self.edge_index,
                          edge_type=self.edge_type, edge_time=self.edge_time,
                          rel_embed=r,
                          quals=self.quals,
                          static_index=self.static_index,
                          static_type=self.static_type) if self.n_layer == 2 else (x, r)


        x = drop2(x) if self.n_layer == 2 else x

        sub_emb = torch.index_select(x, 0, sub)
        # Time-Encoding
        # time_emb = None
        sub_emb, time_emb = self.get_ent_time_emb(sub_emb, time[:, np.newaxis])
        rel_emb = torch.index_select(r, 0, rel)

        if  self.embed_statics:
            static_all = torch.zeros(self.raw_num_ent, self.static_all.size(1), dtype=torch.long, device= self.device)
            static_all[:self.static_all.size(0), :] = self.static_all
            all_static_emb = self.static_emb(static_all, x, r)[:, 1:, :]
            # sub_static = torch.index_select(self.static_all[:, 1:], 0, sub)
            # sub_static_emb = torch.index_select(all_entity[:, 1:, :], 0, sub)

            if self.embed_qualifiers:
                assert quals is not None, "Expected a tensor as quals."

                # unique_qual_pair_embedding
                qual_pair_obj_emb = torch.index_select(x, 0, self.quals_pairs[1])
                qual_pair_rel_emb = torch.index_select(r, 0, self.quals_pairs[0])
                qual_pair_emb = torch.concat([qual_pair_rel_emb, qual_pair_rel_emb], dim=1)
                # qualifier_sub_mask
                qual_sub_mask = torch.stack([torch.from_numpy(list(self.sub_qual_mask_dict.values())[i]).type(torch.long).to(self.device)  for i in sub])
                #qualifier_rel_mask
                # qual_sub_mask = torch.stack([torch.from_numpy(list(self.sub_qual_mask_dict.values())[i]).type(torch.long).to(self.device)  for i in rel])

                # flatten quals
                quals_ents = quals[:, 1::2].view(1,-1).squeeze(0)
                quals_rels = quals[:, 0::2].view(1,-1).squeeze(0)
                qual_obj_emb = torch.index_select(x, 0, quals_ents)
                # qual_obj_emb = torch.index_select(x, 0, quals[:, 1::2])
                qual_rel_emb = torch.index_select(r, 0, quals_rels)
                qual_obj_emb = qual_obj_emb.view(sub_emb.shape[0], -1 ,sub_emb.shape[1])
                qual_rel_emb = qual_rel_emb.view(rel_emb.shape[0], -1, rel_emb.shape[1])
                qual_emb = torch.cat((qual_rel_emb, qual_obj_emb), 2).view(-1, 2 * qual_rel_emb.shape[1],
                                                                            qual_rel_emb.shape[2])
                if not return_mask:
                    return sub_emb, rel_emb, qual_emb, all_static_emb, x[:self.raw_num_ent], None, None, qual_pair_emb, qual_sub_mask, time_emb
                else:
                    # mask which shows which entities were padded - for future purposes, True means to mask (in transformer)
                    # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py : 3770
                    qual_mask = torch.zeros((sub.shape[0], quals.shape[1]+1)).bool().to(self.device)
                    # and put True where qual entities and relations are actually padding index 0
                    qual_mask[:, 1:] = quals == 0
                    static_mask = torch.zeros((static_all.shape[0], static_all.shape[1])).bool().to(self.device)
                    static_mask[:, 1:] = static_all[:, 1:] == 0
                    # mask = torch.cat((qual_mask, static_mask), dim=1)
                    return sub_emb, rel_emb, qual_emb, all_static_emb, x[:self.raw_num_ent], qual_mask, static_mask, qual_pair_emb, qual_sub_mask, time_emb
            else:
                if not return_mask:
                    return sub_emb, rel_emb, None, all_static_emb, x[:self.raw_num_ent], None, None, None, None, time_emb
                else:
                    static_mask = torch.zeros((static_all.shape[0], static_all.shape[1])).bool().to(self.device)
                    static_mask[:, 1:] = static_all[:, 1:] == 0
                    return sub_emb, rel_emb, None, all_static_emb, x[:self.raw_num_ent], None, static_mask, None, None, time_emb

        elif self.embed_qualifiers:
            assert quals is not None, "Expected a tensor as quals."

            if self.quals_pairs.numel() != 0:
                # unique_qual_pair_embedding
                qual_pair_obj_emb = torch.index_select(x, 0, self.quals_pairs[1])
                qual_pair_rel_emb = torch.index_select(r, 0, self.quals_pairs[0])
                qual_pair_emb = torch.concat([qual_pair_rel_emb, qual_pair_rel_emb], dim=1)
                # qualifier_sub_mask
                qual_sub_mask = torch.stack(
                    [torch.from_numpy(list(self.sub_qual_mask_dict.values())[i]).type(torch.long).to(self.device) for i in
                     sub])
            else:
                qual_pair_emb = None
                qual_sub_mask = None
            # qualifier_rel_mask
            # qual_sub_mask = torch.stack([torch.from_numpy(list(self.sub_qual_mask_dict.values())[i]).type(torch.long).to(self.device)  for i in rel])

            # flatten quals
            quals_ents = quals[:, 1::2].view(1, -1).squeeze(0)
            quals_rels = quals[:, 0::2].view(1, -1).squeeze(0)
            qual_obj_emb = torch.index_select(x, 0, quals_ents)
            # qual_obj_emb = torch.index_select(x, 0, quals[:, 1::2])
            qual_rel_emb = torch.index_select(r, 0, quals_rels)
            qual_obj_emb = qual_obj_emb.view(sub_emb.shape[0], -1, sub_emb.shape[1])
            qual_rel_emb = qual_rel_emb.view(rel_emb.shape[0], -1, rel_emb.shape[1])
            qual_emb = torch.cat((qual_rel_emb, qual_obj_emb), 2).view(-1, 2 * qual_rel_emb.shape[1],
                                                                       qual_rel_emb.shape[2])
            if not return_mask:
                return sub_emb, rel_emb, qual_emb, None, x[:self.raw_num_ent], None, None, qual_pair_emb, qual_sub_mask, time_emb

            else:
                qual_mask = torch.zeros((sub.shape[0], quals.shape[1]+1)).bool().to(self.device)
                qual_mask[:, 1:] = quals == 0
                return sub_emb, rel_emb, qual_emb, None, x[:self.raw_num_ent], qual_mask, None, qual_pair_emb, qual_sub_mask, time_emb
        else:
           return sub_emb, rel_emb, None, None, x[:self.raw_num_ent], None, None, None, None, time_emb

    def static_emb(self, sta, x, r):
        '''
        :param sta: (10038, 17)
        :return:
        '''
        sub = sta[:, 0]  #(10038)
        rel = sta[:, 1::2] # (10038, 8)
        obj = sta[:, 2::2] # (10038, 8)

        rel = rel.contiguous().view(-1)
        obj = obj.contiguous().view(-1)
        sub_emb = torch.index_select(x, 0, sub)
        rel_emb = torch.index_select(r, 0, rel)
        obj_emb = torch.index_select(x, 0, obj)
        sub_emb = sub_emb.unsqueeze(1)         # (10038, 1, 200)
        rel_emb = rel_emb.view(sub.shape[0], -1, self.emb_dim) # (10038, 8, 200)
        obj_emb = obj_emb.view(sub.shape[0], -1, self.emb_dim) # (10038, 8, 200)

        sta_emb = torch.cat((rel_emb, obj_emb), 2).view(-1, 2 * rel_emb.shape[1],
                                                                    rel_emb.shape[2])
        sta_emb = torch.cat([sub_emb, sta_emb], 1)

        return sta_emb  # (10038, 17, 200)

    def get_ent_time_emb(self, ent_emb, ts):

        time_emb = self.time_encoder(ts)
        time_emb = torch.squeeze(time_emb, dim=1)
        assert time_emb.shape == ent_emb.shape
        t_ent_emb = self.proj(torch.cat([ent_emb, time_emb], dim=-1))
        # t_ent_emb = ent_emb + self.res_cof * self.act(self.proj(torch.cat([ent_emb, time_emb], dim=-1)))
        return t_ent_emb, time_emb



