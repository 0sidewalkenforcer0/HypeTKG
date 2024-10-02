from pathlib import Path

import random
import numpy as np
import pickle
from tqdm import tqdm


# from typing import Callable


class DataManager(object):
    """ Load the processed data"""

    @staticmethod
    def load(dataset) -> dict:
        # Function to randomly select half of the non-padding qualifier-pairs
        def select_half_qualifiers(data_line):
            main_qudruple, qualifiers = data_line[:4], data_line[4:]

            # Filter out padding (zeros)
            real_qualifiers = [q for q in zip(qualifiers[::2], qualifiers[1::2]) if q != (0, 0)]

            # Check if all qualifiers are zeros (padding)
            if all(q == 0 for q in qualifiers):
                return data_line  # Return the original line if all qualifiers are padding

            # Determine whether to round up or down for half length
            half_len = len(real_qualifiers) // 2
            if len(real_qualifiers) % 2 != 0:  # Check if the number of qualifiers is odd
                half_len += random.randint(0, 1)  # Randomly decide to add 0 (round down) or 1 (round up)

            # Randomly selecting qualifiers
            selected_qualifiers = random.sample(real_qualifiers, half_len) if real_qualifiers else []

            # Flattening the list of tuples and adding padding if necessary
            flattened_selected = [elem for pair in selected_qualifiers for elem in pair]
            padding_needed = len(qualifiers) - len(flattened_selected)
            flattened_selected.extend([0] * padding_needed)

            return main_qudruple + flattened_selected

        # Function to padding all qualifiers to 0
        def zero_out_nonzero_qualifiers(data_line):
            main_quadruple, qualifiers = data_line[:4], data_line[4:]

            # Check if all qualifiers are zeros (padding)
            if all(q == 0 for q in qualifiers):
                return data_line

            # Set all non-zero qualifiers to zero
            zeroed_qualifiers = [0 if q != 0 else 0 for q in qualifiers]

            return main_quadruple + zeroed_qualifiers

        # load the gcn data
        DIRNAME = Path('./data/{}/data_in_model'.format(dataset))
        with open(DIRNAME / './train_bi_direction.txt', 'r') as f:
            train = []
            for line in tqdm(f.readlines(), desc="load the train data"):
                each_line = line.strip("\n").split()
                each_line = list(map(int, each_line))
                # Select half qualifiers
                # each_line = select_half_qualifiers(each_line)
                # Select 0 qualifier
                # each_line = zero_out_nonzero_qualifiers(each_line)
                train.append(each_line)

        with open(DIRNAME / './valid_bi_direction.txt', 'r') as f:
            valid = []
            for line in tqdm(f.readlines(), desc="load the valid data"):
                each_line = line.strip("\n").split()
                each_line = list(map(int, each_line))
                # Select half qualifiers
                # each_line = select_half_qualifiers(each_line)
                # Select 0 qualifier
                # each_line = zero_out_nonzero_qualifiers(each_line)
                valid.append(each_line)

        with open(DIRNAME / './test_bi_direction.txt', 'r') as f:
            test = []
            for line in tqdm(f.readlines(), desc="load the test data"):
                each_line = line.strip("\n").split()
                each_line = list(map(int, each_line))
                # Select half qualifiers
                # each_line = select_half_qualifiers(each_line)
                # Select 0 qualifier
                # each_line = zero_out_nonzero_qualifiers(each_line)
                test.append(each_line)

#        load the static data
        with open(DIRNAME / './static_agg.txt', 'r') as f:
            static = []
            for line in tqdm(f.readlines(), desc="load the static_agg data"):
                each_line = line.strip("\n").split()
                each_line = list(map(int, each_line))
                static.append(each_line)

        # load the number of entities and relations
        # num_raw_entities = len(open(Path('./data/{}/data_in_model/static_agg.txt'.format(dataset)), 'rb').readlines())
        if dataset.lower().startswith('yago'):
            num_raw_entities = int(10026 + 359 + 1)
        elif dataset.lower().startswith('wiki'):
            num_raw_entities = int(11140 + 1694 + 1)
        elif dataset in ['icews14', 'icews05-15', 'gdelt']:
            num_raw_entities = len(open(Path('./data/{}/all_entity2id.txt'.format(dataset)), 'rb').readlines()) + 1
        else:
            raise NotImplementedError
        num_new_entities = len(open(Path('./data/{}/all_entity2id.txt'.format(dataset)), 'rb').readlines())
        num_rels = len(open('./data/{}/all_relation2id.txt'.format(dataset), 'rb').readlines())
        if dataset in ['icews14', 'icews05-15', 'gdelt']:
            return {"static": static, "num_raw_entities": num_raw_entities,"num_new_entities": num_new_entities + 1,
                "num_rels": num_rels + 1, "train": train, "valid": valid, "test": test,
                "len_qualifier": len(test[0])-4, "len_static": 0}
        else:
            return {"static": static, "num_raw_entities": num_raw_entities,"num_new_entities": num_new_entities + 1,
                "num_rels": num_rels + 1, "train": train, "valid": valid, "test": test,
                "len_qualifier": len(test[0])-4, "len_static": len(static[0])-1}

    @staticmethod
    def get_unique_qualifier_pairs(quali_pairs, quals_sub):
        columns = list(map(tuple, quali_pairs.T))
        seen = set()
        result = []
        sub_result = []

        for col, sub in zip(columns, quals_sub):
            if col not in seen:
                result.append(col)
                seen.add(col)
            sub_result.append(sub)

        unique_columns = []
        unique_subs = []

        for col in result:
            indices = [i for i, c in enumerate(columns) if c == col]
            unique_subs.append(list(set(sub_result[i] for i in indices)))

        # quali_pairs_unique = torch.tensor(result, dtype=torch.long, device=self.device).T
        # qual_sub_unique = [torch.unique(torch.tensor(subs, dtype=torch.long, device=self.device)) for subs in
        #                    unique_subs]

        return result, unique_subs

    @staticmethod
    def get_sub_mask(sub, sub_unique):
        sub_mask = np.ones(len(sub_unique), dtype=np.bool)
        for i, sub_arr in enumerate(sub_unique):
            if sub in sub_arr:
                sub_mask[i] = False
        return sub_mask

    @staticmethod
    def get_alternative_graph_repr(train_data, static_data, dataset):
        # quadruples
        edge_sub = []
        edge_type = []
        edge_obj = []
        edge_time = []
        # qualifiers
        qualifier_rel = []
        qualifier_obj = []
        qualifier_index = []
        # statics
        static_sub = []
        static_rel = []
        static_obj = []
        # qualifier_sub
        quals_sub = []

        # Add data
        for i, data in enumerate(tqdm(train_data, desc="build the graph representation")):
            edge_sub.append(data[0])
            edge_type.append(data[1])
            edge_obj.append(data[2])
            edge_time.append(data[3])

            # add qualifiers
            qual_rel = np.array(data[4::2])
            qual_ent = np.array(data[5::2])
            non_zero_rels = qual_rel[np.nonzero(qual_rel)]
            non_zero_ents = qual_ent[np.nonzero(qual_ent)]
            for j in range(non_zero_ents.shape[0]):
                qualifier_rel.append(non_zero_rels[j])
                qualifier_obj.append(non_zero_ents[j])
                qualifier_index.append(i)
                quals_sub.append(data[0])
                # quals_sub.append(data[1])

        edge_index = np.stack((edge_sub, edge_obj), axis=0)
        quals = np.stack((qualifier_rel, qualifier_obj, qualifier_index), axis=0)

        # get unique qual-pairs
        sub_qual_mask_dict = {}
        all_unique_quals, all_qual_sub_unique = DataManager.get_unique_qualifier_pairs(quals[:2], quals_sub)
        # sub
        if dataset.lower().startswith('yago'):
            for sub in tqdm(range(10026+1), desc="build the qualifier mask for all entities"):
                qual_sub_mask = DataManager.get_sub_mask(sub, all_qual_sub_unique)
                sub_qual_mask_dict[sub] = qual_sub_mask

        elif dataset.lower().startswith('wiki'):
            for sub in tqdm(range(11140+1), desc="build the qualifier mask for all entities"):
                qual_sub_mask = DataManager.get_sub_mask(sub, all_qual_sub_unique)
                sub_qual_mask_dict[sub] = qual_sub_mask

        # rel
        # num_rel = len(open('./data/{}/all_relation2id.txt'.format(dataset), 'rb').readlines())
        #
        # for sub in tqdm(range(num_rel*2+1), desc="build the qualifier mask for all entities"):
        #     qual_sub_mask = DataManager.get_sub_mask(sub, all_qual_sub_unique)
        #     sub_qual_mask_dict[sub] = qual_sub_mask
        # add the statics

        for j, sta in enumerate(tqdm(static_data, desc="build the static-triples")):
            s = np.array(sta)
            non_zero_s = s[np.nonzero(s)]
            s_rel = non_zero_s[1::2]
            s_obj = non_zero_s[2::2]
            static_rel = static_rel + list(s_rel)
            static_obj = static_obj + list(s_obj)
            static_sub = static_sub + len(s_rel) * [sta[0]]

        static_index = np.stack((static_sub,  static_obj), axis=0)



        return {'edge_index': edge_index,
                'edge_type': edge_type,
                'edge_time': edge_time,
                'quals': quals,
                'static_index': static_index,
                'static_type': static_rel,
                'quals_sub_mask': sub_qual_mask_dict,
                }

if __name__ == '__main__':
    data = DataManager.load()
    with open('./data/processed_data/train_data_gcn.pickle', 'rb') as f:
        train_data_gcn = pickle.load(f)
    gcn_train = data["gcn_train"]
    gcn_valid = data["gcn_valid"]
    static_data = data["static"]
    # print(type(train_data + valid_data))
