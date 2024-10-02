import os
os.environ['MKL_NUM_THREADS'] = '1'

from functools import partial
import random
import wandb
import sys
import collections

# Local imports
from data_loader.data_manager import DataManager
from utils.utils import *
from utils.utils_mytorch import FancyDict, parse_args, BadParameters, mt_save_dir
from loops.evaluation import EvaluationBenchGNNMultiClass, evaluate_pointwise
from loops.evaluation import acc, mrr, mr, hits_at
from models.models_statements import HypeTKG
from loops.corruption import Corruption
from loops.sampler import MultiClassSampler
from loops.loops import training_loop_gcn

"""
    CONFIG Things
"""

# Clamp the randomness
np.random.seed(66)
random.seed(66)
torch.manual_seed(166)

DEFAULT_CONFIG = {
    'BATCH_SIZE': 256,
    'DATASET':'YAGO-hy',
    'DEVICE': 'cuda',  # cpu
    'EMBEDDING_DIM': 300,
    'ENT_POS_FILTERED': False,
    'EPOCHS': 400,
    'EVAL_EVERY': 5,
    'LEARNING_RATE': 0.0001,

    'MAX_QPAIRS': 40,
    'MODEL_NAME': 'HypeTKG',
    'CORRUPTION_POSITIONS': [0, 2],
    'TEST_MODEL_PATH': None,
    'PRE_TRAIN_PATH': None,

    # important args
    'SAVE':False,
    'STATEMENT_LEN': -1,
    'USE_TEST': True,
    'WANDB': False,
    'LABEL_SMOOTHING': 0.1,
    'SAMPLER_W_QUALIFIERS': True, # use the qualifiers
    'SAMPLER_W_STATICS': True,  # use the statics
    'OPTIMIZER': 'adam',
    'GRAD_CLIPPING': True,
    'LR_SCHEDULER': True
}

STAREARGS = {
    'LAYERS': 2,
    'N_BASES': 0,
    'GCN_DIM': 300,
    'GCN_DROP': 0.1,
    'HID_DROP': 0.3,
    'BIAS': False,
    'OPN': 'con_add_mul',
    'TRIPLE_QUAL_WEIGHT': 0.8,
    'QUAL_AGGREGATE': 'sum',  # or concat or mul
    # 'QUAL_OPN': 'rotate',
    'QUAL_OPN': 'con_add_mul',
    'QUAL_N': 'sum',  # or mean
    'SUBBATCH': 0,
    'QUAL_REPR': 'sparse',  # sparse or full. Warning: full is 10x slower
    'ATTENTION': False,
    'ATTENTION_HEADS': 4,
    'ATTENTION_SLOPE': 0.2,
    'ATTENTION_DROP': 0.1,
    'HID_DROP2': 0.1,

    # For ConvE Only
    'FEAT_DROP': 0.3,
    'N_FILTERS': 200,
    'KERNEL_SZ': 7,
    'K_W': 10,
    'K_H': 20,

    # For Transformer
    'T_LAYERS': 2,
    'T_N_HEADS': 4,
    'T_HIDDEN': 512,
    'POSITIONAL': True,
    'POS_OPTION': 'default',
    'TIME': False,
    'POOLING': 'avg'

}

DEFAULT_CONFIG['STAREARGS'] = STAREARGS

if __name__ == "__main__":

    # Get parsed arguments
    config = DEFAULT_CONFIG.copy()
    gcnconfig = STAREARGS.copy()

    config['STAREARGS'] = gcnconfig

    data = DataManager.load(config['DATASET'])

    # Break down the data
    try:
        train_data = data['train']
        valid_data = data['valid']
        test_data =  data['test']
        static_data = data['static']
    except ValueError:
        raise ValueError(f"Honey I broke the loader for {config['DATASET']}")

    config['NUM_RAW_ENTITIES'] = data['num_raw_entities']
    config['NUM_ENTITIES'] = data['num_new_entities']
    config['NUM_RELATIONS'] = data['num_rels']
    config["STATIC_ALL"] = data['static']
    config["len_qualifier"] = data["len_qualifier"]
    config["len_static"] = data["len_static"]

    """
     However, when we want to run a GCN based model, we also work with
            COO representations of quadruples and qualifiers.
    
            In this case, for each split: [train, valid, test], we return
            -> edge_index (2 x n) matrix with [subject_ent, object_ent] as each row.
            -> edge_type (n) array with [relation] corresponding to sub, obj above.
            -> edge_time (n) array with [time] corresponding to sub, obj, rel above.
            -> quals (3 x nQ) matrix where columns represent quals [qr, qv, k] for each k-th edge that has quals
    
        So here, train_data_gcn will be a dict containing these ndarrays.
    """

    # Replace the data with their graph repr formats
    if config['USE_TEST']:
        train_data_gcn = DataManager.get_alternative_graph_repr(train_data + valid_data, static_data, config['DATASET'])
    else:
        train_data_gcn = DataManager.get_alternative_graph_repr(train_data, static_data, config['DATASET'])
    """
        Make the model.
    """
    config['DEVICE'] = torch.device(config['DEVICE'])
    model = HypeTKG(train_data_gcn, config)
    model.to(config['DEVICE'])
    print("Model params: ",sum([param.nelement() for param in model.parameters()]))

    if config['OPTIMIZER'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['LEARNING_RATE'])
    elif config['OPTIMIZER'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])
    else:
        print("Unexpected optimizer, we support `sgd` or `adam` at the moment")
        raise NotImplementedError


    if config['WANDB']:
        wandb.init(project="HypeTKG")
        for k, v in config.items():
            wandb.config[k] = v

    """
        Prepare test benches.
        
            When computing train accuracy (`ev_tr_data`), we wish to use all the other data 
                to avoid generating true triples during corruption. 
            Similarly, when computing test accuracy, we index train and valid splits 
                to avoid generating negative triples.
    """
    if config['USE_TEST']:
        ev_vl_data = {'index': combine(train_data, valid_data), 'eval': combine(test_data)}
        ev_tr_data = {'index': combine(valid_data, test_data), 'eval': combine(train_data)}
        tr_data = {'train': combine(train_data, valid_data), 'valid': ev_vl_data['eval']}
        test_data ={'index': combine(train_data, valid_data), 'eval': combine(test_data)}
        valid_data = {'index': combine(train_data), 'eval': combine(valid_data)}

    else:
        ev_vl_data = {'index': combine(train_data, test_data), 'eval': combine(valid_data)}
        ev_tr_data = {'index': combine(valid_data, test_data), 'eval': combine(train_data)}
        tr_data = {'train': combine(train_data), 'valid': ev_vl_data['eval']}
        test_data ={'index': combine(train_data, test_data), 'eval': combine(valid_data)}


    eval_metrics = [acc, mrr, mr, partial(hits_at, k=3),
                    partial(hits_at, k=5), partial(hits_at, k=10)]


    evaluation_valid = None
    evaluation_train = None

    # Saving stuff
    if config['SAVE']:
        savedir = Path(f"./models/{config['DATASET']}/{config['MODEL_NAME']}")
        if not savedir.exists(): savedir.mkdir(parents=True)
        savedir = mt_save_dir(savedir, _newdir=True)
        save_content = {'model': model, 'config': config}
    else:
        savedir, save_content = None, None

    # The args to use if we're training w default stuff
    args = {
        "epochs": config['EPOCHS'],
        "data": tr_data,
        "opt": optimizer,
        "train_fn": model,
        # "neg_generator": Corruption(n=n_entities, excluding=[0],
        #                             position=list(range(0, config['MAX_QPAIRS'], 2))),
        "device": config['DEVICE'],
        "data_fn": None,
        "eval_fn_trn": evaluate_pointwise,
        "val_testbench": evaluation_valid.run if evaluation_valid else None,
        "trn_testbench": evaluation_train.run if evaluation_train else None,
        "eval_every": config['EVAL_EVERY'],
        "log_wandb": config['WANDB'],
        "run_trn_testbench": False,
        "savedir": savedir,
        "save_content": save_content,
        "qualifier_aware": config['SAMPLER_W_QUALIFIERS'],
        "grad_clipping": config['GRAD_CLIPPING'],
        "scheduler": None
    }


    training_loop = training_loop_gcn
    sampler = MultiClassSampler(data= args['data']['train'],
                                n_entities=config['NUM_RAW_ENTITIES'],
                                lbl_smooth=config['LABEL_SMOOTHING'],
                                bs=config['BATCH_SIZE'],
                                with_q=config['SAMPLER_W_QUALIFIERS'],
                                )
    evaluation_valid = EvaluationBenchGNNMultiClass(ev_vl_data, model, bs=config['BATCH_SIZE'], metrics=eval_metrics,
                                       filtered=True, n_ents=config['NUM_RAW_ENTITIES'],
                                       positions=config.get('CORRUPTION_POSITIONS', None), config=config)
    args['data_fn'] = sampler.reset
    args['val_testbench'] = evaluation_valid.run
    args['trn_testbench'] = None
    if config['LR_SCHEDULER']:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.95)
        args['scheduler'] = scheduler

    if config['TEST_MODEL_PATH'] is not None:
        assert os.path.exists(config['TEST_MODEL_PATH']), "model file does not exist."
        model_state_dict = torch.load(config['TEST_MODEL_PATH']+'/model.torch',  map_location=config['DEVICE'])
        model.load_state_dict(model_state_dict)
        # print(model.state_dict())
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.shape)
        model.to(config['DEVICE'])
        evaluation_test = EvaluationBenchGNNMultiClass(test_data, model, bs=config['BATCH_SIZE'], metrics=eval_metrics,
                           filtered=True, n_ents=config['NUM_RAW_ENTITIES'],
                           positions=config.get('CORRUPTION_POSITIONS', None), config=config)
        summary_val = evaluation_test.run()
    else:
        if config['PRE_TRAIN_PATH'] is not None:
            assert os.path.exists(config['PRE_TRAIN_PATH']), "model file does not exist."

            # device_count = torch.cuda.device_count()
            # #
            # # check the number of cuda device
            # # assert device_count >= 2, "At least more than 2 CUDA devices"
            #
            # if device_count > 1:
            #     map_location = {"cuda:2": config['device']}  # Map device 2 to 0
            # else:
            #     map_location = config['device']  #Use default device if there is only one.

            model_state_dict = torch.load(config['PRE_TRAIN_PATH'] + '/model.torch', map_location=config['DEVICE'])

            model.load_state_dict(model_state_dict)
            model.to(config['DEVICE'])
        traces = training_loop(**args)
        with open('traces.pkl', 'wb+') as f:
            pickle.dump(traces, f)
