import argparse
import json

import sys
sys.path.append(f'../')
from DEE.models import model, prob_models

def json2args(json_path):
    with open(json_path) as f:
        args = argparse.Namespace(**json.load(f))
    return args

def get_DEE_from_json(json_path):
    """_summary_
    input : json path
    output : DEE model
    """
    args = json2args(json_path)
    DEE = None
    if args.loss in ['soft_contrastive','csd']:
        print('using prob model')
        DEE = prob_models.ProbDEE(args)
    else:
        if args.use_emo2vec:
            print('using DEE_v2')
            DEE = model.DEE_v2(args)
        else:
            print('using DEE')
            DEE = model.DEE(args)
    return DEE,args
    
