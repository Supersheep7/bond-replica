import re
import numpy as np
from tqdm import tqdm
from os.path import join
from gensim.models import word2vec
from datetime import datetime
import json
import os

def load_dataset(mode):
    """
    Load dataset by mode.

    Args:
        mode.
    Returns:
        names(list):[guanhua_du, ...]
        pubs(dict): {author:[...], title: xxx, ...}
    """
    if mode == 'train':
        with open(join(os.path.dirname(__file__), "data_train.json"), "r", encoding="utf-8") as f:
            raw_pubs = json.load(f)['data_train'][0]
    elif mode == 'valid':
        with open(join(os.path.dirname(__file__), "data_valid.json"), "r", encoding="utf-8") as f:
            raw_pubs = json.load(f)['data_valid'][0]
    elif mode == 'test':
        with open(join(os.path.dirname(__file__), "data_test.json"), "r", encoding="utf-8") as f:
            raw_pubs = json.load(f)['data_test'][0]
    else:
        raise ValueError('choose right mode')

    names = []
    for name in raw_pubs:
        names.append(name)
    
    return names, raw_pubs
