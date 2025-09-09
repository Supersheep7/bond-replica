import os
import re
from tqdm import tqdm
from os.path import join
import pinyin
import unicodedata
import json
import codecs

''' HELPER FUNCTS '''

puncs = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'
stopwords = ['at', 'based', 'in', 'of', 'for', 'on', 'and', 'to', 'an', 'using', 'with',
            'the', 'by', 'we', 'be', 'is', 'are', 'can']
stopwords_extend = ['university', 'univ', 'china', 'department', 'dept', 'laboratory', 'lab',
                    'school', 'al', 'et', 'institute', 'inst', 'college', 'chinese', 'beijing',
                    'journal', 'science', 'international', 'key', 'sciences', 'research',
                    'academy', 'state', 'center']
stopwords_check = ['a', 'was', 'were', 'that', '2', 'key', '1', 'technology', '0', 'sciences', 'as',
                    'from', 'r', '3', 'academy', 'this', 'nanjing', 'shanghai', 'state', 's', 'research',
                    'p', 'results', 'peoples', '4', 'which', '5', 'high', 'materials', 'study', 'control',
                    'method', 'group', 'c', 'between', 'or', 'it', 'than', 'analysis', 'system',  'sci',
                    'two', '6', 'has', 'h', 'after', 'different', 'n', 'national', 'japan', 'have', 'cell',
                    'time', 'zhejiang', 'used', 'data', 'these']

names_wrong = [
    # find in train
    (['takahiro', 'toshiyuki', 'takeshi', 'toshiyuki', 'tomohiro', 'takamitsu', 'takahisa', 'takashi',
     'takahiko', 'takayuki'], 'ta(d|k)ashi'),
    (['akimasa', 'akio', 'akito'], 'akira'),
    (['kentarok'], 'kentaro'),
    (['xiaohuatony', 'tonyxiaohua'], 'xiaohua'),
    (['ulrich'], 'ulrike'),
    # find in valid
    (['naoto', 'naomi'], 'naoki'),
    (['junko'], 'junichi'),
    # find in test
    (['isaku'], 'isao')
]

def unify_name_order(name):
    """
    unifying different orders of name.
    Args:
        name
    Returns:
        name and reversed name
    """
    token = name.split("_")
    name = token[0] + token[1]
    name_reverse = token[1] + token[0]
    if len(token) > 2:
        name = token[0] + token[1] + token[2]
        name_reverse = token[2] + token[0] + token[1]

    return name, name_reverse

def is_contains_chinese(strs):
    """
    Check if contains chinese characters.
    """
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False

def match_name(name, target_name):
    """
    Match different forms of names.
    """
    [first_name, last_name] = target_name.split('_')
    first_name = re.sub('-', '', first_name)
    # change Chinese name to pinyin
    if is_contains_chinese(name):
        name = re.sub('[^ \u4e00-\u9fa5]', '', name).strip()
        name = pinyin.get(name, format='strip')
        # remove blank space between characters
        name = re.sub(' ', '', name)
        target_name = last_name + first_name
        return name == target_name
    else:
        # unifying Pinyin characters with tones
        str_bytes = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore')
        name = str_bytes.decode('ascii')

        name = name.lower()
        name = re.sub('[^a-zA-Z]', ' ', name)
        tokens = name.split()

        if len(tokens) < 2:
            return False
        if len(tokens) == 3:
            # just ignore middle name
            if re.match(tokens[0], first_name) and re.match(tokens[-1], last_name):
                return True
            # ignore tail noise char
            if tokens[-1] == 'a' or tokens[-1] == 'c':
                tokens = tokens[:-1]

        if re.match(tokens[0], last_name):
            # unifying two-letter abbreviation of the Chinese name
            if len(tokens) == 2 and len(tokens[1]) == 2:
                if re.match(f'{tokens[1][0]}.*{tokens[1][1]}.*', first_name):
                    return True
            remain = '.*'.join(tokens[1:]) + '.*'

            if re.match(remain, first_name):
                return True
            if len(tokens) == 3 and len(tokens[1]) == 1 and len(tokens[2]) == 1:
                remain_reverse = f'{tokens[2]}.*{tokens[1]}.*'
                if re.match(remain_reverse, first_name):
                    return True
        if re.match(tokens[-1], last_name):
            candidate = ''.join(tokens[:-1])
            find_remain = False
            for (wrong_list, right_one) in names_wrong:
                if candidate in wrong_list:
                    remain = right_one
                    find_remain = True
                    break
            if not find_remain:
                remain = '.*'.join(tokens[:-1]) + '.*'

            if re.match(remain, first_name):
                return True
            if len(tokens) == 3 and len(tokens[0]) == 1 and len(tokens[1]) == 1:
                remain_reverse = f'{tokens[1]}.*{tokens[0]}.*'
                if re.match(remain_reverse, first_name):
                    return True
        return False
    
''' ACTUAL PREPROCESSING FUNCTS '''

def read_pubinfo(mode):
    """
    Read pubs' meta-information. >> This reads the title/org/keywords metadata. >> For building embeddings
    """
    if mode == 'train':
        with open(join(os.path.dirname(__file__), "data_train.json"), "r", encoding="utf-8") as f:
            pubs = json.load(f)[0][1]
    elif mode == 'valid':
        with open(join(os.path.dirname(__file__), "data_val.json"), "r", encoding="utf-8") as f:
            pubs = json.load(f)[0][1]
    elif mode == 'test':
        with open(join(os.path.dirname(__file__), "data_test.json"), "r", encoding="utf-8") as f:
            pubs = json.load(f)[0][1]
    else:
        raise ValueError('choose right mode')
    
    return pubs

def read_raw_pubs(mode):
    """
    Read raw pubs. >> This reads the co-author/co-venue/co-org metadata >> For building graph
    """
    if mode == 'train':
        with open(join(os.path.dirname(__file__), "data_train.json"), "r", encoding="utf-8") as f:
            raw_pubs = json.load(f)[0][0]
    elif mode == 'valid':
        with open(join(os.path.dirname(__file__), "data_valid.json"), "r", encoding="utf-8") as f:
            raw_pubs = json.load(f)[0][0]
    elif mode == 'test':
        with open(join(os.path.dirname(__file__), "data_test.json"), "r", encoding="utf-8") as f:
            raw_pubs = json.load(f)[0][0]
    else:
        raise ValueError('choose right mode')
    
    return raw_pubs


def dump_name_pubs():
    """
    Split publications informations by {name} and dump files as {name}.json
    """
    for mode in ['train', 'valid', 'test']:
        raw_pubs = read_raw_pubs(mode)
        pub_info = read_pubinfo(mode)
        file_path = join(os.getcwd(), 'names_pub', mode)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        for name in tqdm(raw_pubs):
            name_pubs_raw = {}
            if mode != "train":
                for i, pid in enumerate(raw_pubs[name]):
                    name_pubs_raw[pid] = pub_info[pid]
            else:
                pids = []
                for aid in raw_pubs[name]:
                    pids.extend(raw_pubs[name][aid])
                for pid in pids:
                    name_pubs_raw[pid] = pub_info[pid]
            
            with codecs.open(join(file_path, name+'.json'), 'w', encoding='utf-8') as wf:
                json.dump(name_pubs_raw, wf, ensure_ascii=False, indent=4)


def dump_features_relations_to_file():
    """
    Generate paper features and relations by raw publication data and dump to files.
    Paper features consist of title, org, keywords. >> To be embedded in graph.py
    Paper relations consist of author_name, org, venue. >> To be graphed in graph.py
    """
    r = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'

    for mode in ['train', 'valid', 'test']:
        raw_pubs = read_raw_pubs(mode)
        for n, name in tqdm(enumerate(raw_pubs)):
            file_path = join(os.getcwd(), 'relations', mode, name)
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            coa_file = open(join(file_path, 'paper_author.txt'), 'w', encoding='utf-8')
            cov_file = open(join(file_path, 'paper_venue.txt'), 'w', encoding='utf-8')
            cot_file = open(join(file_path, 'paper_title.txt'), 'w', encoding='utf-8')
            coo_file = open(join(file_path, 'paper_org.txt'), 'w', encoding='utf-8')
            authorname_dict = {} # maintain a author-name-dict

            with open(join(os.getcwd(), 'names_pub', mode, name+'.json'), 'r', encoding='utf-8') as f:
                pubs_dict = json.load(f)

            ori_name = name
            name, name_reverse = unify_name_order(name)

            for i, pid in enumerate(pubs_dict):
                pub = pubs_dict[pid]

                # Save title (relations)
                title = pub["title"]
                pstr = title.strip()
                pstr = pstr.lower()
                pstr = re.sub(r, ' ', pstr)
                pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
                pstr = pstr.split(' ')
                pstr = [word for word in pstr if len(word) > 1]
                pstr = [word for word in pstr if word not in stopwords]
                pstr = [word for word in pstr if word not in stopwords_check]
                for word in pstr:
                    cot_file.write(pid + '\t' + word + '\n')

                # Save keywords
                word_list = []
                if "keywords" in pub:
                    for word in pub["keywords"]:
                        word_list.append(word)
                    pstr = " ".join(word_list)
                    pstr = re.sub(' +', ' ', pstr)
                keyword = pstr

                # Save org (relations)
                org = ""
                find_author = False
                for author in pub["authors"]:
                    authorname = ''.join(filter(str.isalpha, author['name'])).lower()
                    token = authorname.split(" ")
                    if len(token) == 2:
                        authorname = token[0] + token[1]
                        authorname_reverse = token[1] + token[0]
                        if authorname not in authorname_dict:
                            if authorname_reverse not in authorname_dict:
                                authorname_dict[authorname] = 1
                            else:
                                authorname = authorname_reverse
                    else:
                        authorname = authorname.replace(" ", "")
                    
                    if authorname != name and authorname != name_reverse:
                        coa_file.write(pid + '\t' + authorname + '\n')  # current name is a name of co-author
                    else:
                        if "org" in author:
                            org = author["org"]  # current name is a name for disambiguating
                            find_author = True

                if not find_author:
                    for author in pub['authors']:
                        if match_name(author['name'], ori_name):
                            org = author['org']
                            break

                pstr = org.strip()
                pstr = pstr.lower()
                pstr = re.sub(puncs, ' ', pstr)
                pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
                pstr = pstr.split(' ')
                pstr = [word for word in pstr if len(word) > 1]
                pstr = [word for word in pstr if word not in stopwords]
                pstr = [word for word in pstr if word not in stopwords_extend]
                pstr = set(pstr)
                for word in pstr:
                    coo_file.write(pid + '\t' + word + '\n')
                
                # Save venue (relations)
                if pub["venue"]:
                    pstr = pub["venue"].strip()
                    pstr = pstr.lower()
                    pstr = re.sub(puncs, ' ', pstr)
                    pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
                    pstr = pstr.split(' ')
                    pstr = [word for word in pstr if len(word) > 1]
                    pstr = [word for word in pstr if word not in stopwords]
                    pstr = [word for word in pstr if word not in stopwords_extend]
                    pstr = [word for word in pstr if word not in stopwords_check]
                    for word in pstr:
                        cov_file.write(pid + '\t' + word + '\n')
                    if len(pstr) == 0:
                        cov_file.write(pid + '\t' + 'null' + '\n')

            coa_file.close()
            cov_file.close()
            cot_file.close()
            coo_file.close()
        print(f'Finish {mode} data extracted.')
    print(f'All paper features extracted.')
