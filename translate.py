import os
import json
import argparse
import time
import torch

from easynmt import EasyNMT
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tgt_lang", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    
    conf = parser.parse_args()
    tgt_lang = conf.tgt_lang
    
    if tgt_lang not in ['ja', 'fr', 'de', 'zh', 'es']:
        raise ValueError
        
    save_iter = 5000
    translator = EasyNMT('m2m_100_418M', device=conf.device)

#     root_dir = '/input/jongwooko/xlt/data/download/marc/'
    root_dir = './marc/'
    train_en_json_dir = os.path.join(root_dir, 'test', 'dataset_{}_test.json'.format(tgt_lang))
    tgt_dir = train_en_json_dir.replace('{}'.format(tgt_lang), '{}_en'.format(tgt_lang))

    new_json = []
    with open(train_en_json_dir, "r", encoding="utf-8") as f:
        for idx, line in tqdm(enumerate(f.readlines())):
            line = json.loads(line.strip())
            new_line = line

            title = line["review_title"].strip()
            review = line["review_body"].strip()
                
            with torch.no_grad():
                new_line["trans_review_title"] = translator.translate(title, source_lang=tgt_lang, target_lang='en')
                new_line["trans_review_body"] = translator.translate(review, source_lang=tgt_lang, target_lang='en')

            new_json.append(new_line)

            if (idx+1)%save_iter == 0:
#                 tmp_tgt_dir = tgt_dir.replace('train.json', 'train_{}.json'.format((idx+1)//save_iter))
                with open(tgt_dir, encoding="utf-8", mode="w") as file:
                    for i in new_json:
                        file.write(json.dumps(i) + "\n")
                new_json = []