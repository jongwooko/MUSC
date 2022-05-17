import os
import json
import googletrans
import argparse

from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tgt_lang", type=str, default="es")
    
    conf = parser.parse_args()
    tgt_lang = conf.tgt_lang
    
    if tgt_lang not in ['ja', 'fr', 'de', 'zh', 'es']:
        raise ValueError
        
    translator = googletrans.Translator()
    root_dir = './marc/'
    train_en_json_dir = os.path.join(root_dir, 'train', 'dataset_en_train.json')
    tgt_dir = train_en_json_dir.replace('en', 'en_{}'.format(tgt_lang))
    
    new_json = []
    with open(train_en_json_dir, "r", encoding="utf-8") as f:
        for idx, line in tqdm(enumerate(f.readlines())):
            line = json.loads(line.strip())
            new_line = line

            title = line["review_title"].strip()
            review = line["review_body"].strip()

            new_line["trans_review_title"] = translator.translate(title, src='en', dest='ja').text
            new_line["trans_review_body"] = translator.translate(review, src='en', dest='ja').text

            new_json.append(new_line)

            if (idx+1)%4000 == 0:
                tmp_tgt_dir = tgt_dir.replace('train.json', 'train_{}.json'.format((idx+1)//4000))
                with open(tmp_tgt_dir, encoding="utf-8", mode="w") as file:
                    for i in new_json:
                        file.write(json.dumps(i) + "\n")
                new_json = []