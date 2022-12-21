import subprocess

import numpy as np
import pandas as pd
import json
import re
import sys
from subprocess import run
from tqdm import tqdm

INPUT_DATA = ['dataset/JDR.json',
              'dataset/JDF.json']

FOLDER = './bonzai_data/'
STEM = 'textmine'

OVERSAMPLING_CLASSES = {'Project': 3,
                        'Social_Network': 20,
                        'Reference_User': 30,
                        'Reference_CS': 6,
                        'Reference_CEDEX': 2}

CLASSES = ['Human', 'Organization', 'Function', 'Project', 'Location', 'Reference_CEDEX', 'Reference_CS',
           'Reference_Code_Postal', 'Phone_Number', 'Email', 'Url', 'Social_Network', 'Reference_User']
BERT_CLASSES = []
# BERT_CLASSES = ['Project', 'Organization', 'Function', 'Location']

FEATURES = {'token': 'text',
            'token_split': 'text',
            'before': 'text',
            'before_on_line': 'text',
            'after_on_line': 'text',
            'after': 'text',
            # 'line': 'continuous'
            }

REGEX = {'mail': r'[\w\-\'\.]+@[\w\-]+[.]\w+',
         'tel': r'(?:[+]\D?)?\d+(?:\D{0,2}\d){6,12}',
         'url': r'[\w\-\.]+[.][a-zA-Z]+',
         'http': r'http.*',
         'handle': r'^@.*',
         'postcode': r'\d{5}',
         'caps': r'^[A-Z]',
         'allcaps': r'^[A-Z]+$',
         # 'social': r'Facebook|Linkedin|Instagram|Twitter'
         }
TEXT_PARAMS = {'expert_length': 1,
               'expert_type': 'ngram',
               'cutoff': 0}

EVAL_DATA = 'dataset/JDC_for_eval.json'

def cleanup(s):
    return s.replace('\\n', ' \\n ').replace('\\r', '')


def regex_string(name, regex):
    return 'regex=' + name + '_%M[0,"' + str(regex) + '"]'


def create_names(filename):
    with open(filename, 'w') as f:
        labels_activated = [i for i in CLASSES if i not in BERT_CLASSES]
        if len(BERT_CLASSES) != 0:
            labels_activated += ['BERT']
        f.write(','.join(labels_activated) + '.\n')
        for i in FEATURES:
            if FEATURES[i] == 'text':
                param_str = ' '.join([i + '=' + str(TEXT_PARAMS[i]) for i in TEXT_PARAMS])
                regex_str = ' '.join([regex_string(i, REGEX[i]) for i in REGEX])
                f.write(i + ': text: ' + param_str + ' ' + regex_str + '.\n')
            elif FEATURES[i] == 'continuous':
                f.write(i + ': continuous.\n')


def process_features(token, text):
    begin = token['begin']
    end = token['end']

    before = text[:begin]
    after = text[end:]

    features = {'token': token['form'].replace('\n', ''),
                'token_split': ' '.join(re.split('(\W)', token['form'].replace('\n', ''))),
                'before': cleanup(before),
                'after': cleanup(after),
                'before_on_line': before.split('\\n')[-1] if len(before) != 0 else "",
                'after_on_line': after.split('\\n')[0] if len(after) != 0 else "",
                'line': str(len(re.findall(r'\\n', before)))
                }

    return features


def process_data_train(data, out_file):
    for signature in data:
        text = signature['text'].replace('\n', ' \\n ')
        for token in signature['annotations']:
            features = process_features(token, text)

            label = token['label']

            oversampling = OVERSAMPLING_CLASSES.get(label, None)
            oversampling_marker = str(oversampling) + '::' if oversampling is not None else ''

            label_show = label if label not in BERT_CLASSES else 'BERT'

            out_file.write(oversampling_marker + '|'.join([features[i] for i in FEATURES] + [label_show]) + '.\n')


def process_data_test(data, out_file):
    for signature in tqdm(data):
        text = signature['text'].replace('\n', ' \\n ')
        for token in signature['annotations']:
            features = process_features(token, text)

            line = '|'.join([features[i] for i in FEATURES] + ['']) + '.'

            result = run([
                "bonzaiboost -S " + STEM + " -boost adamh -n 200 --sep '|' -C -c single -o backoff -v 0 2> " + 'noeval.log'],
                shell=True, cwd=FOLDER, stdout=subprocess.PIPE, input=line.encode())
            prediction = result.stdout.decode("utf-8").split('->')[1].strip(' []\n')

            token['label'] = prediction

    outs = json.dumps(data, ensure_ascii=False)
    out_file.write(outs)


TRAIN = True
TEST = False
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    create_names(FOLDER + STEM + '.names')

    if TRAIN:
        print("Training")
        for i, filename in enumerate(INPUT_DATA):
            with open(filename) as in_f:
                print('Loading', filename)
                data = json.load(in_f)
            open_mode = 'w' if i == 0 else 'a'
            with open(FOLDER + STEM + '.data', open_mode) as out_f:
                process_data_train(data, out_f)

        # Run cross-validation for an evaluation of the model
        run([
            "bonzaiboost -S " + FOLDER + STEM + " -boost adamh -n 200 --sep '|' -cross k10 -jobs 4 -c single -o backoff > " + FOLDER + "crossval.log"],
            shell=True)
        # Train one simple model to be able to extract its features
        run(["bonzaiboost -S " + STEM + " -boost adamh -n 200 --sep '|'"], cwd=FOLDER, shell=True)
        run(["bonzaiboost -S " + STEM + " -boost adamh -n 200 --sep '|' --info -o 0.005"], cwd=FOLDER, shell=True)

    if TEST:
        print("Evaluation")
        with open(EVAL_DATA) as in_f:
            print('Loading', EVAL_DATA)
            data = json.load(in_f)
        with open(FOLDER + 'preds.json', 'w', encoding='utf-8') as out_f:
            process_data_test(data, out_f)
