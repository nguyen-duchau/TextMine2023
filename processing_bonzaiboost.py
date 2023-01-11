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

N_ESTIMATORS = '200'
OVERSAMPLING_CLASSES = {'Email': 4,
                        'Function': 4,
                        'Phone_Number': 2,
                        'Project': 15,
                        'Reference_CEDEX': 15,
                        'Reference_CS': 30,
                        'Reference_Code_Postal': 3,
                        'Reference_User': 30,
                        'Social_Network': 100,
                        'Url': 8}

CLASSES = ['Human', 'Organization', 'Function', 'Project', 'Location', 'Reference_CEDEX', 'Reference_CS',
           'Reference_Code_Postal', 'Phone_Number', 'Email', 'Url', 'Social_Network', 'Reference_User']
BERT_CLASSES = []
# BERT_CLASSES = ['Project', 'Organization', 'Function', 'Location']

FEATURES = {'token': 'text',
            'token_split': 'text',
            'token_lower': 'text',
            'before_on_line': 'text',
            'before_on_line_lower': 'text',
            'before': 'text',
            'after_on_line': 'text',
            'after_on_line_lower': 'text',
            'after': 'text',
            # 'line': 'continuous'
            }

REGEX_M = {
    'mail': r'[\w\-\'\.]+@[\w\-]+[.]\w+',
    'tel': r'(?:[+]\D?)?\d+(?:\D{0,2}\d){6,12}',
    'url': r'[\w\-\.]+[.][a-zA-Z]+',
    'http': r'http.*',
    'handle': r'^@.*',
    'postcode': r'\d{5}',
    'caps': r'^[A-Z]',
    'allcaps': r'^[A-Z]+$',
    'nonword': r'^[\d\W]+$',
    'social': r'Facebook|Linkedin|Instagram|Twitter|Google|Youtube'
}
REGEX_T = {
    'suffix_': r'.?.?.?$'
}
TEXT_PARAMS = {'expert_length': 1,
               'expert_type': 'ngram',
               'cutoff': 0
               }

EVAL_DATA = 'dataset/JDA.json'


def cleanup(s):
    return s.replace('\n', ' \\n ')


def regex_string(name, regex, mode="M", level='0'):
    return 'regex=' + name + '_%' + mode + '[' + level + ',"' + str(regex) + '"]'


def create_names(filename):
    with open(filename, 'w') as f:
        labels_activated = [i for i in CLASSES if i not in BERT_CLASSES]
        if len(BERT_CLASSES) != 0:
            labels_activated += ['BERT']
        f.write(','.join(labels_activated) + '.\n')
        for i in FEATURES:
            if FEATURES[i] == 'text':
                param_str = ' '.join([i + '=' + str(TEXT_PARAMS[i]) for i in TEXT_PARAMS])
                regex_str = ' '.join([regex_string(i, REGEX_M[i], 'M') for i in REGEX_M] +
                                     [regex_string(i, REGEX_T[i], 'T') for i in REGEX_T]
                                     )
                f.write(i + ': text: ' + param_str + ' ' + regex_str + '.\n')
            elif FEATURES[i] == 'continuous':
                f.write(i + ': continuous.\n')


def process_features(token, text):
    # text = text.replace('\n', ' ')

    begin = token['begin']
    end = token['end']

    before = text[:begin]
    after = text[end:]

    before_on_line = before.split('\n')[-1] if len(before) != 0 else ""
    after_on_line = after.split('\n')[0] if len(after) != 0 else ""

    features = {'token': token['form'].replace('\n', ''),
                'token_split': ' '.join(re.split('(\W)', token['form'].replace('\n', ''))),
                'before': cleanup(before),
                'after': cleanup(after),
                'before_on_line': before_on_line,
                'after_on_line': after_on_line,
                'token_lower': token['form'].replace('\n', '').lower(),
                'before_on_line_lower': before_on_line.lower(),
                'after_on_line_lower': after_on_line.lower(),
                'line': str(len(re.findall(r'\n', before)))
                }

    return features


def process_data_train(data, out_file):
    for signature in data:
        text = signature['text']  # .replace('\n', ' \\n ')
        for token in signature['annotations']:
            features = process_features(token, text)

            label = token['label']

            oversampling = OVERSAMPLING_CLASSES.get(label, None)
            oversampling_marker = str(oversampling) + '::' if oversampling is not None else ''

            label_show = label if label not in BERT_CLASSES else 'BERT'

            out_file.write(oversampling_marker + '?'.join([features[i] for i in FEATURES] + [label_show]) + '.\n')


def process_data_predict(data, out_file):
    with open('bonzai_data/predict.log', 'w') as f:
        f.write('')
    for signature in tqdm(data):
        text = signature['text']  # .replace('\n', ' \\n ')
        for token in signature['annotations']:
            features = process_features(token, text)

            line = '?'.join([features[i] for i in FEATURES] + ['']) + '.'

            result = run([
                "bonzaiboost -S " + STEM + " -boost adamh -n " + N_ESTIMATORS + " --sep '?' -C -c single -o backoff -v 0 2> " + 'predict_err.log'],
                shell=True, cwd=FOLDER, stdout=subprocess.PIPE, input=line.encode(), stderr=subprocess.STDOUT)
            pipe_out = result.stdout.decode("utf-8")
            with open('bonzai_data/predict.log', 'a') as f:
                f.write(line + '\n' + pipe_out + '\n')
            prediction = pipe_out.split('->')[1].strip(' []\n')
            token['label_bonzaiboost'] = prediction

    outs = json.dumps(data, ensure_ascii=False)
    out_file.write(outs)


TRAIN = True
EVAL = True
PRED = False

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

        if EVAL:
            # Run cross-validation for an evaluation of the model
            run([
                "bonzaiboost -S " + FOLDER + STEM + " -boost adamh -n " + N_ESTIMATORS + " --sep '?' -cross k10 -jobs 4 -c single -o backoff > " + FOLDER + "crossval.log"],
                shell=True)

        # Train one simple model to be able to extract its features
        run(["bonzaiboost -S " + STEM + " -boost adamh -n " + N_ESTIMATORS + " --sep '?'"], cwd=FOLDER, shell=True)
        run(["bonzaiboost -S " + STEM + " -boost adamh -n " + N_ESTIMATORS + " --sep '?' --info -o 0.005"], cwd=FOLDER,
            shell=True)

    if PRED:
        print("Predictions")
        with open(EVAL_DATA) as in_f:
            print('Loading', EVAL_DATA)
            data = json.load(in_f)
        with open(FOLDER + 'preds.json', 'w', encoding='utf-8') as out_f:
            process_data_predict(data, out_f)
