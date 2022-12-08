import numpy as np
import pandas as pd
import json
import re
import sys


def cleanup(s):
    return s.replace('\\n', ' \\n ').replace('\\r', '')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    try:
        with open(sys.argv[1]) as f:
            data = json.load(f)
    except IndexError as e:
        sys.stderr.write("Please specify data\n")
        exit(-1)

    if len(sys.argv) > 2:
        out_f = sys.argv[2]
    else:
        out_f = './data.crf'

    with open(out_f, 'w') as f:
        for signature in data:
            text = signature['text']
            for token in signature['annotations']:
                begin = token['begin']
                end = token['end']

                before = text[:begin]
                after = text[end:]
                line = len(re.findall(r'\\n', before))

                immediate_before = before.split('\\n')[-1] if len(before) != 0 else ""
                immediate_after = after.split('\\n')[0] if len(after) != 0 else ""

                f.write('|'.join([cleanup(before), immediate_before, token['form'], immediate_after, cleanup(after), str(line), token['label']]) + '.\n')
