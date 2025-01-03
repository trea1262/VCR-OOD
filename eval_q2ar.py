"""
You can use this script to evaluate prediction files (valpreds.npy). Essentially this is needed if you want to, say,
combine answer and rationale predictions.
"""

import numpy as np
import json
import os
from config import VCR_ANNOTS_DIR
import argparse

parser = argparse.ArgumentParser(description='Evaluate question -> answer and rationale')
parser.add_argument(
    '-answer_preds',
    dest='answer_preds',
    default='/data/zoujy/MSGT/saves/vcr/answer1/valpreds1.npy',
    help='Location of question->answer predictions',
    type=str,
)
parser.add_argument(
    '-rationale_preds',
    dest='rationale_preds',
    default='/data/zoujy/MSGT/saves/vcr/rationale1/valpreds1.npy',
    help='Location of question+answer->rationale predictions',
    type=str,
)
parser.add_argument(
    '-split',
    dest='split',
    default='val',
    help='Split you\'re using. Probably you want val.',
    type=str,
)

args = parser.parse_args()

answer_preds = np.load(args.answer_preds)
rationale_preds = np.load(args.rationale_preds)

rationale_labels = []
answer_labels = []

with open(os.path.join(VCR_ANNOTS_DIR, '{}.jsonl'.format(args.split)), 'r') as f:
    for i, l in enumerate(f):
        if i == 26528: #1680:#2732:#1040:#2720:  # Skip the last line1040:#
            break
    #if a < 1044:
        #for l in f:
        
        item = json.loads(l)
        answer_labels.append(item['answer_label'])
        rationale_labels.append(item['rationale_label'])
            

answer_labels = np.array(answer_labels)
rationale_labels = np.array(rationale_labels)
#rationale_preds = rationale_preds[:1680,:]
#answer_preds = answer_preds[:1040,:]
print(answer_labels.shape)
print(rationale_labels.shape)
print(answer_preds.shape)
print(rationale_preds.shape)

# Sanity checks
assert answer_preds.shape[0] == answer_labels.size
assert rationale_preds.shape[0] == answer_labels.size
assert answer_preds.shape[1] == 4
assert rationale_preds.shape[1] == 4

answer_hits = answer_preds.argmax(1) == answer_labels
rationale_hits = rationale_preds.argmax(1) == rationale_labels
joint_hits = answer_hits & rationale_hits

print("Answer acc:    {:.3f}".format(np.mean(answer_hits)), flush=True)
print("Rationale acc: {:.3f}".format(np.mean(rationale_hits)), flush=True)
print("Joint acc:     {:.3f}".format(np.mean(answer_hits & rationale_hits)), flush=True)
