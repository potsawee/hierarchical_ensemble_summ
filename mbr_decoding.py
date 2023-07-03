'''
- Given multiple submission files, select most average sample and save output file
- This script was written by Vyas Raina (vy313@cam.ac.uk)
'''

import sys
import os
import argparse
from tqdm import tqdm
import glob
from rouge_score import rouge_scorer

if __name__ == "__main__":
    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--filedir', type=str, required=False, help='Alternative give dir with all txt files')
    commandLineParser.add_argument('--outfile', type=str, required=True, help='path to save final predictions')
    args = commandLineParser.parse_args()

    # load data
    data = []
    filepaths = glob.glob(f'{args.filedir}/*.txt')
    for fpath in filepaths:
        with open(fpath, 'r') as f:
            summ = f.readlines()
        summ = [s.strip('\n') for s in summ]
        data.append(summ)

    # select samples
    selected_sample = []
    metric = 'rougeL'
    scorer = rouge_scorer.RougeScorer([metric], use_stemmer=True)
    for samples in tqdm(zip(*data)):
        best = [None, 0] # [seed , rouge_score]
        for i in range(len(samples)):
            total = 0
            for j in range(len(samples)):
                score = scorer.score(samples[j], samples[i])
                total += score[metric][2]
            if total > best[1]:
                best = [i, total]
        selected_sample.append(samples[best[0]])

    # save selected samples
    with open(args.outfile, 'w') as f:
        for sample in selected_sample:
            f.write(sample+'\n')
