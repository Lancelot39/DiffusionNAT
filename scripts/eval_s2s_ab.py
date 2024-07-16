import os, sys, glob, json
import numpy as np
import argparse
import torch

import files2rouge

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from nltk import word_tokenize


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='decoding args.')
    parser.add_argument('--folder', type=str, default='', help='path to the folder of decoded texts')
    parser.add_argument('--mbr', action='store_true', help='mbr decoding or not')
    parser.add_argument('--sos', type=str, default='<s>', help='start token of the sentence')
    parser.add_argument('--eos', type=str, default='</s>', help='end token of the sentence')
    parser.add_argument('--sep', type=str, default='</s>', help='sep token of the sentence')
    parser.add_argument('--pad', type=str, default='[PAD]', help='pad token of the sentence')

    args = parser.parse_args()

    files = sorted(glob.glob(f"{args.folder}/*json"))
    sample_num = 0
    with open(files[0], 'r') as f:
        for row in f:
            sample_num += 1

    sentenceDict = {}
    referenceDict = {}
    sourceDict = {}
    for i in range(sample_num):
        sentenceDict[i] = []
        referenceDict[i] = []
        sourceDict[i] = []

    div4 = []
    selfBleu = []

    for path in files:
        print(path)
        sources = []
        references = []
        recovers = []
        bleu1 = []
        bleu2 = []
        rougel = []
        avg_len = []
        dist1 = []
        dist2 = []

        with open(path, 'r') as f:
            cnt = 0
            for row in f:
                source = json.loads(row)['source'].strip()
                reference = json.loads(row)['reference'].strip()
                recover = json.loads(row)['recover'].strip()
                source = source.replace(args.eos, '').replace(args.sos, '')
                reference = reference.replace(args.eos, '').replace(args.sos, '').replace(args.sep, '')
                recover = recover.replace(args.eos, '').replace(args.sos, '').replace(args.sep, '').replace(args.pad, '')
                reference = reference.lower()   #' '.join(word_tokenize(reference)).lower()
                recover = recover.lower()   #' '.join(word_tokenize(recover)).lower()

                sources.append(source)
                references.append(reference+'\n')
                recovers.append(recover+'\n')
                cnt += 1

        ref_f=open('reference.txt', 'w')
        ref_f.writelines(references)
        ref_f.close()

        rec_f=open('recover.txt', 'w')
        rec_f.writelines(recovers)
        rec_f.close()

        files2rouge.run('reference.txt', 'recover.txt')

