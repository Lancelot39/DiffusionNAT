import os, sys, glob, json
import numpy as np
import argparse
import torch

import files2rouge

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from nltk import word_tokenize

def get_bleu(recover, reference):
    return sentence_bleu([recover.split()], reference.split(), smoothing_function=SmoothingFunction().method4,)

def selectBest(sentences):
    selfBleu = [[] for i in range(len(sentences))]
    for i, s1 in enumerate(sentences):
        for j, s2 in enumerate(sentences):
            try:
                score = get_bleu(s1, s2)
            except:
                score = 0
            selfBleu[i].append(score)
    for i, s1 in enumerate(sentences):
        selfBleu[i][i] = 0
    idx = np.argmax(np.sum(selfBleu, -1))
    return sentences[idx]

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
                reference = ' '.join(word_tokenize(reference)).lower()
                recover = ' '.join(word_tokenize(recover)).lower()

                sentenceDict[cnt].append(recover)
                referenceDict[cnt].append(reference)
                sourceDict[cnt].append(source)

                sources.append(source)
                references.append(reference+'\n')
                recovers.append(recover+'\n')
                cnt += 1
        '''
        ref_f=open('reference.txt', 'w')
        ref_f.writelines(references)
        ref_f.close()

        rec_f=open('recover.txt', 'w')
        rec_f.writelines(recovers)
        rec_f.close()

        files2rouge.run('reference.txt', 'recover.txt')
        '''

    if len(files) > 1:
        print('*' * 30)
        print('MBR...')
        print('*' * 30)
        bleu = []
        rougel = []
        avg_len = []
        dist1 = []
        recovers = []
        references = []
        sources = []

        for k, v in sentenceDict.items():
            if len(v) == 0 or len(referenceDict[k]) == 0:
                continue

            recovers.append(selectBest(v)+'\n')
            references.append(referenceDict[k][0]+'\n')
            sources.append(sourceDict[k][0]+'\n')

        ref_f = open('reference.txt', 'w')
        ref_f.writelines(references)
        ref_f.close()

        rec_f = open('recover.txt', 'w')
        rec_f.writelines(recovers)
        rec_f.close()

        files2rouge.run('reference.txt', 'recover.txt')