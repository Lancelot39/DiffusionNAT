# import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset, DistributedSampler

import torch
import json
import psutil
import datasets
from datasets import Dataset as Dataset2
import random


def load_data_text_ddp(
        batch_size,
        seq_len, src_seq_len,
        deterministic=False,
        data_args=None,
        model_emb=None,
        split='train',
        loaded_vocab=None,
        loop=True,
):
    """
    For a dataset, create a generator over (seqs, kwargs) pairs.

    Each seq is an (bsz, len, h) float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for some meta information.

    :param batch_size: the batch size of each returned pair.
    :param seq_len: the max sequence length (one-side).
    :param deterministic: if True, yield results in a deterministic order.
    :param data_args: including dataset directory, num of dataset, basic settings, etc.
    :param model_emb: loaded word embeddings.
    :param loaded_vocab: loaded word vocabs.
    :param loop: loop to get batch data or not.
    """

    print('#' * 30, '\nLoading text data...')

    training_data = get_corpus(data_args, seq_len, src_seq_len, split=split, loaded_vocab=loaded_vocab)

    dataset = TextDataset(
        training_data,
        data_args,
        model_emb=model_emb
    )
    sampler = DistributedSampler(dataset)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,  # 20,
        sampler=sampler,
        # drop_last=True,
        shuffle=not deterministic,
        num_workers=0,
    )
    return data_loader


def load_data_text(
        batch_size,
        seq_len, src_seq_len,
        deterministic=False,
        data_args=None,
        model_emb=None,
        split='train',
        loaded_vocab=None,
        loop=True,
):
    """
    For a dataset, create a generator over (seqs, kwargs) pairs.

    Each seq is an (bsz, len, h) float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for some meta information.

    :param batch_size: the batch size of each returned pair.
    :param seq_len: the max sequence length (one-side).
    :param deterministic: if True, yield results in a deterministic order.
    :param data_args: including dataset directory, num of dataset, basic settings, etc.
    :param model_emb: loaded word embeddings.
    :param loaded_vocab: loaded word vocabs.
    :param loop: loop to get batch data or not.
    """

    print('#' * 30, '\nLoading text data...')

    training_data = get_corpus(data_args, seq_len, src_seq_len, split=split, loaded_vocab=loaded_vocab)

    dataset = TextDataset(
        training_data,
        data_args,
        model_emb=model_emb
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,  # 20,
        # drop_last=True,
        shuffle=not deterministic,
        num_workers=0,
    )
    if loop:
        return infinite_loader(data_loader)
    else:
        # print(data_loader)
        return iter(data_loader)


def infinite_loader(data_loader):
    while True:
        yield from data_loader


def helper_tokenize(sentence_lst, vocab_dict, seq_len, src_seq_len=256):
    # Process.memory_info is expressed in bytes, so convert to megabytes
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    raw_datasets = Dataset2.from_dict(sentence_lst)
    print(raw_datasets)
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    def tokenize_function(examples):
        input_id_x = vocab_dict.encode_token(examples['src'])
        input_id_y = vocab_dict.encode_token(examples['trg'])
        result_dict = {'input_ids': input_id_x, 'decoder_input_ids': input_id_y,
                       'attention_mask': [[1] * len(ele) for ele in input_id_x]}

        return result_dict

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=['src', 'trg'],
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
    print('### tokenized_datasets', tokenized_datasets)
    print('### tokenized_datasets...example', tokenized_datasets['input_ids'][0])
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    def pad_function(group_lst):
        max_length = seq_len
        group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], vocab_dict.pad_token_id, src_seq_len)
        group_lst['attention_mask'] = _collate_batch_helper(group_lst['attention_mask'], 0, src_seq_len)
        group_lst['decoder_input_ids'] = _collate_batch_helper(group_lst['decoder_input_ids'], vocab_dict.pad_token_id,
                                                               max_length)
        return group_lst

    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    lm_datasets = tokenized_datasets.map(
        pad_function,
        batched=True,
        num_proc=1,
        desc=f"padding",
    )

    print(lm_datasets, 'padded dataset')
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    raw_datasets = datasets.DatasetDict()
    raw_datasets['train'] = lm_datasets
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    return raw_datasets


def get_corpus(data_args, seq_len, src_seq_len, split='train', loaded_vocab=None):
    print('#' * 30, '\nLoading dataset {} from {}...'.format(data_args.dataset, data_args.data_dir))

    sentence_lst = {'src': [], 'trg': []}

    if split == 'train':
        print('### Loading form the TRAIN set...')
        path = f'{data_args.data_dir}/train.jsonl'
    elif split == 'valid':
        print('### Loading form the VALID set...')
        path = f'{data_args.data_dir}/valid.jsonl'
    elif split == 'test':
        print('### Loading form the TEST set...')
        path = f'{data_args.data_dir}/test.jsonl'
    else:
        assert False, "invalid split for dataset"

    with open(path, 'r') as f_reader:
        for row in f_reader:
            sentence_lst['src'].append(
                json.loads(row)['src'].strip().replace('[SEP]', '</s>').replace('[UNK]', '<unk>'))
            sentence_lst['trg'].append(
                json.loads(row)['trg'].strip().replace('[SEP]', '</s>').replace('[UNK]', '<unk>'))

    print('### Data samples...\n', sentence_lst['src'][:2], sentence_lst['trg'][:2])

    # get tokenizer.
    vocab_dict = loaded_vocab

    train_dataset = helper_tokenize(sentence_lst, vocab_dict, seq_len, src_seq_len)
    return train_dataset


class TextDataset(Dataset):
    def __init__(self, text_datasets, data_args, model_emb=None):
        super().__init__()
        self.text_datasets = text_datasets
        self.length = len(self.text_datasets['train'])
        self.data_args = data_args
        self.model_emb = model_emb
        self.delete_prob = 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_ids = self.text_datasets['train'][idx]['input_ids']
        input_mask = self.text_datasets['train'][idx]['attention_mask']
        decoder_input_ids = self.text_datasets['train'][idx]['decoder_input_ids']
        out_kwargs = {}
        out_kwargs['input_ids'] = np.array(input_ids)
        out_kwargs['input_mask'] = np.array(input_mask)
        out_kwargs['decoder_input_ids'] = np.array(decoder_input_ids)
        return out_kwargs


def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False):
    result = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    mask_ = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result
