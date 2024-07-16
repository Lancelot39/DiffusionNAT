"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os, json
from tracemalloc import start

import numpy as np
import torch
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from transformers import set_seed
from diffuseq.rounding import denoised_fn_round, get_weights
from diffuseq.text_datasets import load_data_text, load_data_text_ddp
from diffuseq.gaussian_diffusion import GaussianDiffusion, index_to_log_onehot

# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import time
from diffuseq.utils import dist_util, logger
from functools import partial
from basic_utils import (
    load_defaults_config,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    load_model_emb,
    load_tokenizer
)


def create_argparser():
    defaults = dict(model_path='', step=0, out_dir='', top_p=0, hardcoded_pseudo_diralpha=5)
    decode_defaults = dict(split='valid', clamp_step=0, seed2=105, clip_denoised=False)
    defaults.update(load_defaults_config())
    defaults.update(decode_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    print(config_path)
    # sys.setdefaultencoding('utf-8')
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)

    logger.log("### Creating model and diffusion...")
    model, _ = create_model_and_diffusion(
        **args_to_dict(args, load_defaults_config().keys())
    )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'### The parameter count is {pytorch_total_params}')

    model.to(dist_util.dev())
    model = th.nn.parallel.DistributedDataParallel(
        model, device_ids=[dist.get_rank()], output_device=dist.get_rank(), find_unused_parameters=False,
    )
    model.eval()

    tokenizer = load_tokenizer(args)

    set_seed(args.seed2)

    print("### Sampling...on", args.split)

    ## load data
    data_valid = load_data_text_ddp(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        src_seq_len=args.src_seq_len,
        deterministic=True,
        data_args=args,
        split=args.split,
        loaded_vocab=tokenizer,
        model_emb=None,  # using the same embedding wight with training data
        loop=False
    )

    start_t = time.time()
    diffusion = GaussianDiffusion(Timestep=1000)
    # batch, cond = next(data_valid)
    # print(batch.shape)

    model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
    out_dir = os.path.join(args.out_dir, f"{model_base_name.split('.ema')[0]}")
    if not os.path.isdir(out_dir) and dist.get_rank() == 0:
        os.mkdir(out_dir)

    out_path = os.path.join(out_dir, f"ema{model_base_name.split('.ema')[1]}.samples")
    if not os.path.isdir(out_path) and dist.get_rank() == 0:
        os.mkdir(out_path)
    out_path = os.path.join(out_path, f"seed{args.seed2}_step{args.clamp_step}.json")
    # fout = open(out_path, 'a')

    from tqdm import tqdm

    for cond in tqdm(data_valid):
        decoder_input_ids = cond.pop('decoder_input_ids').to(dist_util.dev())
        input_ids = cond.pop('input_ids').to(dist_util.dev())
        input_mask = cond.pop('input_mask').to(dist_util.dev())

        sample = diffusion.sample(model, decoder_input_ids, input_ids, input_mask)  # [batch, len]

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)
        all_sentence = [sample.cpu().numpy() for sample in gathered_samples]

        gathered_input_ids = [th.zeros_like(input_ids) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_input_ids, input_ids)
        all_input_ids = [input_ids.cpu().numpy() for input_ids in gathered_input_ids]

        gathered_decoder_input_ids = [th.zeros_like(decoder_input_ids) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_decoder_input_ids, decoder_input_ids)
        all_decoder_input_ids = [decoder_input_ids.cpu().numpy() for decoder_input_ids in gathered_decoder_input_ids]
        # print('sampling takes {:.2f}s .....'.format(time.time() - start_t))

        word_lst_recover = []
        word_lst_ref = []
        word_lst_source = []

        sample = np.concatenate(all_sentence, axis=0)
        input_ids = np.concatenate(all_input_ids, axis=0)
        decoder_input_ids = np.concatenate(all_decoder_input_ids, axis=0)
        # tokenizer = load_tokenizer(args)

        for seq in sample:
            tokens = tokenizer.decode_token(seq)
            word_lst_recover.append(tokens)

        for seq, input_seq in zip(decoder_input_ids, input_ids):
            word_lst_source.append(tokenizer.decode_token(input_seq))
            word_lst_ref.append(tokenizer.decode_token(seq))

        if dist.get_rank() == 0:
            fout = open(out_path, 'a')
            for (recov, ref, src) in zip(word_lst_recover, word_lst_ref, word_lst_source):
                print(json.dumps({"recover": recov, "reference": ref, "source": src}), file=fout)
            fout.close()

    print('### Total takes {:.2f}s .....'.format(time.time() - start_t))
    print(f'### Written the decoded output to {out_path}')


if __name__ == "__main__":
    main()
