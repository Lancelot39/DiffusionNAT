CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u run_decode_1.py \
--model_dir diffusion_models/diffuseq_Squad_h32_lr5e-05_t2000_sqrt_lossaware_seed102_Squad20230107-06:31:53 \
--seed 123 \
--split test
