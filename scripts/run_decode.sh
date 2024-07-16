CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u run_decode.py \
--model_dir diffusion_models/diffuseq_Persona_h32_lr5e-05_t2000_sqrt_lossaware_seed102_Persona20230107-15:55:26 \
--seed 123 \
--split test
