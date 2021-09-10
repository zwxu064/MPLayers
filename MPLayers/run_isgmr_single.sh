#!/bin/bash

# when enable_sgm_single is enabled,
# n_iter will be changed to 1 internally

python3 main.py \
--img_name="tsukuba" \
--context="TL" \
--mode="ISGMR" \
--n_dir=16 \
--n_iter=5 \
--p_weight=20 \
--n_disp=16 \
--truncated=2 \
--enable_sgm_single