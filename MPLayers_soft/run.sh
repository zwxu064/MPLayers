CUDA_VISIBLE_DEVICES="2" python main.py --img_name="tsukuba" \
                                        --mode="TRWP" \
                                        --n_dir=4 \
                                        --n_iter=5 \
                                        --context="TL" \
                                        --rho=0.5 \
                                        --truncated=2 \
                                        --n_disp=16 \
                                        --p_weight=10 \
                                        --enable_cuda \
                                        --enable_display