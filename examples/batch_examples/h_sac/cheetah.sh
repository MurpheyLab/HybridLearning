#!/bin/zsh

for i in {1..1}
do
    python3 h_sac.py \
                --env "HalfCheetahEnv" \
                --max_steps 1000 \
                --max_frames 40000 \
                --horizon 10 \
                --frame_skip 4 \
                --lam 0.2 \
                --model_iter 5 \
                --no_render
    echo "trial $i out of 2"
done
