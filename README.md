# DeosciRec
Deoscillated Graph Collaborative Filtering

This code is based on Tensorflow 1.14 and python 3.6.

script to run DGCF on ml100k dataset:
``python DGCF_osci.py --dataset ml100k --model_type DGCF_osci --alg_type dgcf --epoch 500 --regs [0.01]  --lr 0.001 --batch_size 512 --stop_step 5 --embed_size 64 --layer_size [64,64,64,64]``
