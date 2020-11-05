# DeosciRec
Deoscillated Graph Collaborative Filtering. [Paper in arxiv](https://arxiv.org/abs/2011.02100)

This code is based on Tensorflow 1.14 and python 3.6.

script to run DGCF on ml100k dataset:
```
python DGCF_osci.py --dataset ml100k --model_type DGCF_osci --alg_type dgcf --epoch 500 --regs [0.01]  --lr 0.001 --batch_size 512 --stop_step 5 --embed_size 64 --layer_size [64,64,64,64]
```

# Notes
In order to train the model on large dataset, the arguement ``--low`` should be applied. For example, on ml1m dataset, one can use the script:
```
python DGCF_osci.py --dataset ratings_ml-1m --model_type DGCF_osci --alg_type dgcf --epoch 500 --regs [0.01]  --lr 0.001 --batch_size 1024 --low 0.01 --stop_step 5 --embed_size 64 --layer_size [64,64,64,64]
```

This argument is to filter the CHP laplacian matrix, where the value < low is filtered. More details can be find in the paper. 

# Acknowledgement
We reuse some part of the code in ``Neural Graph Collaborative Filtering`` <https://github.com/xiangwang1223/neural_graph_collaborative_filtering>
