'''
Deoscillated Graph Collaborative Filtering
'''
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run NGCF.")
    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default='./Data/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')

    parser.add_argument('--dataset', nargs='?', default='ml100k',
                        help='Choose a dataset from {ml100k, ratings_ml-1m, gowalla}')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=500,
                        help='Number of epoch.')
    parser.add_argument('--residual', type=str, default='None', help='deprecated')
    parser.add_argument('--n_fold', type=int, default=100, help='the number of folds for splitting adj')

    parser.add_argument('--embed_size', type=int, default=128,
                        help='Embedding size.')
    parser.add_argument('--layer_num', type=int, default=None,
                        help='The number of layers')
    parser.add_argument('--layer_size', nargs='?', default='[128,128]',
                        help='Output sizes of every layer')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--test_epoch', type=int, default=1,
                        help='The epoch number for starting to test (validate).')
    parser.add_argument('--test_interval', type=int, default=1,
                        help='The interval epoch number for testing (validating).')

    parser.add_argument('--lareg', type=int, default=0,
                        help='Regularizations for LA layers')
    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]',
                        help='Regularizations.')
    parser.add_argument('--local_factor', type=float, default= 0.5,
                        help='local_factor.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--low', type=float, default= 0.01,
                        help='low pass of the laplacian')
    parser.add_argument('--high', type=float, default= 1.0,
                        help='high stop of the laplacian')

    parser.add_argument('--model_type', nargs='?', default='dgcf',
                        help='Specify the name of model .')
    parser.add_argument('--adj_type', nargs='?', default='laplacian_no_eye',
                        help='Specify the type of the adjacency (laplacian) matrix from {plain, laplacian, laplacian_no_eye}.')
    parser.add_argument('--alg_type', nargs='?', default='dgcf',
                        help='Specify the type of the graph convolutional layer from {ngcf, gcn, gcmc}.')
    parser.add_argument('--interaction', nargs='?', default='deprecated',
                        help='Specify the type of the interaction')

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')

    parser.add_argument('--node_dropout_flag', type=int, default=1,
                        help='0: Disable node dropout, 1: Activate node dropout')
    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]',
                        help='Output sizes of every layer')

    parser.add_argument('--save_flag', type=int, default=1,
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    
    parser.add_argument('--stop_step', type=int, default=5,
                        help='how many steps for get the stop flags')

    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')
    return parser.parse_args()
