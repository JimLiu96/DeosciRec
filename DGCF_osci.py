import tensorflow as tf
import os
import sys
from utility.helper import *
from utility.batch_test import *
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

tf.random.get_seed(1)

class DGCF(object):
    def __init__(self, data_config, pretrain_data):
        # argument settings
        self.model_type = args.model_type
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type

        self.pretrain_data = pretrain_data

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.n_fold = 100

        self.norm_adj = data_config['norm_adj']
        self.cross_adj = data_config['cross_adj']
        self.n_nonzero_elems = self.norm_adj.count_nonzero()

        self.lr = args.lr

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)

        self.model_type += '_%s_%s_l%d' % (self.adj_type, self.alg_type, self.n_layers)

        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.local_factor = args.local_factor

        self.verbose = args.verbose

        '''
        *********************************************************
        Create Placeholder for Input Data & Dropout.
        '''
        # placeholder definition
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        # dropout: node dropout (adopted on the ego-networks);
        #          ... since the usage of node dropout have higher computational cost,
        #          ... please use the 'node_dropout_flag' to indicate whether use such technique.
        #          message dropout (adopted on the convolution operations).
        self.node_dropout_flag = args.node_dropout_flag
        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights).
        """
        # initialization of model parameters
        self.weights = self._init_weights()

        """
        *********************************************************
        Compute Graph-based Representations of all users & items via Message-Passing Mechanism of Graph Neural Networks.
        Different Convolutional Layers:
            1. ngcf: defined in 'Neural Graph Collaborative Filtering', SIGIR2019;
            2. gcn:  defined in 'Semi-Supervised Classification with Graph Convolutional Networks', ICLR2018;
            3. gcmc: defined in 'Graph Convolutional Matrix Completion', KDD2018;
        """
        if self.alg_type in ['lightgcn']:
            self.ua_embeddings, self.ia_embeddings = self._create_lightgcn_embed()

        elif self.alg_type in ['dgcf']:
            self.ua_embeddings, self.ia_embeddings = self._create_dgcf_embed()

        elif self.alg_type in ['dgcf-wochp']:
            self.ua_embeddings, self.ia_embeddings = self._create_dgcf_embed_wochp()

        elif self.alg_type in ['dgcf-wola']:
            self.ua_embeddings, self.ia_embeddings = self._create_dgcf_embed_wola()

        elif self.alg_type in ['dgcf_final']: # only keep the last layers
            print('using final embeddings ')
            self.ua_embeddings, self.ia_embeddings = self._create_dgcf_embed_final()    

        elif self.alg_type in ['gcn']:
            self.ua_embeddings, self.ia_embeddings = self._create_gcn_embed()

        elif self.alg_type in ['gcmc']:
            self.ua_embeddings, self.ia_embeddings = self._create_gcmc_embed()

        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        """
        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)
        self.u_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        self.pos_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
        self.neg_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)

        """
        *********************************************************
        Inference for the testing phase.
        """
        self.batch_ratings = tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings, transpose_a=False, transpose_b=True)

        """
        *********************************************************
        Generate Predictions & Optimize via BPR loss.
        """
        self.mf_loss, self.emb_loss, self.reg_loss = self.create_bpr_loss(self.u_g_embeddings,
                                                                          self.pos_i_g_embeddings,
                                                                          self.neg_i_g_embeddings)
        self.loss = self.mf_loss + self.emb_loss # + self.reg_loss

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
#         self.opt = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _init_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()

        if self.pretrain_data is None:
            all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
            all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')
            print('using xavier initialization')
        else:
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'][:,0:self.emb_dim], trainable=True,
                                                        name='user_embedding', dtype=tf.float32)
            all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embed'][:,0:self.emb_dim], trainable=True,
                                                        name='item_embedding', dtype=tf.float32)
            print('using pretrained embeddings')

        self.weight_size_list = [self.emb_dim] + self.weight_size
        for k in range(self.n_layers):
            all_weights['locality_%d' %k] = tf.Variable(initializer([self.n_users + self.n_items, 1]), name='locality_%d' %k)

        return all_weights

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            # A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat

    def _create_dgcf_embed(self):
        # Generate a set of adjacency sub-matrix.
        if self.node_dropout_flag:
            # node dropout.
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj + self.cross_adj)
            del self.norm_adj, self.cross_adj
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj + self.cross_adj)
            del self.norm_adj, self.cross_adj

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings =  ego_embeddings / (self.n_layers+1)

        for k in range(0, self.n_layers):

            temp_embed = []
            residual_embed = []
            ego_embeddings = tf.multiply(tf.nn.sigmoid(self.weights['locality_%d' % k]), ego_embeddings)
            for f in range(self.n_fold): # save memory
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings)) # sum messages of neighbors.
            ego_embeddings = tf.concat(temp_embed, 0)
            all_embeddings += ego_embeddings / (self.n_layers+1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)

        return u_g_embeddings, i_g_embeddings

    def _create_dgcf_embed_wochp(self):
        # Generate a set of adjacency sub-matrix.
        if self.node_dropout_flag:
            # node dropout.
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
            del self.norm_adj, self.cross_adj
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)
            del self.norm_adj, self.cross_adj

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings =  ego_embeddings / (self.n_layers+1)

        for k in range(0, self.n_layers):

            temp_embed = []
            residual_embed = []
            ego_embeddings = tf.multiply(tf.nn.sigmoid(self.weights['locality_%d' % k]), ego_embeddings)
            for f in range(self.n_fold): # save memory
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings)) # sum messages of neighbors.
            ego_embeddings = tf.concat(temp_embed, 0)
            all_embeddings += ego_embeddings / (self.n_layers+1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)

        return u_g_embeddings, i_g_embeddings

    def _create_dgcf_embed_wola(self):
        # Generate a set of adjacency sub-matrix.
        if self.node_dropout_flag:
            # node dropout.
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
            del self.norm_adj, self.cross_adj
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)
            del self.norm_adj, self.cross_adj

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings =  ego_embeddings / (self.n_layers+1)

        for k in range(0, self.n_layers):

            temp_embed = []
            residual_embed = []
            # ego_embeddings = tf.multiply(tf.nn.sigmoid(self.weights['locality_%d' % k]), ego_embeddings)
            for f in range(self.n_fold): # save memory
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings)) # sum messages of neighbors.
            ego_embeddings = tf.concat(temp_embed, 0)
            all_embeddings += ego_embeddings / (self.n_layers+1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)

        return u_g_embeddings, i_g_embeddings


    def _create_dgcf_embed_final(self):
        # Generate a set of adjacency sub-matrix.
        if self.node_dropout_flag:
            # node dropout.
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj + self.cross_adj)
            del self.norm_adj, self.cross_adj
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj + self.cross_adj)
            del self.norm_adj, self.cross_adj

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings =  ego_embeddings / (self.n_layers+1)

        for k in range(0, self.n_layers):

            temp_embed = []
            residual_embed = []
            ego_embeddings = tf.multiply(tf.nn.sigmoid(self.weights['locality_%d' % k]), ego_embeddings)
            for f in range(self.n_fold): # save memory
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings)) # sum messages of neighbors.
            ego_embeddings = tf.concat(temp_embed, 0)
            # all_embeddings += ego_embeddings / (self.n_layers+1)
        u_g_embeddings, i_g_embeddings = tf.split(ego_embeddings, [self.n_users, self.n_items], 0)

        return u_g_embeddings, i_g_embeddings


    def _create_lightgcn_embed(self):
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj)
            del self.norm_adj
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj)
            del self.norm_adj
        
        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        all_embeddings = [ego_embeddings]
        
        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            side_embeddings = tf.concat(temp_embed, 0)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings=tf.stack(all_embeddings,1)
        all_embeddings=tf.reduce_mean(all_embeddings,axis=1,keepdims=False)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings


    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        
        regularizer = tf.nn.l2_loss(self.u_g_embeddings_pre) + tf.nn.l2_loss(
                self.pos_i_g_embeddings_pre) + tf.nn.l2_loss(self.neg_i_g_embeddings_pre)
        regularizer = regularizer / self.batch_size
        
        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
        

        emb_loss = self.decay * regularizer

        reg_loss = tf.constant(0.0, tf.float32, [1])

        return mf_loss, emb_loss, reg_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)

def load_pretrained_data():
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'embeddings')
    print(pretrain_path)
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained embeddings.')
    except Exception:
        pretrain_data = None
    return pretrain_data

if __name__ == '__main__':

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    plain_adj, norm_adj, norm_adj_no_eye, cross_adj = data_generator.get_adj_mat(args.low, args.high)
    # cross_adj.setdiag(0.0)
    # cross_adj.eliminate_zeros()

    if args.adj_type == 'plain':
        config['norm_adj'] = plain_adj
        print('use the plain adjacency matrix')

    elif args.adj_type == 'laplacian':
        config['norm_adj'] = norm_adj
        print('use the laplacian adjacency matrix')

    elif args.adj_type == 'laplacian_no_eye':
        config['norm_adj'] = norm_adj_no_eye
        print('use the laplacian adjacency matrix without eye')

    else:
        config['norm_adj'] = norm_adj + sp.eye(norm_adj.shape[0])
        print('use the mean adjacency matrix')

    config['cross_adj'] = cross_adj
    t0 = time()

    if args.pretrain == -1:
        pretrain_data = load_pretrained_data()
    else:
        pretrain_data = None

    model = DGCF(data_config=config, pretrain_data=pretrain_data)
    del config, cross_adj

    """
    *********************************************************
    Save the model parameters.
    """
    saver = tf.train.Saver()
    layer = '-'.join([str(l) for l in eval(args.layer_size)])

    if args.save_flag == 1:
        weights_save_path = '%sweights/%s/%s/%s/l%s_r%s_d%s_low%s_high%s' % (args.weights_path, args.dataset, model.model_type, layer,
                                                            str(args.lr), '-'.join([str(r) for r in eval(args.regs)]), str(eval(args.node_dropout)[0]),
                                                            str(args.low), str(args.high))
        ensureDir(weights_save_path)
        save_saver = tf.train.Saver(max_to_keep=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    """
    *********************************************************
    Reload the pretrained model parameters.
    """
    if args.pretrain == 1:
        pretrain_path = '%sweights/%s/%s/%s/l%s_r%s_d%s_low%s_high%s' % (args.weights_path, args.dataset, model.model_type, layer,
                                                            str(args.lr), '-'.join([str(r) for r in eval(args.regs)]), str(eval(args.node_dropout)[0]),
                                                            str(args.low), str(args.high))

        ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('load the pretrained model parameters from: ', pretrain_path)

            # *********************************************************
            # get the performance from pretrained model.
            if args.report != 1:
                users_to_test = list(data_generator.test_set.keys())
                ret = test(sess, model, users_to_test, drop_flag=True)
                cur_best_pre_0 = ret['recall'][0]

                pretrain_ret = "recall=[%s], precision=[%s], hit=[%s], ndcg=[%s]"% \
                 ('\t'.join(['%.5f' % r for r in ret['recall']]),
                  '\t'.join(['%.5f' % r for r in ret['precision']]),
                  '\t'.join(['%.5f' % r for r in ret['hit_ratio']]),
                  '\t'.join(['%.5f' % r for r in ret['ndcg']]))
                print(pretrain_ret)
        else:
            sess.run(tf.global_variables_initializer())
            cur_best_pre_0 = 0.
            print('without pretraining.')

    else:
        sess.run(tf.global_variables_initializer())
        cur_best_pre_0 = 0.
        print('without pretraining weights.')

    """
    *********************************************************
    Get the performance w.r.t. different sparsity levels.
    """
    if args.report == 1:
        assert args.test_flag == 'full'
        users_to_test_list, split_state = data_generator.get_sparsity_split()
        users_to_test_list.append(list(data_generator.test_set.keys()))
        split_state.append('all')

        report_path = '%sreport/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
        ensureDir(report_path)
        f = open(report_path, 'w')
        f.write(
            'embed_size=%d, lr=%.4f, layer_size=%s, keep_prob=%s, regs=%s, loss_type=%s, adj_type=%s\n'
            % (args.embed_size, args.lr, args.layer_size, args.keep_prob, args.regs, args.loss_type, args.adj_type))

        for i, users_to_test in enumerate(users_to_test_list):
            ret = test(sess, model, users_to_test, drop_flag=True)

            final_perf = "recall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                         ('\t'.join(['%.5f' % r for r in ret['recall']]),
                          '\t'.join(['%.5f' % r for r in ret['precision']]),
                          '\t'.join(['%.5f' % r for r in ret['hit_ratio']]),
                          '\t'.join(['%.5f' % r for r in ret['ndcg']]))
            print(final_perf)

            f.write('\t%s\n\t%s\n' % (split_state[i], final_perf))
        f.close()
        exit()

    """
    *********************************************************
    Train.
    """
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    time_loger = []
    stopping_step = 0
    should_stop = False

    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample()
            _, batch_loss, batch_mf_loss, batch_emb_loss, batch_reg_loss = sess.run([model.opt, model.loss, model.mf_loss, model.emb_loss, model.reg_loss],
                               feed_dict={model.users: users, model.pos_items: pos_items,
                                          model.node_dropout: eval(args.node_dropout),
                                          model.mess_dropout: eval(args.mess_dropout),
                                          model.neg_items: neg_items})
            loss += batch_loss/n_batch
            mf_loss += batch_mf_loss/n_batch
            emb_loss += batch_emb_loss/n_batch
            reg_loss += batch_reg_loss/n_batch

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        # print the validation metrics each 10 epochs; pos:neg = 1:10.
        test_epoch = args.test_epoch
        loss_loger.append(loss)
        if (epoch < test_epoch) or (epoch + 1) % args.test_interval != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss)
                print(perf_str)
            continue
        perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss)
        print(perf_str)
        t2 = time()
        time_loger.append(time() - t1)
        print("---------start validation--------------")
        users_to_valid = list(set(data_generator.train_items.keys()).intersection(set(data_generator.valid_set.keys())))
        ret = validate(sess, model, users_to_valid, drop_flag=True)

        t3 = time()

        
        rec_loger.append(ret['recall'][0])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = "recall=[%s], precision=[%s], hit=[%s], ndcg=[%s]"% \
                 ('\t'.join(['%.5f' % r for r in ret['recall']]),
                  '\t'.join(['%.5f' % r for r in ret['precision']]),
                  '\t'.join(['%.5f' % r for r in ret['hit_ratio']]),
                  '\t'.join(['%.5f' % r for r in ret['ndcg']]))
            print(perf_str)
            

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=args.stop_step)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            print('---------find the best validation------------')
            # print("---------start testing--------------")
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
            print('save the weights in path: ', weights_save_path)
            perf_str = "-----Best!-----recall=[%s], precision=[%s], hit=[%s], ndcg=[%s]-------Best!------"% \
                 ('\t'.join(['%.5f' % r for r in ret['recall']]),
                  '\t'.join(['%.5f' % r for r in ret['precision']]),
                  '\t'.join(['%.5f' % r for r in ret['hit_ratio']]),
                  '\t'.join(['%.5f' % r for r in ret['ndcg']]))
            u_embed, i_embed = sess.run([model.u_g_embeddings, model.pos_i_g_embeddings], 
                        feed_dict={model.users:list(range(model.n_users)), 
                        model.pos_items:list(range(model.n_items)), model.node_dropout: [0.]*len(eval(args.layer_size)),
                                                                    model.mess_dropout: [0.]*len(eval(args.layer_size))})
            for k in range(model.n_layers): # store the locality weight for checking
                locality_weights = sess.run(model.weights['locality_%d' %k], feed_dict={model.users:list(range(1)), 
                        model.pos_items:list(range(1)), model.node_dropout: [0.]*len(eval(args.layer_size)),
                                                                    model.mess_dropout: [0.]*len(eval(args.layer_size))})
                locality_file_name = weights_save_path + '/locality_%d.npz'%k
                np.savez(locality_file_name, locality_weights=locality_weights)
                print('save the locality weights at layer %d' %k)
            embed_file_name = weights_save_path + '/embeddings.npz'
            np.savez(embed_file_name, user_embed=u_embed, item_embed=i_embed)
            print('save the embeddings in path:',embed_file_name)
            print(perf_str)
    # begin to test
    print('------------start testing!------------')
    if args.save_flag == 0:
        weights_save_path = '%sweights/%s/%s/%s/l%s_r%s_d%s_low%s_high%s' % (args.weights_path, args.dataset, model.model_type, layer,
                                                            str(args.lr), '-'.join([str(r) for r in eval(args.regs)]), str(eval(args.node_dropout)[0]),
                                                            str(args.low), str(args.high))
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(weights_save_path + '/checkpoint'))
    saver.restore(sess, ckpt.model_checkpoint_path)
    users_to_test = list(set(data_generator.train_items.keys()).intersection(set(data_generator.test_set.keys())))
    ret = test(sess, model, users_to_test, drop_flag=True)
    final_perf = "recall=[%s], precision=[%s], hit=[%s], ndcg=[%s]"% \
         ('\t'.join(['%.5f' % r for r in ret['recall']]),
          '\t'.join(['%.5f' % r for r in ret['precision']]),
          '\t'.join(['%.5f' % r for r in ret['hit_ratio']]),
          '\t'.join(['%.5f' % r for r in ret['ndcg']]))
    print(final_perf)

    save_path = '%soutput/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
    ensureDir(save_path)
    f = open(save_path, 'a')
    f.write(
        'embed_size=%d, lr=%.4f, layer_size=%s, , low=%.4f, high=%.4f, node_dropout=%s, mess_dropout=%s, regs=%s, adj_type=%s\n\t%s\n'
        % (args.embed_size, args.lr, args.layer_size, args.low, args.high, args.node_dropout, args.mess_dropout, args.regs,
           args.adj_type, final_perf))
    f.close()
    out_path = '%strain/%s/%s.train%s' % (args.proj_path, args.dataset, model.model_type, time())
    ensureDir(out_path)
    with open(out_path, 'a') as o_f:
        o_f.write('loss:')
        loss_string = str(loss_loger)
        o_f.write(loss_string)
        o_f.write('\n')
        o_f.write('recall:')
        recall_string = str(rec_loger)
        o_f.write(recall_string)
        o_f.write('\n')
        o_f.write('time:')
        time_string = str(time_loger)
        o_f.write(time_string)
        o_f.write('\n')
