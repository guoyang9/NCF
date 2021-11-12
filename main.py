import os
import time
import argparse
import random
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

import model
import config
import evaluate
import data_utils
import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="training epochs")
    parser.add_argument("--top_k", type=int, default=10, help="compute metrics@top_k")
    parser.add_argument("--factor_num", type=int, default=32, help="predictive factors numbers in the model")
    parser.add_argument("--num_layers", type=int, default=3, help="number of layers in MLP model")
    parser.add_argument("--num_ng", type=int, default=4, help="sample negative items for training")
    parser.add_argument("--test_num_ng", type=int, default=99, help="sample part of negative items for testing")
    parser.add_argument("--out", default=True, help="save model or not")
    parser.add_argument('--fix-seed', action='store_true', help='Whether to fix random seed')
    args = parser.parse_args()

    """
    utils.set_logger(os.path.join(params.model_dir, 'train.log'))
    logger = logging.getLogger(f'TS.{args.model}')

    # use GPU if available
    cuda_exist = torch.cuda.is_available()

    # Set random seeds for reproducible experiments if necessary
    if args.fix_seed:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
    if cuda_exist:
        params.device = torch.device('cuda:0')
        logger.info('Using Cuda...')
        torch.backends.cudnn.benchmark = True
        if args.fix_seed:
            torch.cuda.manual_seed_all(0)
        model = net.Net(data_formatter, params).cuda()
    else:
        params.device = torch.device('cpu')
        logger.info('Not using cuda...')
        model = net.Net(data_formatter, params)
    """

    ############################## PREPARE DATASET ##########################
    train_data, test_data, user_num, item_num, train_mat = data_utils.load_all()

    # construct the train and test datasets
    train_dataset = data_utils.NCFData(
        train_data, item_num, train_mat, args.num_ng, True)
    test_dataset = data_utils.NCFData(
        test_data, item_num, num_ng=0, is_training=False)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = data.DataLoader(test_dataset,
                                  batch_size=args.test_num_ng + 1, shuffle=False, num_workers=0)

    ########################### CREATE MODEL #################################
    if config.model == 'NeuMF-pre':
        assert os.path.exists(config.GMF_model_path), 'lack of GMF model'
        assert os.path.exists(config.MLP_model_path), 'lack of MLP model'
        GMF_model = torch.load(config.GMF_model_path)
        MLP_model = torch.load(config.MLP_model_path)
    else:
        GMF_model = None
        MLP_model = None

    model = model.NCF(user_num, item_num, args.factor_num, args.num_layers,
                      args.dropout, config.model, GMF_model, MLP_model)
    model.cuda()
    loss_function = nn.BCEWithLogitsLoss()

    if config.model == 'NeuMF-pre':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # writer = SummaryWriter() # for visualization

    ########################### TRAINING #####################################
    count, best_hr = 0, 0
    for epoch in trange(args.epochs):
        model.train()  # Enable dropout (if have).
        start_time = time.time()
        train_loader.dataset.ng_sample()

        HR, NDCG = evaluate.metrics(model, test_loader, args.top_k)
        for user, item, label in train_loader:
            user = user.cuda()
            item = item.cuda()
            label = label.float().cuda()

            model.zero_grad()
            prediction = model(user, item)
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()
            # writer.add_scalar('data/loss', loss.item(), count)
            count += 1

        model.eval()
        HR, NDCG = evaluate.metrics(model, test_loader, args.top_k)

        elapsed_time = time.time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " +
              time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

        if HR > best_hr:
            best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
            if args.out:
                if not os.path.exists(config.model_path):
                    os.mkdir(config.model_path)
                torch.save(model,
                           '{}{}.pth'.format(config.model_path, config.model))

    print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(
        best_epoch, best_hr, best_ndcg))
