import os
import argparse
import random
import logging
import importlib
import numpy as np

import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn

import evaluate
import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ml-1m', help='Name of the dataset')
    parser.add_argument('--dataloader', default='ncf_default', help='Which data loader to use')
    parser.add_argument('--model', default='ncf', choices=utils.model_list(), help='Which model to use')
    parser.add_argument('--model-dir', default='base_model',
                        help='Directory containing params.json, and training results')
    parser.add_argument('--param-set', default=None, help='Set of model parameters created for hypersearch')
    parser.add_argument('--log-output', action='store_true', help='Whether to save the run logs using tensorboard')
    parser.add_argument('--fix-seed', action='store_true', help='Whether to fix random seed')
    parser.add_argument('--save-best', action='store_true', help='Whether to save best ND to param_search.txt')
    parser.add_argument('--restore-file', default=None,
                        help='Optional, name of the file in --model_dir containing weights to reload before \
                        training')  # 'best' or 'epoch_#'
    args = parser.parse_args()

    # Load the parameters from json file
    data_dir = os.path.join('data', args.dataset, args.dataloader)
    data_json_path = os.path.join(data_dir, 'params.json')
    assert os.path.isfile(data_json_path), f'No data json config file found at {data_json_path}'
    params = utils.Params(data_json_path)

    if args.param_set is not None:
        params.model_dir = os.path.join('experiments', args.model_dir, args.dataset, args.dataloader, args.model,
                                        args.param_set)
    else:
        params.model_dir = os.path.join('experiments', args.model_dir, args.dataset, args.dataloader, args.model)
    json_path = os.path.join(params.model_dir, 'params.json')
    assert os.path.isfile(json_path), f'No model json configuration file found at {json_path}'
    params.update(json_path)

    utils.set_logger(os.path.join(params.model_dir, 'train.log'))
    logger = logging.getLogger(f'RMD.main')

    params.dataset = args.dataset
    params.model = args.model
    params.plot_dir = os.path.join(params.model_dir, 'figures')
    if args.param_set is not None:
        params.plot_title = os.path.join(args.model_dir, args.dataset, args.dataloader, args.model,
                                         args.param_set)
    else:
        params.plot_title = os.path.join(args.model_dir, args.dataset, args.dataloader, args.model)

    net = importlib.import_module(f'models.{args.model}')
    data_loader = importlib.import_module(f'data.{args.dataset}.{args.dataloader}.dataloader')

    # create missing directories
    try:
        os.mkdir(params.plot_dir)
    except FileExistsError:
        pass

    print('Building the datasets...')
    train_data, test_data, params.user_num, params.item_num, train_mat = data_loader.load_all(params)

    # construct the train and test datasets
    train_dataset = data_loader.TrainSet(features=train_data, num_item=params.item_num, train_mat=train_mat,
                                         num_ng=params.num_ng)
    test_dataset = data_loader.TestSet(features=test_data, num_item=params.item_num, num_ng=0)
    train_loader = data.DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=8)
    test_loader = data.DataLoader(test_dataset, batch_size=params.test_num_ng + 1, shuffle=False, num_workers=0)

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
    else:
        params.device = torch.device('cpu')
        logger.info('Not using cuda...')

    params.log_output = args.log_output
    final_model = net.train(params, evaluate.metrics, train_loader, test_loader)
