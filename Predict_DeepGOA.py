#-*- encoding:utf8 -*-

import fire
import os
import time
import math

import pickle
import pandas as pd
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.optim import lr_scheduler
from torch.nn.init import xavier_normal,xavier_normal_
from torch import nn

from utils.config import DefaultConfig
from models.DeepGOA_model import DeepGOA
from data import data_generator
from utils.evaluation import  compute_mcc, compute_roc, compute_performance



configs = DefaultConfig()
THREADHOLD = configs.THREADHOLD

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def weight_init(m):
    if isinstance(m,nn.Conv2d):
        xavier_normal_(m.weight.data)
    elif isinstance(m,nn.Linear):
        xavier_normal_(m.weight.data)
    
def test(gpu_tag, model, loader,class_tag,data_flag):

    # Model on eval mode
    model.eval()
    length = len(loader)
    result = []
    all_trues = []
    all_gos = []
    for batch_idx, (input,embed_input,interpro_info,target,gos) in enumerate(loader):
        with torch.no_grad():
            if gpu_tag and torch.cuda.is_available():
                input_var = torch.autograd.Variable(input.cuda(async=True))
                embed_input_var = torch.autograd.Variable(embed_input.cuda(async=True))
                interpro_info_var = torch.autograd.Variable(interpro_info.cuda(async=True))
                target_var = torch.autograd.Variable(target.cuda(async=True))
            else:
                input_var = torch.autograd.Variable(input)
                embed_input_var = torch.autograd.Variable(embed_input)
                interpro_info_var = torch.autograd.Variable(interpro_info)
                target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var, embed_input_var, interpro_info_var)
        result.append(output.data.cpu().numpy())
        all_trues.append(target)
        all_gos.extend(gos)

    if data_flag=="cafa":
        result_file = "{1}/cafa_predict_{0}_result.pkl".format(class_tag,path_dir)
    else:
        result_file = "{1}/swiss_predict_{0}_result.pkl".format(class_tag,path_dir)
    with open(result_file,"wb") as fp:
        pickle.dump(result,fp)

    #caculate
    all_trues = np.concatenate(all_trues, axis=0)
    all_preds = np.concatenate(result, axis=0)
    print("all_trues:", all_trues.shape)
    print("all_preds:", all_preds.shape)
    print("all_gos:", len(all_gos))
    f_max, p_max, r_max, t_max, predictions_max = compute_performance(all_preds, all_trues,all_gos)

    if data_flag == "cafa":
        result_file = "{1}/cafa_target_predictions_max_{0}.pkl".format(class_tag, path_dir)
    else:
        result_file = "{1}/swiss_target_predictions_max_{0}.pkl".format(class_tag, path_dir)
    with open(result_file, "wb") as fp:
        pickle.dump(predictions_max, fp)


    auc = compute_roc(all_preds, all_trues)
    if predictions_max is not None:
        mcc = compute_mcc(predictions_max, all_trues)
    else:
        mcc = 0
    print("F_max\tPrecision\tRecall\tmcc\tauc\tthreadhold")
    if data_flag=="cafa":
        print_result = "{0:.2f}\t{1:.2f}\t{2:.2f}\t{3:.2f}\t{4:.2f}\t:{5:.2f}\n".format(
            f_max, p_max, r_max, mcc, auc, t_max)
    else:
        print_result = "{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}\t{4:.3f}\t{5:.3f}\n".format(
            f_max, p_max, r_max,mcc,auc,t_max)
    print(print_result)


def predict(model_file,interPro_size,class_tag,data_flag="swiss",gpu_tag=0):

    if data_flag=="swiss":
        test_sequences_file = 'data_cache/test_{0}_all_sequences.pkl'.format(class_tag)
        test_features_file = 'data_cache/test_{0}_sequence_features.pkl'.format(class_tag)
        test_net_embedding_file = 'data_cache/filter_test_{0}.embedding.pkl'.format(class_tag)
        test_gos_file = 'data_cache/test_{0}_gos.pkl'.format(class_tag)
        test_label_file = 'data_cache/test_{0}_label.pkl'.format(class_tag)
    else:
        test_features_file = 'data_cache/cafa_test_{0}_sequence_features.pkl'.format(class_tag)
        test_sequences_file = 'data_cache/cafa_test_{0}_all_sequences.pkl'.format(class_tag)
        test_net_embedding_file = 'data_cache/filter_cafa_test_{0}.embedding.pkl'.format(class_tag)
        test_gos_file = 'data_cache/cafa_all_{0}_gos.pkl'.format(class_tag)
        test_label_file = 'data_cache/cafa_test_label_{0}.pkl'.format(class_tag)
    embeddings_file = 'utils/embeddings.npy'
    # parameters
    batch_size = configs.batch_size

    # Datasets
    print(test_gos_file,test_label_file)
    test_dataSet = data_generator.dataSet_V2(test_sequences_file, test_label_file, test_gos_file, embeddings_file,
                                             test_net_embedding_file, test_features_file)

    test_loader = torch.utils.data.DataLoader(test_dataSet, batch_size=3, shuffle=False,
                                              pin_memory=(torch.cuda.is_available()), num_workers=1,drop_last=False)

    # Models
    class_nums = configs.class_nums[class_tag]
    model = DeepGOA(class_nums, interPro_size, class_tag)
    model.load_state_dict(torch.load(model_file))
    if gpu_tag and torch.cuda.is_available():
        print("ok")
        model = model.cuda()
    print("test_{0}".format(class_tag))
    test(gpu_tag, model, test_loader, class_tag,data_flag=data_flag)

    print('Done!')

if __name__ == '__main__':

    path_dir = "./saved_model/DeepGOA"
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    gpu_tag = 1  
    data_flag = "swiss"
    #data_flag = "cafa"
    interPro_size = configs.interPro_size
    for tag in ['mf','bp','cc']:
            model_file = "{1}/DeepGOA_{0}_model.dat".format(tag,path_dir)
            print(model_file)
            predict(model_file,interPro_size[tag][0],tag,data_flag,gpu_tag)

