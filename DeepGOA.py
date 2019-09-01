#-*- encoding:utf8 -*-

import os
import time


import numpy as np
import torch
from torch.optim import lr_scheduler
from torch.nn.init import xavier_normal,xavier_normal_
from torch import nn
import torch.utils.data.sampler as sampler


from utils.config import DefaultConfig
from models.DeepGOA_model import DeepGOA
from data import data_generator

from utils.evaluation import micro_score, compute_mcc, compute_roc, compute_aupr,compute_performance




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


def train_epoch(model, loader, optimizer, epoch, all_epochs, print_freq=100):
    batch_time = AverageMeter()
    losses = AverageMeter()

    global THREADHOLD
    # Model on train mode
    model.train()

    end = time.time()
    for batch_idx, (input, embed_input, interpro_info, target, gos) in enumerate(loader):

        # Create vaiables
        with torch.no_grad():
            if torch.cuda.is_available():
                input_var = torch.autograd.Variable(input.cuda(async=True))
                interpro_info_var = torch.autograd.Variable(interpro_info.cuda(async=True))
                embed_input_var = torch.autograd.Variable(embed_input.cuda(async=True))
                target_var = torch.autograd.Variable(target.cuda(async=True).float())
            else:
                input_var = torch.autograd.Variable(input)
                interpro_info_var = torch.autograd.Variable(interpro_info)
                embed_input_var = torch.autograd.Variable(embed_input)
                target_var = torch.autograd.Variable(target.float())

        # compute output
        
        output = model(input_var, embed_input_var,interpro_info_var)

        loss = torch.nn.functional.binary_cross_entropy(output, target_var).cuda()

        # measure accuracy and record loss
        batch_size = target.size(0)
        pred_out = output.ge(THREADHOLD)
        MiP, MiR, MiF, PNum, RNum = micro_score(pred_out.data.cpu().numpy(),
                                                target_var.data.cpu().numpy())
        losses.update(loss.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Epoch: [%d/%d]' % (epoch + 1, all_epochs),
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'Micro-Precision:%.6f' % (MiP),
                'Micro-Recall:%.6f' % (MiR),
                'Micro-F Measure:%.6f' % (MiF),
                'PNum:%.2f' % (PNum),
                'RNum:%.2f' % (RNum)])
            print(res)

    # Return summary statistics
    return batch_time.avg, losses.avg


def eval_epoch(model, loader, print_freq=10, is_test=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    global THREADHOLD
    # Model on eval mode
    model.eval()

    all_trues = []
    all_preds = []
    all_gos = []
    end = time.time()
    for batch_idx, (input, embed_input, interpro_info,target, gos) in enumerate(loader):

        # Create vaiables
        with torch.no_grad():
            if torch.cuda.is_available():
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
        output = model(input_var, embed_input_var,interpro_info_var)

        loss = torch.nn.functional.binary_cross_entropy(output, target_var).cuda()

        # measure accuracy and record loss
        batch_size = target.size(0)
        losses.update(loss.item(), batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Test' if is_test else 'Valid',
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
            ])
            print(res)
        all_trues.append(target.numpy())
        all_preds.append(output.data.cpu().numpy())
        all_gos.extend(gos)

    all_trues = np.concatenate(all_trues, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    print("all_trues:", all_trues.shape)
    print("all_preds:", all_preds.shape)
    print("all_gos:", len(all_gos))
    f_max, p_max, r_max, t_max, predictions_max = compute_performance(all_preds, all_trues, all_gos)
    auc = compute_roc(all_preds, all_trues)
    aupr = compute_aupr(all_preds, all_trues)
    if predictions_max is not None:
        mcc = compute_mcc(predictions_max, all_trues)
        MiP, MiR, MiF, P_NUM, T_NUM = micro_score(predictions_max, all_trues)
    else:
        mcc = 0
        MiP = 0
        MiR = 0
        MiF = 0
        P_NUM = 0
        T_NUM = 0
  
    THREADHOLD = t_max
    # Return summary statistics
    return batch_time.avg, losses.avg, f_max, p_max, r_max, t_max, auc, aupr, mcc


def train(class_tag, interPro_size,model, train_data_set, test_data_set, save, n_epochs=3,
          batch_size=64, lr=0.1, wd=0.0001, momentum=0.9, seed=None, num=1):
    if seed is not None:
        torch.manual_seed(seed)

    # split data
    
    samples_num = configs.train_samples_num[class_tag]
    split_num = int(configs.splite_rate * samples_num)
    print(split_num)
    data_index = np.arange(samples_num)
    np.random.shuffle(data_index)
    train_index = data_index[:split_num]
    eval_index = data_index[split_num:]
    train_samples = sampler.SubsetRandomSampler(train_index)
    eval_samples = sampler.SubsetRandomSampler(eval_index)


    # Data loaders
    train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=batch_size,
                                               sampler=train_samples, pin_memory=(torch.cuda.is_available()),
                                               num_workers=5, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(train_data_set, batch_size=batch_size,
                                               sampler=eval_samples, pin_memory=(torch.cuda.is_available()),
                                               num_workers=5, drop_last=False)
    test_loader = None
    # Model on cuda
    if torch.cuda.is_available():
        model = model.cuda()

    # Wrap model for multi-GPUs, if necessary
    model_wrapper = model

    # Optimizer
    optimizer = torch.optim.Adam(model_wrapper.parameters(), lr=0.001)
    
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[i for i in range(1, n_epochs, 10)],
                                         gamma=0.01)

    # Start log
    with open(os.path.join(save, '{0}_{1}_results_V1_{2}-{3}.csv'.format(class_tag, num,interPro_size[0],interPro_size[1])), 'w') as f:
        f.write('epoch,loss,auc,aupr,mcc,F_max,Precision,Recall,threadhold\n')

        # Train model
        best_mcc = 0
        threadhold = 0
        count = 0
        for epoch in range(n_epochs):
            # scheduler.step()
            _, train_loss = train_epoch(
                model=model_wrapper,
                loader=train_loader,
                optimizer=optimizer,
                epoch=epoch,
                all_epochs=n_epochs,
            )
            _, valid_loss, f_max, p_max, r_max, t_max, auc, aupr, mcc = eval_epoch(
                model=model_wrapper,
                loader=valid_loader,
                is_test=(not valid_loader)
            )
            print(
            'epoch:%03d,loss:%0.5f\nauc:%0.6f,aupr:%0.6f,mcc:%0.6f\nF_max:%.6f,Precision:%.6f,Recall:%.6f,threadhold:%.6f\n' % (
                (epoch + 1), valid_loss,
                auc, aupr, mcc,
                f_max, p_max, r_max, t_max))

            if f_max > best_mcc:
                count = 0
                best_mcc = f_max
                threadhold = t_max
                print("new best f_max:{0}(threadhold:{1})".format(f_max, threadhold))
                torch.save(model.state_dict(), os.path.join(save, '{0}_{1}_model_V1_{2}-{3}.dat'.format(class_tag, num,interPro_size[0],interPro_size[1])))
            else:
                count += 1
                if count>=5:
                    return None
            # # Log results
            f.write('%03d,%0.6f,%0.6f,%0.6f,%0.6f,%.6f,%.6f,%.6f,%.6f\n' % (
                (epoch + 1), valid_loss, aupr,f_max, p_max, r_max, mcc, auc, t_max))

    
    
def demo(class_tag,interPro_size, save=None, train_num = 1, splite_rate = 0.1, efficient=True,
              epochs=20, seed=None,pretrained_result=None):

    train_sequences_file = 'data_cache/train_{0}_all_sequences.pkl'.format(class_tag)
    train_features_file = 'data_cache/train_{0}_sequence_features.pkl'.format(class_tag)
    train_net_embedding_file = 'data_cache/filter_train_{0}.embedding.pkl'.format(class_tag)
    embeddings_file = 'utils/embeddings.npy'

    train_label_file = 'data_cache/train_{0}_label.pkl'.format(class_tag)
    train_gos_file = 'data_cache/train_{0}_gos.pkl'.format(class_tag)
   
    
    #parameters
    batch_size = configs.batch_size

    # Datasets
    train_dataSet = data_generator.dataSet_V2(train_sequences_file, train_label_file, train_gos_file, embeddings_file,
                                             train_net_embedding_file, train_features_file)
    test_dataSet = None
    # Models

    print("train {0}-{1}".format(class_tag,train_num))
    class_nums = configs.class_nums[class_tag]
    model = DeepGOA(class_nums,interPro_size,class_tag)
    model.apply(weight_init)
    print(model)

    # Train the model
    train(class_tag,interPro_size,model=model, train_data_set=train_dataSet, test_data_set=test_dataSet, save=save,
          n_epochs=epochs, batch_size=batch_size, seed=seed,num=train_num)
    print('Done!')


if __name__ == '__main__':
    """
    """
    interPro_size = configs.interPro_size
    path_dir = "./checkpoints/deepGOA"
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    for ii in range(1,11):
        for tag in ["mf",'bp','cc']:
            for size in interPro_size[tag]:
                demo(tag,size,path_dir,ii)
