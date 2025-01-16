import torch
import utils
import argparse
from torch.utils.tensorboard import SummaryWriter
import os
import torchvision
import tqdm
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
# import pytorch_warmup as warmup
from sklearn.metrics import f1_score
import time
import  torchattacks

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config_file',required=True,type=str)
parser.add_argument('--local_rank',type=int,default=-1)
parser.add_argument('--use_ddp',action='store_true',default=False)

def train(config,train_loader,model,logger,step):
    running_dic = None
    count = 0
    total_num = len(train_loader)
    for i, data in tqdm.tqdm(enumerate(train_loader)):
        data['epoch'] = step
        dic = model.optimize_parameters(data)
        count += 1

        if running_dic == None:
            running_dic = {}
            for k, v in dic.items():
                if k != 'train_print_img':
                    running_dic[k] = v
        else:
            for k, v in dic.items():
                if k != 'train_print_img' and k != 'recon_weight':
                    running_dic[k] += v

        if i % config['print_loss'] == 0:
            txt = 'epoch: {},\t step: {},\t'.format(step, i)
            for k in list(dic.keys()):
                if k != 'train_print_img':
                    txt += ',{}: {},\t'.format(k, dic[k])
            print(txt)

        if config['print_img'] != None and i % config['print_img'] == 0 and 'train_print_img' in dic and dic['train_print_img'] != None:
            print_img = dic['train_print_img']
            grid = torchvision.utils.make_grid(print_img,nrow=1)
            logger.add_image('train_img',grid,global_step=total_num * step + i)

    running_dic['train_loss'] /= count
    if 'train_acc1' in running_dic.keys():
        running_dic['train_acc1'] /= count
    if 'train_acc5' in running_dic.keys():
        running_dic['train_acc5'] /= count

    if 'train_acc1_exp' in running_dic.keys():
        running_dic['train_acc1_exp'] /= count
    if 'train_acc5_exp' in running_dic.keys():
        running_dic['train_acc5_exp'] /= count

    if 'train_acc1_pose' in running_dic.keys():
        running_dic['train_acc1_pose'] /= count
    if 'train_acc5_pose' in running_dic.keys():
        running_dic['train_acc5_pose'] /= count

    if 'train_acc1_flip' in running_dic.keys():
        running_dic['train_acc1_flip'] /= count
    if 'train_acc5_flip' in running_dic.keys():
        running_dic['train_acc5_flip'] /= count

    for k, v in running_dic.items():
        logger.add_scalar(k, v, global_step=step)

def eval(config,val_loader,model,logger,step):
    running_dic = None
    count = 0
    total_num = len(val_loader)

    for i, data in tqdm.tqdm(enumerate(val_loader)):
        dic = model.eval(data)
        count += 1

        if running_dic == None:
            running_dic = {}
            for k, v in dic.items():
                if k != 'eval_print_img':
                    running_dic[k] = v
        else:
            for k, v in dic.items():
                if k != 'eval_print_img':
                    running_dic[k] += v

        if i % config['print_loss'] == 0:
            txt = 'epoch: {},\t step: {},\t'.format(step, i)
            for k in list(dic.keys()):
                if k != 'eval_print_img':
                    txt += ',{}: {},\t'.format(k, dic[k])
            print(txt)

        if config['print_img'] != None and i % config['print_img'] == 0:
            print_img = dic['eval_print_img']
            grid = torchvision.utils.make_grid(print_img,nrow=1)
            logger.add_image('test_img',grid,global_step=total_num * step + i)

    running_dic['eval_loss'] /= count
    for k, v in running_dic.items():
        logger.add_scalar(k, v, global_step=step)

    return running_dic['eval_loss']

def cal_acc(pred,label,threadhold=0.5):
    pred_bool = torch.zeros_like(label)
    pred_bool[pred>threadhold] = 1.
    pred_bool[pred<=threadhold] = -1.

    label_bool = torch.zeros_like(label)
    label_bool[label>threadhold] = 1.

    correct_pred = torch.sum(pred_bool==label_bool,dim=0)

    return correct_pred

def cal_acc_ave(pred_list,label_list,total):
    ans = 0.
    for i in range(0,len(label_list)):
        ans += ((pred_list[i]/label_list[i]))
    ans /= len(label_list)
    return ans

def linear_eval(config,train_loader,val_loader,model,logger):
    count = 0
    linear_classifier = torch.nn.Sequential(torch.nn.BatchNorm1d(config['linear_dim']),torch.nn.Linear(in_features=config['linear_dim'],out_features=config['classes_num'])).cuda()
    sigmoid = torch.nn.Sigmoid()
    optimizer = torch.optim.Adam(linear_classifier.parameters(),lr=config['linear_lr'],weight_decay=config['wd'])
    lr_schduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['eval_epochs'])
    criterizer = torch.nn.BCELoss().cuda()

    best_linear_acc = 0.
    for eval_step in range(config['eval_epochs']):
        train_count = torch.zeros(config['classes_num']).cuda()
        train_acc_count = torch.zeros(config['classes_num']).cuda()
        train_loss = 0.
        train_total = 0.
        for i,data in tqdm.tqdm(enumerate(train_loader)):
            img = data['img_normal'].cuda()
            label = data['label'].cuda()
            fea = model.linear_eval(img)
            count += 1
            pred = linear_classifier(fea)
            pred = sigmoid(pred)
            loss = criterizer(pred,label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_total += label.size(0)
            train_acc_count += cal_acc(pred,label,threadhold=config['eval_threads'])
            train_count += torch.sum(label,dim=0)
            train_loss += loss.item()

        train_acc = train_acc_count / train_count
        train_ave_acc = cal_acc_ave(train_acc_count.cpu().tolist(),train_count.cpu().tolist(),train_total)
        train_loss = train_loss
        lr_schduler.step()

        test_loss = 0.
        test_total = 0.
        test_label = []
        test_pred = []
        linear_classifier.eval()
        # attack = torchattacks.PGD(model.model, eps=8/255, alpha=2/255, steps=10, random_start=True)
        # attack = torchattacks.UPGD(model.model, eps=8/255, alpha=2/255, steps=10, random_start=False)
        # attack = torchattacks.BIM(model.model, eps=8/255, alpha=2/255, steps=10)
        # attack = torchattacks.MIFGSM(model.model, eps=8/255, steps=10, decay=1.0)
        # attack = torchattacks.EOTPGD(model.model, eps=8/255, alpha=2/255, steps=10, eot_iter=2)
        # attack = torchattacks.DIFGSM(model.model, eps=8/255, alpha=2/255, steps=10, decay=0.0, resize_rate=0.9, diversity_prob=0.5, random_start=False)
        # attack = torchattacks.NIFGSM(model.model, eps=8/255, alpha=2/255, steps=10, decay=1.0)
        attack = torchattacks.SINIFGSM(model.model, eps=8/255, alpha=2/255, steps=10, decay=1.0, m=5)
        for i,data in tqdm.tqdm(enumerate(val_loader)):
            img = data['img_normal'].cuda()
            label = data['label'].cuda()
            adv_img = attack(img, label)
            fea = model.linear_eval(adv_img)

            with torch.no_grad():
                pred = linear_classifier(fea)
                pred = sigmoid(pred)
                pred_bool = torch.zeros_like(pred)
                pred_bool[pred>0.5] = 1.

            loss = criterizer(pred,label)
            test_label.extend(label.cpu().tolist())
            test_pred.extend(pred_bool.cpu().tolist())
            test_loss += loss.item()
            test_total += label.size(0)
        from sklearn.metrics import recall_score
        from sklearn.metrics import precision_score
        test_recall = recall_score(test_label,test_pred,average=None)
        test_precision = precision_score(test_label,test_pred,average=None)
        test_f1 = np.mean(f1_score(test_label,test_pred,average=None))
        print(test_recall)
        print(test_precision)
        print(f1_score(test_label,test_pred,average=None))
        test_linear_loss = test_loss

        train_linear_acc_list = train_acc.tolist()

        txt = 'eval step: {},\n'.format(eval_step)
        for tj in range(0, len(train_linear_acc_list)):
            txt += 'linear train acc au_{}: {}\n'.format(tj, train_linear_acc_list[tj])
        txt += 'linear train loss: {},\t linear train ave acc: {}\n'.format(train_loss, train_ave_acc)
        txt += 'linear eval f1: {},\t'.format(test_f1)
        txt += 'linear eval loss: {}\n'.format(test_linear_loss)
        print(txt)

        for tj in range(0,len(train_linear_acc_list)):
            logger.add_scalar('linear_train_acc_au_{}'.format(tj),train_linear_acc_list[tj],eval_step)
        logger.add_scalar('linear_train_loss',train_loss,eval_step)
        logger.add_scalar('linear_train_ave_acc',train_ave_acc,eval_step)
        logger.add_scalar('linear_eval_loss', test_linear_loss, eval_step)
        logger.add_scalar('linear_eval_f1',test_f1,eval_step)

        if best_linear_acc < test_f1:
            best_linear_acc = test_f1
            model_state = model.model.state_dict()
            linear_state = linear_classifier.state_dict()
            for k in list(linear_state.keys()):
                model_state[k] = linear_state[k]
            utils.save_checkpoint({
                'epoch': eval_step + 1,
                'state_dict': model_state,
        }, config)

    return best_linear_acc

def main(config,logger):
    model = utils.create_model(config)
   
    if config['eval']:
        test_dataset = utils.create_dataset(config,'test')
        train_dataset = utils.create_dataset(config,'train')

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            num_workers=config['num_threads'],
            pin_memory=True,
            drop_last=False,
            shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            num_workers=config['num_threads'],
            pin_memory=True,
            drop_last=False,
            shuffle=False
        )
        linear_acc = linear_eval(config,train_loader,test_loader,model,logger)
        print('test linear acc is : {}'.format(linear_acc))
        exit(0)

if __name__ == '__main__':
    opt = parser.parse_args()

    config = utils.read_config(opt.config_file)
    utils.init(config,opt.local_rank,opt.use_ddp)
    logger = SummaryWriter(log_dir=os.path.join(config['log_path'], config['experiment_name']),
                           comment=config['experiment_name'])

    main(config, logger)

    logger.close()