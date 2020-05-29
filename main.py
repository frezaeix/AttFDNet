import sys
sys.path.append('./util')
sys.path.append('./model')
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import argparse
import os
import time
import gc
import tensorflow as tf
from dataloader import salicon
from evaluation import cal_cc_score, cal_sim_score, cal_kld_score, cal_auc_score, cal_nss_score, add_center_bias
from unet import standard_unet
from loss import NSS, CC, KLD, cross_entropy
from sam import SAM


parser = argparse.ArgumentParser(description='Saliency prediction for images')
parser.add_argument('--mode', type=str, default='train', help='Selecting running mode (default: train)')
parser.add_argument('--img_dir', type=str, default='../data/images', help='Directory to the image data')
parser.add_argument('--fix_dir', type=str, default='../data/fixations', help='Directory to the raw fixation file')
parser.add_argument('--anno_dir', type=str, default='../data/maps', help='Directory to the saliency maps')
parser.add_argument('--width', type=int, default=320, help='Width of input data')
parser.add_argument('--height', type=int, default=240, help='Height of input data')
parser.add_argument('--clip', type=float, default=-1, help='Gradient clipping')
parser.add_argument('--batch', type=int, default=10, help='Batch size')
parser.add_argument('--epoch', type=int, default=30, help='Number of epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--lr_decay', type=float, default=0.25, help='Learning rate decay factor')
parser.add_argument('--lr_decay_step', type=int, default=10, help='Learning rate decay step')
parser.add_argument('--checkpoint', type=str, default='../save/', help='Checkpoint path')
parser.add_argument('--resume', type=bool, default=False, help='Resume from checkpoint or not')
parser.add_argument('--center_bias', type=bool, default=True, help='Adding center bias or not')
args = parser.parse_args()

seed = 1
np.random.seed(seed)
torch.manual_seed(seed)

transform = transforms.Compose([
                                transforms.Resize((args.height,args.width)),
                                transforms.ToTensor(), 
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])

def add_summary_value(writer, key, value, iteration): #tensorboard visualization
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, iteration)

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_learning_rate(optimizer, epoch):
    "adatively adjust lr based on iteration"
    if epoch >= 1: #30-adam
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * (args.lr_decay ** (epoch/args.lr_decay_step))

def main():
    # IO
    tf_summary_writer = tf and tf.summary.FileWriter(args.checkpoint)
    train_data = salicon(args.anno_dir,args.fix_dir,args.img_dir,args.width,args.height,'train',transform)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=4)    
    test_data = salicon(args.anno_dir,args.fix_dir,args.img_dir,args.width,args.height,'val',transform)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch, shuffle=False, num_workers=4)

    # model construction
    #model = standard_unet().cuda()
    model = SAM().cuda()

    if args.resume:
        model.load_state_dict(torch.load(os.path.join(args.checkpoint,'model.pth')))

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5, nesterov=True)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3) #1e-8

    def train(iteration):
        model.train()
        avg_loss = 0
        for i, (img,sal_map,fix) in enumerate(trainloader):
            load_t0 = time.time()
            img, sal_map, fix = img.cuda(), sal_map.cuda(), fix.cuda()
            optimizer.zero_grad()
            pred = model(img)
            loss = NSS(pred,fix) + KLD(pred,sal_map) + CC(pred,sal_map)
            # loss = cross_entropy(pred,sal_map)
            loss.backward()
            if args.clip != -1 :
                clip_gradient(optimizer,args.clip) #gradient clipping without normalization
            optimizer.step()

            avg_loss = (avg_loss*np.maximum(0,i) + loss.data.cpu().numpy())/(i+1)
            load_t1 = time.time()
            if i%25 == 0:
                add_summary_value(tf_summary_writer, 'train_loss', avg_loss, iteration)
                tf_summary_writer.flush()
                # show the information
                print('Totel iter ' + repr(iteration) + ' || L: %.4f || ' % (loss.item()) +
                      'Batch time: %.4f sec. || ' % (load_t1 - load_t0) + 'LR: %.8f' % (args.lr))
            iteration += 1
        return iteration

    def test(iteration):
        model.eval()
        nss_score = []
        cc_score = []
        auc_score = []
        sim_score = []
        kld_score = []
        for i, (img,sal_map,fix) in enumerate(testloader):
            img = img.cuda()
            #pred = model(img,softmax=False)
            with torch.no_grad():
                pred = model(img)
            pred = pred.data.cpu().numpy()
            sal_map = sal_map.data.numpy()
            fix = fix.data.numpy()
            # computing score for each data
            for j in range(len(img)):
                cur_pred = pred[j].squeeze()
                cur_pred /= cur_pred.max() # for training with Saliency evaluation metrics
                if args.center_bias:
                    cur_pred = add_center_bias(cur_pred)
                cc_score.append(cal_cc_score(cur_pred,sal_map[j]))
                sim_score.append(cal_sim_score(cur_pred,sal_map[j]))
                kld_score.append(cal_kld_score(cur_pred,sal_map[j]))
                nss_score.append(cal_nss_score(cur_pred,fix[j]))
                auc_score.append(cal_auc_score(cur_pred,fix[j]))
 
        add_summary_value(tf_summary_writer, 'NSS', np.mean(nss_score), iteration)
        add_summary_value(tf_summary_writer, 'CC', np.mean(cc_score), iteration)
        add_summary_value(tf_summary_writer, 'AUC', np.mean(auc_score), iteration)
        add_summary_value(tf_summary_writer, 'SIM', np.mean(sim_score), iteration)
        add_summary_value(tf_summary_writer, 'KLD', np.mean(kld_score), iteration)
        tf_summary_writer.flush()
        return np.mean(cc_score)

    iteration = 0 
    best_score = 0
    for epoch in range(args.epoch):
        adjust_learning_rate(optimizer, epoch+1)
        iteration = train(iteration)
        cur_score = test(iteration)
        # torch.save(model.module.state_dict(),os.path.join(args.checkpoint,'model.pth'))
        torch.save(model.state_dict(),os.path.join(args.checkpoint,'model.pth')) # for single-GPU training
        if cur_score > best_score:
            best_score = cur_score
            # torch.save(model.module.state_dict(),os.path.join(args.checkpoint,'model_best.pth'))
            torch.save(model.state_dict(),os.path.join(args.checkpoint,'model_best.pth')) # for single-GPU training


# evaluation-only
def evaluation():
    pass

if args.mode == 'train':
    main()
else:
    evaluation()












