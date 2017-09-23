from __future__ import print_function
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
from warpctc_pytorch import CTCLoss
import os
import utils
import dataset

import time
from util.visualizer import Visualizer
from collections import OrderedDict
import Levenshtein as Lev

import models.crnn as crnn

parser = argparse.ArgumentParser()
parser.add_argument('--trainroot', required=True, help='path to dataset')
parser.add_argument('--valroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imgC', type=int, default=3, help='the channels of the input image to network')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--nepochs', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate for Critic, default=0.01')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--gpu_idx', action='store_true', help='specific which gpu to use')
# parser.add_argument('--crnn', default='', help="path to crnn (to continue training)")
# parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
parser.add_argument('--checkpoints_dir', default='expr', help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')

# for visualize.
parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')        
parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
parser.add_argument('--name', type=str, default='experiment_name', help='name of the checkpoints_dir. It decides where to store samples and models')


# for model weight reload.
parser.add_argument('--model_path', default='', help="path to crnn (to continue training)")
# for model weight initialization.
parser.add_argument('--kaiming', action='store_true', help="whether use kaiming initialization.")


opt = parser.parse_args()
opt.isTrain = True
print(opt)
visualizer = Visualizer(opt)

if opt.checkpoints_dir is None:
    opt.checkpoints_dir = 'expr'
os.system('mkdir {0}'.format(opt.checkpoints_dir))

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

train_dataset = dataset.lmdbDataset(root=opt.trainroot)
assert train_dataset
if not opt.random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler = None
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=True, sampler=sampler,
    num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))
test_dataset = dataset.lmdbDataset(
    root=opt.valroot, transform=dataset.resizeNormalize((100, 32)))

nclass = len(opt.alphabet) + 1

converter = utils.strLabelConverter(opt.alphabet, ignore_case=False)
criterion = CTCLoss()


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__    
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        if opt.kaiming:
            init.kaiming_uniform(m.weight)
        else:
            m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

crnn = crnn.CRNN(opt.imgH, opt.imgC, nclass, opt.nh)
crnn.apply(weights_init)
if opt.model_path != '':
    print('loading pretrained model from %s' % opt.model_path)
    state_dict = torch.load(opt.model_path)
    state_dict_rename = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        state_dict_rename[name] = v
    crnn.load_state_dict(state_dict_rename)
print(crnn)
image = torch.FloatTensor(opt.batchSize, opt.imgC, opt.imgH, opt.imgH)
text = torch.IntTensor(opt.batchSize * 5)
length = torch.IntTensor(opt.batchSize)

if opt.cuda:
    crnn.cuda()
    if opt.gpu_idx:
        crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.gpu_idx, opt.gpu_idx+opt.ngpu) )
    else:
        crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
    image = image.cuda()
    criterion = criterion.cuda()

image = Variable(image)
text = Variable(text)
length = Variable(length)

# loss averager
loss_avg = utils.averager()

# setup optimizer
if opt.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=opt.lr,
                           betas=(opt.beta1, 0.999))
elif opt.adadelta:
    optimizer = optim.Adadelta(crnn.parameters(), lr=opt.lr)
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)


def val(net, dataset, criterion, max_iter=100):
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers))
    val_iter = iter(data_loader)

    epoch_iter = 0
    n_correct = 0
    loss_avg = utils.averager()
    edit_distance = 0
    max_iter = min(max_iter, len(data_loader))
    for epoch_iter in range(max_iter):
        data = val_iter.next()
        epoch_iter += 1
        cpu_images, cpu_texts = data
        # print (cpu_texts)
        # from matplotlib import pyplot as plt
        # import numpy as np
        # for i in range(cpu_images.shape[0]):
        #     tmp = cpu_images[i].numpy()
        #     # tmp = np.squeeze(tmp, axis=0)
        #     tmp = tmp.transpose(1, 2, 0)
        #     plt.imshow(tmp)
        #     plt.show()
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)
        _, preds = preds.max(2)
        # preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        
        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target.lower():
                n_correct += 1
            # add edit distance.
            edit_distance += Lev.distance(pred, target)

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * opt.batchSize)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))
    print ('Total distance: ', edit_distance)
    return accuracy, edit_distance


def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)

    preds = crnn(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    crnn.zero_grad()
    cost.backward()
    optimizer.step()
    return cost



total_step = 0
for epoch in range(opt.nepochs):
    train_iter = iter(train_loader)
    epoch_iter = 0
    iter_start_time = time.time()
    while epoch_iter < len(train_loader):
        for p in crnn.parameters():
            p.requires_grad = True
        crnn.train()

        cost = trainBatch(crnn, criterion, optimizer)
        loss_avg.add(cost)
        epoch_iter += 1
        total_step += 1

        if total_step % opt.displayInterval == 0:
            loss = loss_avg.val()
            # print('[%d/%d][%d/%d][%d] Loss: %f' %
            #       (epoch, opt.nepochs, epoch_iter, len(train_loader), total_step, loss))
            loss_to_plot = OrderedDict([('avg_loss', loss), ('cost', cost.data[0])])
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, total_step, loss_to_plot, t)
            visualizer.plot_current_errors(epoch, float(epoch_iter)/len(train_loader) \
                                         , opt, loss_to_plot)
            loss_avg.reset()

        if total_step % opt.valInterval == 0:
            accu, edit_dist = val(crnn, test_dataset, criterion)
            test_to_plot = OrderedDict([('accuracy', accu), ('edit_distacne', edit_dist)])
            visualizer.plot_test_result(epoch, float(epoch_iter)/len(train_loader) \
                                         , opt, test_to_plot, win=2)            

        # do checkpointing
        if total_step % opt.saveInterval == 0:
            torch.save(
                crnn.state_dict(), 
                '{0}/netCRNN_{1}_{2}.pth'.format(opt.checkpoints_dir, epoch, epoch_iter))