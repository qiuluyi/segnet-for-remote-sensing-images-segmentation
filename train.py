
import numpy as np
from camnet import *
from dataset import *
from test import *
import torch.utils.data as data
import torch.optim.lr_scheduler
import torch.nn.init
from dataset import *
from preprocess import *
import os

try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener

def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch=5):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")
    # net = torch.nn.DataParallel(net, device_ids=[0, 1])
    net.to(device)

    weights = weights.cuda()
    criterion = nn.NLLLoss2d(weight=weights.to(device))
    # criterion = nn.NLLLoss2d(weight=weights.cuda())
    iter_ = 0



    for e in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        net.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            # data, target = Variable(data.cuda()), Variable(target.cuda())
            data, target = Variable(data.to(device)), Variable(target.to(device))
            optimizer.zero_grad()
            # net=nn.DataParallel(net)
            output = net(data)
            loss = CrossEntropy2d(output, target, weights.to(device))
            # loss = CrossEntropy2d(output, target, weights)
            loss.backward()
            optimizer.step()

            # losses[iter_] = loss.data[0]
            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_])

            if iter_ % 100 == 0:

                rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0], (1, 2, 0)), dtype='uint8')
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]

                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(e, epochs, batch_idx,
                                                                                                 len(train_loader),
                                                                                                 100. * batch_idx / len(
                                                                                                     train_loader),
                                                                                                 loss.item(),
                                                                                                 accuracy(pred, gt)))
                #plt.plot(mean_losses[:iter_]) and plt.show()
                # fig = plt.figure()
                # fig.add_subplot(131)
                #plt.imshow(rgb)
                #plt.title('RGB')
                #fig.add_subplot(132)
                #plt.imshow(convert_to_color(gt))
                #plt.title('Ground truth')
               # fig.add_subplot(133)
                #plt.title('Prediction')
                #plt.imshow(convert_to_color(pred))
                #plt.show()
            iter_ += 1

            del (data, target, loss)

        if e % save_epoch == 0:
            #acc,meanF1score = retest(net, test_ids, all=False, stride=min(WINDOW_SIZE))
            # torch.save(net.state_dict(), './checkpoint14/two_stage_epoch{}_{}_{}'.format(e,acc,meanF1score))
            torch.save(net.state_dict(), './checkpoint/two_stage_epoch{}'.format(e))

    torch.save(net.state_dict(), './checkpoint/two_stage_final')

def train_method():
    # net = ISegNet()
    net = SegNet()
    # net = DUNet()
    # net =Unet()
    # 得到预训练模型网络权重
    vgg_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
    if not os.path.isfile('./pretrain/vgg16_bn-6c64b313.pth'):
        print("No pretrain-model!")
        #weights = URLopener().retrieve(vgg_url, './pretrain/resnet50-19c8e357.pth')
    vgg16_weights = torch.load('./pretrain/vgg16_bn-6c64b313.pth')
    mapped_weights = {}
    for k_vgg, k_segnet in zip(vgg16_weights.keys(), net.state_dict().keys()):
        if "features" in k_vgg:
            mapped_weights[k_segnet] = vgg16_weights[k_vgg]
           # print("Mapping {} to {}".format(k_vgg, k_segnet))
    try:
        net.load_state_dict(mapped_weights)
        #print("Loaded VGG-16 weights in SegNet !")
    except:
        pass

    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")
    net.to(device)

    base_lr = 0.01
    params_dict = dict(net.named_parameters())
    params = []
    for key, value in params_dict.items():
        if '_D' in key:
            params += [{'params': [value], 'lr': base_lr}]
        else:
            params += [{'params': [value], 'lr': base_lr / 2}]

    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)
    train(net, optimizer, 20, scheduler)

if __name__=="__main__":
    train_ids = ['1','3', '5', '7', '11', '13', '15', '17', '21', '23', '26', '28', '30', '32', '34', '37']

    test_ids = ['2', '4','6',  '8','10','12', '14', '16','20', '22', '24', '27', '29', '31', '33', '35', '38']
    train_set = ISPRS_dataset(train_ids, cache=CACHE)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE)
    train_method()
