import numpy as np
import random
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch.utils.data
import torch.optim as optim
import torch.nn as nn

from A_dataset import InsGANDataset
from _DInsGAN import TrajectoryGenerator

# 设置随机数
opt_manualSeed = 3047  # 114514
random.seed(opt_manualSeed)
torch.manual_seed(opt_manualSeed)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

if __name__ == '__main__':
    opt = {'batchSize': 32, 'workers': 0, 'n_epoch': 500, 'out_f': 'InsGAN', 'k': 20, 'nPoints': 1024, 'numRun': 1,
           'dataset': r'dataset/dataset_robot_first', 'optimizer': 'SGD', 'scheduler': 'cos', 'gLr': 1e-4, 'dLr': 5e-4,
           'out_fig': 'InsGANFig', 'out_loss': 'InsGANLoss'}

    robotParam = torch.tensor([0, -0.425, -0.392, 0, 0, 0,
                               0.089, 0, 0, 0.109, 0.095, 0.082,
                               np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0])
    robotParam = robotParam.view(1, 18)
    robotParam = robotParam.repeat(opt['batchSize'], 1, 1).cuda()

    try:
        os.makedirs(opt['out_f'])
    except OSError:
        pass

    try:
        os.makedirs(opt['out_fig'])
    except OSError:
        pass

    try:
        os.makedirs(opt['out_loss'])
    except OSError:
        pass

    trainDataset = InsGANDataset(root=opt['dataset'], n_points=opt['nPoints'], purpose='train')
    trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=opt['batchSize'], drop_last=True,
                                                  shuffle=True, num_workers=int(opt['workers']))

    testDataset = InsGANDataset(root=opt['dataset'], n_points=opt['nPoints'], purpose='test')
    testDataLoader = torch.utils.data.DataLoader(testDataset, batch_size=opt['batchSize'], drop_last=True,
                                                 shuffle=True, num_workers=opt['workers'])

    generateNet = TrajectoryGenerator().cuda()

    generateNet.train()

    optimizer_g = optim.Adam(generateNet.parameters(), lr=opt['gLr'])
    criterion = nn.MSELoss()

    G_loss_ = []
    count = len(trainDataLoader)
    for epoch in range(opt['n_epoch']):
        G_epoch_loss = 0
        for i, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader)):
            points, target = data
            points, target = points.float(), target.float()
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()

            optimizer_g.zero_grad()
            outputs = generateNet(points, robotParam)

            loss_ = criterion(outputs, target)
            loss_.backward()

            optimizer_g.step()
            with torch.no_grad():
                G_epoch_loss += loss_.item()

        if epoch == opt['n_epoch']-1:
            torch.save(generateNet.state_dict(), '{}/D_InsGAN_{}.pth'.format(opt['out_f'], opt['numRun']))

        G_epoch_loss /= count
        print(epoch, '\t GLoss:', G_epoch_loss)
        G_loss_.append(G_epoch_loss)

    np.save('{}/D_G_LOSS_{}.npy'.format(opt['out_loss'], opt['numRun']), G_loss_)

    plt.plot(G_loss_, 'b', label='LOSS_CURVE')
    plt.legend()
    plt.title('D_GEN_DIS_LOSS_{}'.format(opt['numRun']))

    plt.savefig('{}/D_GEN_DIS_LOSS_FIG_{}'.format(opt['out_fig'], opt['numRun']))

    plt.show()
