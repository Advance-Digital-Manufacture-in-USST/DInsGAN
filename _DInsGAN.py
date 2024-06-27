import torch
import torch.nn as nn


class GanSTN3d(nn.Module):
    def __init__(self):
        super(GanSTN3d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=64, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=(1,))
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=(1,))

        self.fc1 = nn.Linear(1024, 512)  # 用于设置网络中的全连接层的
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 6*6)

        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        pass

    def forward(self, x):  # 输入量：B, 6, N
        batch_size = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))  # B, 64, N
        x = F.relu(self.bn2(self.conv2(x)))  # B, 128, N
        x = F.relu(self.bn3(self.conv3(x)))  # B, 1024, N

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)  # reshape成1024列，但是不确定几行

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        i_den = Variable(torch.from_numpy(np.eye(6).flatten().astype(np.float32))).view(1, 6*6).repeat(batch_size, 1)

        if x.is_cuda:
            i_den = i_den.cuda()
            pass

        x = x + i_den
        x = x.view(-1, 6, 6)  # 输出为Batch * 3 * 3的张量

        return x
    pass


class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(1024, 1024, 1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(1024, 1024, 1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(1024, 128, 1, bias=False),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv1d(1024, 128, 1, bias=False),
                                   nn.BatchNorm1d(128),
                                   nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv1d(1024, 1024, 1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.ReLU())

    def forward(self, x):
        x1 = self.conv1(x)  # B, 1024, N

        v_ = self.conv2(x1)      # B, 1024, N
        q_ = self.conv3(x1)      # B, 128, N
        q_ = q_.transpose(2, 1)  # B, N, 128
        k_ = self.conv4(x1)      # B, 128, N

        sources = torch.bmm(q_, k_)  # B, N, N
        sources = nn.Softmax(dim=-1)(sources)
        sources = sources / (1e-9 + sources.sum(dim=1, keepdims=True))

        sources = torch.bmm(v_, sources)  # B, 1024, N

        x2 = self.conv5(x1 - sources)  # B, 1024, N
        x = x1 + x2                    # B, 1024, N

        return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.stn = GanSTN3d()

        self.conv1 = nn.Sequential(nn.Conv1d(6, 64, 1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 64, 1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(64, 64, 1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv1d(64, 128, 1, bias=False),
                                   nn.BatchNorm1d(128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(128, 1024, 1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        """
        生成器中的编码器模块
        :param x: 大小为(B, 6, N)的矩阵
        :return: 大小为(B, 1024, N)的矩阵 —— 隐向量
        """

        trans_ = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans_)  # 计算两个tensor的矩阵乘法
        x = x.transpose(2, 1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)  # B, 1024, N

        return x


class TrajectoryGenerator(nn.Module):
    def __init__(self):
        super(TrajectoryGenerator, self).__init__()

        self.points_encoder = Encoder()
        self.param_layer = nn.Sequential(nn.Linear(18, 32),
                                         nn.Linear(32, 64),
                                         nn.Linear(64, 128))
        self.attention = SelfAttention()
        self.lstm_features = nn.LSTM(1024 + 128, hidden_size=256, num_layers=3, batch_first=True)
        self.feature_layer = nn.Sequential(nn.Linear(256, 128),
                                           nn.BatchNorm1d(128),
                                           nn.ReLU(True),
                                           nn.Linear(128, 12))

    def forward(self, points, robot_param):
        points = self.points_encoder(points)         # B, 1024, N
        robot_param = self.param_layer(robot_param)  # B, 1, 128

        points = self.attention(points)  # B, 1024, N
        points = torch.max(points, dim=2, keepdim=True)[0]  # B, 1024, 1
        points = points.transpose(2, 1)   # B, 1, 1024

        packed = torch.cat([points, robot_param], dim=2)
        features, _ = self.lstm_features(packed)  # B, 1, 256

        features = torch.squeeze(features)
        outputs = self.feature_layer(features)  # B, 12

        return outputs


if __name__ == '__main__':
    net = TrajectoryGenerator().cuda()

    img = torch.randn((8, 6, 1024)).cuda()
    traj = torch.randn((8, 1, 18)).cuda()

    out = net(img, traj)
