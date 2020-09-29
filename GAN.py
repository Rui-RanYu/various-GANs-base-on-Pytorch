import argparse  # argsparse是python的命令行解析的标准模块，直接在命令行中就可以向程序中传入参数并让程序运行
import os
import numpy as np

# 用于data augmentation
import torchvision.transforms as transforms

# 保存生成图像
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets

# Varibale包含三个属性：
# data：存储了Tensor，是本体的数据
# grad：保存了data的梯度，本是个Variable而非Tensor，与data形状一致
# grad_fn：指向Function对象，用于反向传播的梯度计算之用
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

# 如果根目录下不存在images文件夹，则创建images存放生成图像结果
os.makedirs("images", exist_ok=True)

# 创建解析对象
parser = argparse.ArgumentParser()

# epoch = 200，批大小 = 64，学习率 = 0.0002，衰减率 = 0.5/0.999，线程数 = 4，隐码维数 = 100，样本尺寸 = 28 * 28，通道数 = 1，样本间隔 = 400
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")  # 训练的代数
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")         # 一次训练所选取的样本数
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")              # 学习率
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")  # 用到的cpu数量

parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")   # 一开始生成器输入的是100维服从高斯分布的向量，称为潜在空间的维度数
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")  # 每张图片的尺寸
parser.add_argument("--channels", type=int, default=1, help="number of image channels")       # 每张图片的通道数
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")  # 样本图像之间的间隔

# 定义使用的GPU卡号
# parser.add_argument("--gpu_device", choices=["1", "2", "3"], default="3", help="gpu device number")
# os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_device

# 解析参数
opt = parser.parse_args()
print(opt)

#  定义输入图片shape大小，初始为（1，28，28），即单通道28×28的图片 （通道数，图片维度数，图片维度数）
img_shape = (opt.channels, opt.img_size, opt.img_size)



# 使用cuda的条件
cuda = True if torch.cuda.is_available() else False

# 模型构建两个步骤：①构建子模块；②拼接子模块
class Generator(nn.Module):
    # 构建子模块在init()函数中实现
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            # 这里简单的只对输入数据做线性转换，nn.Linear（）用于设置网络的全连接层，
            # 而全连接层的输入和输出都是二维张量，一般形状为[batch_size, size]
            layers = [nn.Linear(in_feat, out_feat)]  #

            # 使用BN，对小批量的二维输入（三维也可以）进行批标准化
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))

            # 添加LeakyReLU非线性激活层
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # 创建生成器网络模型
        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),  # 此时对于该隐藏层（全连接层），输入是1024，输出是img_shape元素相乘的结果。
                                                       # numpy.prod（img_shape）= img_shape内各个元素相乘之积，之后强制转换为int
            nn.Tanh()  # 经过Tanh激活函数是希望生成的假的图片数据分布能够在-1～1之间
        )

    # 拼接子模块在forward()函数中实现
    # 前向
    def forward(self, z):
        # 生成假样本
        img = self.model(z)  # 往生成器模型中输入噪声z，并赋值给img（假样本）
        img = img.view(img.size(0), *img_shape)  # 利用Sequential.view()，重新调整 tensor 的形状（但总元素不变）
                                                 # 关于view()、size()可参考笔记文档
        # 返回生成图像
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),  # 输入对应了生成器生成的图像（fake），输出为512
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            # 因需判别真假，这里使用Sigmoid函数给出标量的判别结果
            nn.Sigmoid(),
        )

    # 判别
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)  # 此时右式 = img.view(opt.channels, -1), 将值赋给左式.
                                              # 此时左式 img_flat = (opt.channels, opt.img_size * opt.img_size)
        validity = self.model(img_flat)
        # 判别结果
        return validity


# Loss function 损失函数：二分类交叉熵函数
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator  优化器，G和D都使用Adam
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader 加载数据集
os.makedirs("dataset", exist_ok=True)

# ------------------------------------------
#      torch.utils.data.DataLoader
# ------------------------------------------
# 数据加载器，结合了数据集和取样器，并且可以提供多个线程处理数据集。在训练模型时使用到此函数，用来把训练数据分成多个小组，此函数每次抛出一组数据。直至把所有的数据都抛出。就是做一个数据的初始化
#
# torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
#                            batch_sampler=None, num_workers=0, collate_fn=<function default_collate>,
#                            pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None)
# dataset:加载数据的数据集
# batch_size:每批次加载的数据量
# shuffle：默认false，若为True，表示在每个epoch打乱数据
# sampler：定义从数据集中绘制示例的策略,如果指定，shuffle必须为False

dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "dataset",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  训练模型
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # 输入
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  训练G
        # -----------------

        optimizer_G.zero_grad()

        # 采样随机噪声向量
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # 训练得到一批次生成样本
        gen_imgs = generator(z)

        # 计算G的损失函数值
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        #  更新G
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  训练D
        # ---------------------

        optimizer_D.zero_grad()

        # 评估D的判别能力
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        # 更新D
        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )



        # 保存结果
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

        # 保存模型
        torch.save(generator, './model/G_model.pth')
        torch.save(discriminator, './model/D_model.pth')


