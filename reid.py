import torch
import torch.nn as nn
import numpy as np
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F
import os
import net.extractors as extractors
from torch.nn import functional as F
from net.resnet_ibn import resnet50_ibn_a , resnet50_ibn_b

#kaiming normal
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)

#with train reID ,you should load the HPA dict
def load_network(network,name,epochnum):
    path = os.path.abspath(os.path.join(os.getcwd()))
    save_path = os.path.join(path,'checkpoints',name,'PSPNet_%s'%epochnum)
    print(save_path)
    network.load_state_dict(torch.load(save_path,map_location='cuda:0'))
    return network

#using ResNet50 for HPA
class ResNET50(nn.Module):
    def __init__(self,pretrained=True,usinglargefeature=False):
        super(ResNET50, self).__init__()
        model_resnet = models.resnet50(pretrained=pretrained)
        #ResNEt-50
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        if usinglargefeature:
            self.layer4[0].downsample[0].stride = (1,1)
            self.layer4[0].conv2.stride = (1,1)
            self.layer3[0].downsample[0].stride = (1,1)
            self.layer3[0].conv2.stride = (1, 1)

    def forward(self, x):
        x = x.view(-1, *x.size()[-3:])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        #stage 1-4
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x4,x3
#PPM module
class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=False) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)

# PSPupsample
class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.interpolate(input=x, size=(h, w), mode='bilinear', align_corners=False)
        return self.conv(p)

#HPANet
class HPANet(nn.Module):
    def __init__(self, n_classes=20, sizes=(1, 2, 3, 6),modelname = 'densenet',pretrained=True,usinglargefeature=True):
        super(HPANet, self).__init__()
        if modelname == 'resnet50':
            self.feats = ResNET50(pretrained=pretrained,usinglargefeature=usinglargefeature)
            psp_size = 2048
            deep_features_size = 1024
        if modelname == 'densenet':
            self.feats = getattr(extractors, modelname)(pretrained)
            psp_size = 1024
            deep_features_size = 512
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)
        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)
        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.LogSoftmax(dim=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(deep_features_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        f, class_f = self.feats(x)
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)

        auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))

        return self.final(p), self.classifier(auxiliary)

# the classifier-basic model for ReID
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        # add_block = []
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        add_block1 = []
        add_block2 = []
        add_block1 += [nn.BatchNorm1d(input_dim*2)]
        if relu:
            add_block1 += [nn.LeakyReLU(0.1)]
        add_block1 += [nn.Linear(input_dim*2, num_bottleneck, bias=False)]
        add_block2 += [nn.BatchNorm1d(num_bottleneck)]
        add_block1 = nn.Sequential(*add_block1)
        add_block1.apply(weights_init_kaiming)
        add_block2 = nn.Sequential(*add_block2)
        add_block2.apply(weights_init_kaiming)
        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num, bias=False)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block1 = add_block1
        self.add_block2 = add_block2
        self.classifier = classifier

    def forward(self, x):
        xmax = self.maxpool(x)
        xavg = self.avgpool(x)
        x = torch.cat((xmax, xavg), dim=1)
        # x = self.conv1x1(x)
        x = x.view(x.size(0),-1)
        x = self.add_block1(x)
        x1 = self.add_block2(x)
        x2 = self.classifier(x1)
        return x2, x1, x1

# class ClassBlock(nn.Module):
#     def __init__(self, input_dim, class_num, relu=True,):
#         super(ClassBlock, self).__init__()
#         self.conv1x11 = conv1x1(input_dim, 512)
#         self.conv1x12 = conv1x1(input_dim,1024)
#         self.batchnorm = nn.BatchNorm2d(2048)
#         add_block1 = []
#         add_block2 = []
#         add_block1 += [nn.BatchNorm1d(512)]
#         if relu:
#             add_block1 += [nn.LeakyReLU(0.1)]
#
#         add_block2 += [nn.BatchNorm1d(1024)]
#         if relu:
#             add_block1 += [nn.LeakyReLU(0.1)]
#
#         add_block1 = nn.Sequential(*add_block1)
#         add_block1.apply(weights_init_kaiming)
#         add_block2 = nn.Sequential(*add_block2)
#         add_block2.apply(weights_init_kaiming)
#
#         classifier1 = []
#         classifier1 += [nn.Linear(512, class_num, bias=False)]
#
#         classifier2 = []
#         classifier2 += [nn.Linear(1024, class_num, bias=False)]
#
#         classifier1 = nn.Sequential(*classifier1)
#         classifier1.apply(weights_init_classifier)
#         classifier2 = nn.Sequential(*classifier2)
#         classifier2.apply(weights_init_classifier)
#
#         self.add_block1 = add_block1
#         self.add_block2 = add_block2
#         self.classifier1 = classifier1
#         self.classifier2 = classifier2
#
#     def forward(self, x):
#
#         #branch-1
#         xavg = F.adaptive_avg_pool2d(x,1)
#         xavg = self.conv1x11(xavg)
#         xavg = xavg.view(xavg.size(0),-1)
#         xavg = self.add_block1(xavg)
#         xavgclass = self.classifier1(xavg)
#
#         #branch-2
#         xbatch = self.batchnorm(x)
#         xmax = F.adaptive_max_pool2d(xbatch,1)
#         xmax = self.conv1x12(xmax)
#         xmax = xmax.view(xmax.size(0), -1)
#         xmax = self.add_block2(xmax)
#         xmaxclass = self.classifier2(xmax)
#
#         return xavgclass,xmaxclass,xavg,xmax

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class AM(nn.Module):
    expansion = 1

    def __init__(self, planes):
        super(AM, self).__init__()

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):

        out = self.ca(x) * x
        out = self.sa(out) * out

        return out

#the conv1*1 for dimensionality reduction
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

#compute parsing feature
class HPG(nn.Module):

    def __init__(self, in_channels):
        super(HPG, self).__init__()
        self.AM = AM(planes=in_channels)

    def forward(self, feats, back, head, upbody, lowbody, shoes):
        batch_size, channel, h, w = feats.size(0), feats.size(1), feats.size(2), feats.size(3)
        backnew = F.interpolate(input=back, size=((h, w)), mode='nearest')
        headnew = F.interpolate(input=head, size=((h, w)), mode='nearest')
        upbodynew = F.interpolate(input=upbody, size=((h, w)), mode='nearest')
        lowbodynew= F.interpolate(input=lowbody, size=((h, w)), mode='nearest')
        shoesnew = F.interpolate(input=shoes, size=((h, w)), mode='nearest')
        backfeature = self.AM(backnew*feats)
        headfeature = headnew * feats
        upbodyfeature = upbodynew * feats
        lowbodyfeature= lowbodynew * feats
        shoesfeature = shoesnew *feats

        bottle = backfeature + headfeature +upbodyfeature + lowbodyfeature + shoesfeature
        return bottle,headfeature,upbodyfeature,lowbodyfeature,shoesfeature

#the reid processing
class ReIDNet(nn.Module):
    def __init__(self,class_num,testing=False,usinghpa=True,hpamodel='densenet',hpaepoch='last',featuresize=48):
        super(ReIDNet, self).__init__()
        #some agrs
        self.hpa = usinghpa
        #human parsing attention model
        if self.hpa:
            self.hpamodel = HPANet(n_classes=20, sizes=(1, 2, 3, 6),modelname = hpamodel,pretrained=True,usinglargefeature=True)
            self.hpamodel = load_network(self.hpamodel,name=hpamodel,epochnum=hpaepoch)
            self.hpamodel = self.hpamodel.eval()
        #ResNEt-50
        # model_resnet = models.resnet50(pretrained=True)
        attnet = resnet50_ibn_a(pretrained=True)

        # some agrs
        self.test = testing
        self.featuresize = featuresize

        # processing
        self.conv1 = attnet.conv1
        self.bn1 = attnet.bn1
        self.relu = attnet.relu
        self.maxpool = attnet.maxpool
        self.layer1 = attnet.layer1

        # branch-1 attention reid
        self.layer2reid = attnet.layer2
        self.layer3reid = attnet.layer3
        self.layer4reid = attnet.layer4

        # branch-2 human parsing attention
        self.layer2hpa = attnet.layer2
        self.layer3hpa = attnet.layer3
        self.layer4hpa = attnet.layer4

        if self.featuresize >= 24:
            self.layer4reid[0].downsample[0].stride = (1, 1)
            self.layer4reid[0].conv2.stride = (1, 1)
            self.layer4hpa[0].downsample[0].stride = (1, 1)
            self.layer4hpa[0].conv2.stride = (1, 1)

        if self.featuresize >= 48:
            self.layer3reid[0].downsample[0].stride = (1, 1)
            self.layer3reid[0].conv2.stride = (1, 1)
            self.layer3hpa[0].downsample[0].stride = (1, 1)
            self.layer3hpa[0].conv2.stride = (1, 1)

        if self.featuresize >= 96:
            self.layer2reid[0].downsample[0].stride = (1, 1)
            self.layer2reid[0].conv2.stride = (1, 1)
            self.layer2hpa[0].downsample[0].stride = (1, 1)
            self.layer2hpa[0].conv2.stride = (1, 1)


        self.hpmlayer1 = HPG(256)
        self.hpmlayer2 = HPG(512)
        self.hpmlayer3 = HPG(1024)
        self.hpmlayer4 = HPG(2048)

        self.AM1 = AM(256)
        self.AM2 = AM(512)
        self.AM3 = AM(1024)
        self.AM4 = AM(2048)

        #classfier blocks
        #reid 7
        self.classifierreid1 = ClassBlock(2048, class_num)
        self.classifierreid2 = ClassBlock(2048, class_num)
        self.classifierreid3 = ClassBlock(2048, class_num)
        self.classifierreid4 = ClassBlock(2048, class_num)
        self.classifierreid5 = ClassBlock(2048, class_num)
        self.classifierreid6 = ClassBlock(2048, class_num)
        self.classifierreidglobal = ClassBlock(2048, class_num)

        #human parsing
        self.classifierhp1 = ClassBlock(2048, class_num)
        self.classifierhp2 = ClassBlock(2048, class_num)
        self.classifierhp3 = ClassBlock(2048, class_num)
        self.classifierhp4 = ClassBlock(2048, class_num)
        self.classifierhpglobal = ClassBlock(2048, class_num)


    def forward(self, x ):
        x = x.view(-1, *x.size()[-3:])

        #extract human parsing attention
        with torch.no_grad():
            if self.hpa:
                hpafeat,hpaclass1=self.hpamodel(x)

        #you should plus the parsing by channel,the feature size is 384*128
        hpaattention =  hpafeat

        #background, head, upbody, lowbody, shoes
        hpaf= hpaattention
        back = hpaf[:,0,:,:]
        head = hpaf[:,1,:,:]+hpaf[:,2,:,:]+hpaf[:,4,:,:]+hpaf[:,13,:,:]
        upbody = hpaf[:, 5, :, :]+hpaf[:,6,:,:]+hpaf[:,7,:,:]+hpaf[:,11,:,:]+\
                 hpaf[:,10,:,:]+hpaf[:,3,:,:]+hpaf[:,14,:,:]+hpaf[:,15,:,:]
        lowbody = hpaf[:,9,:,:]+hpaf[:,16,:,:]+hpaf[:,17,:,:]+hpaf[:,12,:,:]
        shoes = hpaf[:,18,:,:]+hpaf[:,19,:,:]+hpaf[:,8,:,:]

        #you should resize the attention feature from [1,1,1] to [1,1,1,1]
        back = torch.unsqueeze(back, 1)
        head = torch.unsqueeze(head,1)
        upbody = torch.unsqueeze(upbody,1)
        lowbody = torch.unsqueeze(lowbody,1)
        shoes = torch.unsqueeze(shoes,1)

        #pre-processing
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        #stage-1
        x1 = self.layer1(x)

        # branch-1 re-id attention
        # x1reid = self.AM1(x1)
        x2reid = self.layer2reid(x1)
        # x2reid = self.AM2(x2reid)
        x3reid = self.layer3reid(x2reid)
        # x3reid = self.AM3(x3reid)
        x4reid = self.layer4reid(x3reid)
        # x4reid = self.AM4(x4reid)

        # PCB
        i  = int(self.featuresize//6)
        p1 = x4reid[:, :, 0  :i*1, :]
        p2 = x4reid[:, :, i*1:i*2, :]
        p3 = x4reid[:, :, i*2:i*3, :]
        p4 = x4reid[:, :, i*3:i*4, :]
        p5 = x4reid[:, :, i*4:i*5, :]
        p6 = x4reid[:, :, i*5:i*6, :]

        # branch-2 human parsing

        x1hpm1, _, _, _, _ = self.hpmlayer1(x1, back, head, upbody, lowbody, shoes)
        x2hpm = self.layer2hpa(x1hpm1)
        x2hpm1, _, _, _, _ = self.hpmlayer2(x2hpm, back, head, upbody, lowbody, shoes)
        x3hpm = self.layer3hpa(x2hpm1)
        x3hpm1, _, _, _, _ = self.hpmlayer3(x3hpm, back, head, upbody, lowbody, shoes)
        x4hpm = self.layer4hpa(x3hpm1)
        x4hpm1, head4, up4, low4, shoes4 = self.hpmlayer4(x4hpm, back, head, upbody, lowbody, shoes)


        #reid branch classifier
        pyidp1, pytrp1, testp1 = self.classifierreid1(p1)
        pyidp2, pytrp2, testp2 = self.classifierreid2(p2)
        pyidp3, pytrp3, testp3 = self.classifierreid3(p3)
        pyidp4, pytrp4, testp4 = self.classifierreid4(p4)
        pyidp5, pytrp5, testp5 = self.classifierreid5(p5)
        pyidp6, pytrp6, testp6 = self.classifierreid6(p6)
        pyidglobal, pytrreidglobal, testreidglobal = self.classifierreidglobal(x4reid)

        #human parsing branch classifier
        pyhpidp1, pyhptrp1, testhpp1 = self.classifierhp1(head4)
        pyhpidp2, pyhptrp2, testhpp2 = self.classifierhp1(up4)
        pyhpidp3, pyhptrp3, testhpp3 = self.classifierhp1(low4)
        pyhpidp4, pyhptrp4, testhpp4 = self.classifierhp1(shoes4)
        pyhpidp5, pyhptrp5, testhpp5 = self.classifierhp1(x4hpm1)

        if self.test == True:
            return testp1,testp2,testp3,testp4,testp5,testp6\
                ,testhpp1,testhpp2,testhpp3,testhpp4,testhpp5,testreidglobal
        else:
            return pyidp1, pyidp2, pyidp3, pyidp4, pyidp5, pyidp6, pyidglobal\
                ,pyhpidp1,pyhpidp2,pyhpidp3,pyhpidp4,pyhpidp5\
                ,pytrp1,pytrp2,pytrp3,pytrp4,pytrp5,pytrp6,pytrreidglobal\
                ,pyhptrp1,pyhptrp2,pyhptrp3,pyhptrp4,pyhptrp5


# net = ReIDNet(class_num=751,testing=True,usinghpa=True,hpamodel='densenet',hpaepoch='last',featuresize=12)
# print(net)
# input = Variable(torch.FloatTensor(4 ,3, 384, 128))
# output = net(input,epoch=20)
# for outputs in output:
#     print(outputs.shape)
