from network.deeplabv3p import SingleNetwork, init_weight
from config import default_config
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

import sys
sys.path.append("..")

# Specify the graphics card
# torch.cuda.set_device(7)

def my_net(modelname):

    if modelname == 'imagenet_ResUnet':
        if default_config['Pretrain']:
            print("Using pretrain model")
            model = smp.Unet(
                encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                # use `imagenet` pre-trained weights for encoder initialization
                encoder_weights="imagenet",
                # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                in_channels=1,
                # model output channels (number of classes in your dataset)
                classes=4,
            )
        else:
            model = smp.Unet(
                encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
                # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                in_channels=1,
                # model output channels (number of classes in your dataset)
                classes=4,
            )
    elif modelname == 'ssl_ResUnet':
        if default_config['Pretrain']:
            print("Using pretrain model")
            model = smp.Unet(
                encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                # use `imagenet` pre-trained weights for encoder initialization
                encoder_weights="ssl",
                # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                in_channels=1,
                # model output channels (number of classes in your dataset)
                classes=4,
            )
        else:
            model = smp.Unet(
                encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                # use `imagenet` pre-trained weights for encoder initialization
                encoder_weights=None,
                # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                in_channels=1,
                # model output channels (number of classes in your dataset)
                classes=4,
            )

    elif modelname == 'mydeeplabV3P':
        if default_config['Pretrain']:
            print("Using pretrain backbone")
            model = SingleNetwork(
                num_classes=4,
                criterion=nn.CrossEntropyLoss(),
                pretrained_model=default_config['pretrain_file'],
                norm_layer=nn.BatchNorm2d,
                in_channels=1)
            init_weight(model.business_layer,
                        nn.init.kaiming_normal_,
                        nn.BatchNorm2d,
                        bn_eps=default_config['bn_eps'],
                        bn_momentum=default_config['bn_momentum'],
                        mode='fan_in', nonlinearity='relu')
        else:
            model = SingleNetwork(
                num_classes=4,
                criterion=nn.CrossEntropyLoss(),
                pretrained_model=None,
                norm_layer=nn.BatchNorm2d,
                in_channels=1)
            init_weight(model.business_layer,
                        nn.init.kaiming_normal_,
                        nn.BatchNorm2d,
                        bn_eps=default_config['bn_eps'],
                        bn_momentum=default_config['bn_momentum'],
                        mode='fan_in', nonlinearity='relu')
    else:
        print("model name are wrong")
    return model

if __name__ == "__main__":
    x = torch.randn((2, 1, 288, 288))
    model_r = my_net(modelname='mydeeplabV3P')
    model_l = my_net(modelname='mydeeplabV3P')
    preds_r = model_r(x)
    preds_l = model_l(x)
    preds = preds_r + preds_l
    print(x.shape)
    print(preds_r.shape)
    print(preds_l.shape)
    print(preds.shape)
