from torch import nn
from torch.nn.functional import interpolate

from unipose.model.wasp import WASP
from unipose.model.decoder import Decoder
from unipose.model.resnet import ResNet101


class Unipose(nn.Module):
    def __init__(self, output_stride=16, num_classes=21, freeze_bn=False,
                 stride=8):
        super(Unipose, self).__init__()
        self.stride = stride
        self.num_classes = num_classes

        BatchNorm = nn.BatchNorm2d
        self.pool_center = nn.AvgPool2d(kernel_size=9, stride=8, padding=1)

        self.backbone = ResNet101(output_stride, BatchNorm)
        self.wasp = WASP(output_stride, BatchNorm)
        self.decoder = Decoder(self.num_classes, BatchNorm)
        if freeze_bn:
            # If you're fine-tuning to minimize training,
            # it's typically best to keep batch normalization frozen
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.wasp(x)
        x = self.decoder(x, low_level_feat)
        if self.stride != 8:
            x = interpolate(x, size=(input.size()[2:]), mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            # elif isinstance(m, SynchronizedBatchNorm2d):
            #     m.eval()
