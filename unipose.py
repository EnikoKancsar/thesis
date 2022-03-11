from torch import rand
from torch import nn
from torch.nn.functional import interpolate
from model.modules.wasp import build_wasp
from model.modules.decoder import build_decoder
from model.modules.backbone import build_backbone

class Unipose(nn.Module):
    def __init__(self, dataset, backbone='resnet', output_stride=16,
                 num_classes=21, freeze_bn=False, stride=8):
        super(Unipose, self).__init__()
        self.stride = stride
        self.num_classes = num_classes

        BatchNorm = nn.BatchNorm2d
        self.pool_center = nn.AvgPool2d(kernel_size=9, stride=8, padding=1)

        self.backbone    = build_backbone(backbone, output_stride, BatchNorm)
        self.wasp        = build_wasp(backbone, output_stride, BatchNorm)
        self.decoder     = build_decoder(dataset, num_classes, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.wasp(x)
        x = self.decoder(x, low_level_feat)
        if self.stride != 8:
            x = interpolate(x, size=(input.size()[2:]), mode='bilinear', align_corners=True)

        # If you are extracting bouding boxes as well
#         return x[:,0:self.num_classes+1,:,:], x[:,self.num_classes+1:,:,:] 
    
        # If you are only extracting keypoints
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    model = waspnet(backbone='resnet', output_stride=16)
    model.eval()
    input = rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())

