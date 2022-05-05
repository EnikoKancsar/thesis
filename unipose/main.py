# -*-coding:UTF-8-*-
import argparse
import configparser
import sys

import cv2  # image analysis
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn.functional import interpolate
import torch.optim
from tqdm import tqdm  # Progress Bar Creator

sys.path.append("..")
from thesis.unipose.evaluate import accuracy
from thesis.unipose.model.unipose import Unipose
from thesis.unipose.utils import adjust_learning_rate
from thesis.unipose.utils import draw_paint
from thesis.unipose.utils import getDataloader
from thesis.unipose.utils import get_model_summary
from thesis.unipose.utils import get_kpts
from thesis.unipose.utils import plotting
from thesis.unipose.utils import printAccuracies


CONF = configparser.ConfigParser()
CONF.read('../thesis/conf.ini')


class Trainer(object):
    def __init__(self, args):
        self.args         = args
        self.dataset      = args.dataset
        # CONF.set('HYPERPARAMETERS', 'DATASET', self.dataset)

        with open('./conf.ini', 'w') as conf_file:
            CONF.write(conf_file)

        self.workers       = CONF.getint('HYPERPARAMETERS', 'WORKERS')
        self.weight_decay  = CONF.getfloat('HYPERPARAMETERS', 'WEIGHT_DECAY')
        self.momentum      = CONF.getfloat('HYPERPARAMETERS', 'MOMENTUM')
        self.batch_size    = CONF.getint('HYPERPARAMETERS', 'BATCH_SIZE')
        self.learning_rate = CONF.getfloat('HYPERPARAMETERS', 'LEARNING_RATE')
        self.gamma         = CONF.getfloat('HYPERPARAMETERS', 'GAMMA')
        self.step_size     = CONF.getfloat('HYPERPARAMETERS', 'STEP_SIZE')
        # for sigma-tuning of gaussian kernel
        self.sigma         = CONF.getint('HYPERPARAMETERS', 'SIGMA')
        self.stride        = CONF.getint('HYPERPARAMETERS', 'STRIDE')

        cudnn.benchmark   = True  # good when input sizes do not vary

        try:
            self.numClasses = CONF.getint(self.dataset, "NUMBER_OF_CLASSES")
        except configparser.NoSectionError:
            print("Section not found: ", self.dataset)
            if self.dataset in ['MPII']:
                print("Edit your conf.ini file.")
            else:
                print("Invalid dataset choice.")
            sys.exit()

        self.train_loader, self.val_loader, self.test_loader = getDataloader(
            self.dataset, self.sigma, self.stride, self.workers,
            self.batch_size)

        model = Unipose(num_classes=self.numClasses, output_stride=16,
                        freeze_bn=False, stride=self.stride)

        self.model      = model.cuda()

        self.criterion  = nn.MSELoss().cuda()

        self.optimizer  = torch.optim.Adam(self.model.parameters(),
                                           lr=self.learning_rate)

        self.best_model = 12345678.9

        self.iters      = 0

        if self.args.pretrained is True:
            path_to_weights = CONF.get('PRETRAINED', 'PATH_TO_WEIGHTS')
            checkpoint = torch.load(path_to_weights)

            state_dict = self.model.state_dict()
            model_dict = {}

            for k,v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v

            state_dict.update(model_dict)
            self.model.load_state_dict(state_dict)

        self.isBest   = 0
        self.bestPCK  = 0
        self.bestPCKh = 0

        # Print model summary and metrics
        dump_input = torch.rand([1, 3, 368, 368]).cuda()

        with open('./output.txt', 'w') as output_file:
            output_file.write(get_model_summary(self.model, dump_input))

        print(get_model_summary(self.model, dump_input))

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        print("Epoch " + str(epoch) + ':')
        tbar = tqdm(self.train_loader)

        # unipose.data.mpii.MPII.__getitem__ returns these
        for iteration_count, (input, heatmap) in enumerate(tbar):
            learning_rate = adjust_learning_rate(
                self.optimizer, self.iters, self.learning_rate, policy='step',
                gamma=self.gamma, step_size=self.step_size)

            input_on_cuda = input.cuda()
            heatmap_on_cuda = heatmap.cuda()

            self.optimizer.zero_grad()

            heat = self.model(input_on_cuda)

            loss_heat = self.criterion(heat, heatmap_on_cuda)

            loss = loss_heat

            train_loss += loss_heat.item()

            loss.backward()
            self.optimizer.step()

            tbar.set_description(
                'Train loss: %.6f' % (train_loss / ((iteration_count + 1)*self.batch_size)))

            self.iters += 1

            if iteration_count == 10000:
                break

        with open('./output.txt', 'a') as output_file:
            output_file.write(f'\nEpoch {epoch}:\nTrain loss:')
            output_file.write('\n' + str(train_loss))
            output_file.write('\n' + str(train_loss / (10001*self.batch_size)))
        
        torch.save(self.model.state_dict(), './model_state_dict.pt')
        return train_loss

    def validation(self, epoch):
        self.model.eval()
        tbar = tqdm(self.val_loader, desc='\r')
        val_loss = 0.0

        AP    = np.zeros(self.numClasses + 1)
        PCK   = np.zeros(self.numClasses + 1)
        PCKh  = np.zeros(self.numClasses + 1)
        count = np.zeros(self.numClasses + 1)

        cnt = 0
        for iteration_count, (input, heatmap) in enumerate(tbar):
            cnt += 1

            input_on_cuda = input.cuda()
            heatmap_on_cuda = heatmap.cuda()
            self.optimizer.zero_grad()

            heat = self.model(input_on_cuda)
            loss_heat = self.criterion(heat, heatmap_on_cuda)

            loss = loss_heat

            val_loss += loss_heat.item()

            tbar.set_description(
                'Val loss: %.6f' % (val_loss / ((iteration_count + 1)*self.batch_size)))

            acc, acc_PCK, acc_PCKh, cnt, pred, visible = accuracy(
                output=heat.detach().cpu().numpy(),
                target=heatmap_on_cuda.detach().cpu().numpy(),
                threshold_PCK=0.2, threshold_PCKh=0.5, dataset=self.dataset)

            AP[0]   = (AP[0]  *iteration_count + acc[0])      / (iteration_count + 1)
            PCK[0]  = (PCK[0] *iteration_count + acc_PCK[0])  / (iteration_count + 1)
            PCKh[0] = (PCKh[0]*iteration_count + acc_PCKh[0]) / (iteration_count + 1)

            for j in range(1, self.numClasses+1):
                if visible[j] == 1:
                    AP[j]   = (AP[j]  *count[j] + acc[j])      / (count[j] + 1)
                    PCK[j]  = (PCK[j] *count[j] + acc_PCK[j])  / (count[j] + 1)
                    PCKh[j] = (PCKh[j]*count[j] + acc_PCKh[j]) / (count[j] + 1)
                    count[j] += 1

            mAP   =   AP[1:].sum()/(self.numClasses)
            mPCK  =  PCK[1:].sum()/(self.numClasses)
            mPCKh = PCKh[1:].sum()/(self.numClasses)

        printAccuracies(mAP, AP, mPCKh, PCKh, mPCK, PCK, self.dataset)

        PCKhAvg = PCKh.sum()/(self.numClasses+1)
        PCKAvg  =  PCK.sum()/(self.numClasses+1)

        if mAP > self.isBest:
            self.isBest = mAP
            if self.isBest:
                torch.save({'state_dict': self.model.state_dict()},
                           self.args.model_name + '_best.pth.tar')
            print("Model saved to " + self.args.model_name)

        if mPCKh > self.bestPCKh:
            self.bestPCKh = mPCKh
        if mPCK > self.bestPCK:
            self.bestPCK = mPCK

        print("Best AP = %.2f%%; PCK = %2.2f%%; PCKh = %2.2f%%"
              % (self.isBest*100, self.bestPCK*100, self.bestPCKh*100))
        # "%f": floating point type
        # "%.2f": print only the first 2 decimals (rounded)
        # "%.2f%": error
        # "%.2f%%": write a % sign after the number
        #           % is a special character, it has to be escaped
        # "%2.2f%%": ??? typo???

        with open('./output.txt', 'a') as output_file:
            output_file.write('\nVal loss:')
            output_file.write('\n' + str(val_loss))
            output_file.write('\n' + str(val_loss / (10001*self.batch_size)))

        return val_loss, AP, PCK, PCKh

    def test(self, epoch):
        self.model.eval()
        print("Testing")

        for idx in range(1):
            print(idx,"/",2000)
            img_path = CONF.get('TEST', 'DIR_IMAGES_TEST')

            center   = [184, 184]

            img  = np.array(cv2.resize(cv2.imread(img_path), (368,368)),
                            dtype=np.float32)
            img  = img.transpose(2, 0, 1)
            img  = torch.from_numpy(img)
            mean = [128.0, 128.0, 128.0]
            std  = [256.0, 256.0, 256.0]
            for t, m, s in zip(img, mean, std):
                t.sub_(m).div_(s)

            img = torch.unsqueeze(img, 0)

            self.model.eval()

            input_var = img.cuda()

            heat = self.model(input_var)
            heat = interpolate(
                heat, size=input_var.size()[2:], mode='bilinear',
                align_corners=True)

            kpts = get_kpts(heat, img_h=368.0, img_w=368.0)
            draw_paint(img_path, kpts, idx, epoch, self.dataset)

            heat = heat.detach().cpu().numpy()
            heat = heat[0].transpose(1,2,0)

            for i in range(heat.shape[0]):
                for j in range(heat.shape[1]):
                    for k in range(heat.shape[2]):
                        if heat[i,j,k] < 0:
                            heat[i,j,k] = 0

            im = cv2.resize(cv2.imread(img_path), (368,368))

            heatmap = []
            for i in range(self.numClasses+1):
                heatmap = cv2.applyColorMap(
                    np.uint8(255*heat[:,:,i]), cv2.COLORMAP_JET)
                im_heat = cv2.addWeighted(im, 0.6, heatmap, 0.4, 0)
                cv2.imwrite('samples/heat/unipose' + str(i) + '.png', im_heat)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', default=False, choices=[False, True],
                        type=bool, help="""
    If True, the pretrained weights will be loaded with torch.load()
    from the conf.ini file PRETRAINED section PATH_TO_WEIGHTS value.
    Specify the value to correctly use the argument.
    """)
    parser.add_argument('--dataset', default='MPII', choices=['MPII'], type=str)
    parser.add_argument(
        '--model_name', default='unipose', type=str,
        help="Used as a filename to save the best performing model.")
    parser.add_argument(
        '--test', default=False, choices=[False, True], type=bool,
        help="If True, the model will do testing instead of training.")

    args = parser.parse_args()

    trainer = Trainer(args)

    if args.test:
        trainer.test(0)
    else:
        epochs = CONF.getint('HYPERPARAMETERS', 'EPOCHS')
        numClasses = None
        try:
            numClasses = CONF.getint(args.dataset, 'NUMBER_OF_CLASSES')
        except configparser.NoSectionError:
            print("Section not found: ", args.dataset)
            if args.dataset in ['MPII']:
                print("Edit your conf.ini file.")
            else:
                print("Invalid dataset choice.")
            sys.exit()

        APs, PCKs, PCKhs = {}, {}, {}
        for index in range(numClasses + 1):
            APs[index], PCKs[index], PCKhs[index] = [], [], []
        losses = {'train': [], 'validation': []}

        for epoch in range(epochs):
            train_loss = trainer.training(epoch)
            val_loss, AP, PCK, PCKh = trainer.validation(epoch)

            losses['train'].append(train_loss)
            losses['validation'].append(val_loss)
            for key in AP.keys():
                APs[key].append(AP[key])
                PCKs[key].append(PCK[key])
                PCKhs[key].append(PCKh[key])

        plotting(args.dataset, epochs, APs, PCKs, PCKhs, losses)
