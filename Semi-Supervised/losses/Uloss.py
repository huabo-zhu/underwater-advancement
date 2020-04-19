import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import cv2
import numpy as np


class U_loss(nn.Module):
    def __init__(self, D, fineSize, use_lsgan=True, device=torch.device('cpu')):
        super(U_loss, self).__init__()
        self.device = device
        self.D = D
        self.conv_size, self.critic = self.get_conv_size(fineSize)
        self.final_size = self.conv_size.pop()
        self.conv_size.reverse()
        self.critic.reverse()
        self.Umap0 = torch.zeros((self.final_size, self.final_size), requires_grad=False).to(self.device)
        self.rf_map = np.zeros((self.final_size,self.final_size, 4)).astype(int)
        for i in range(self.final_size):  #x
            for j in range(self.final_size): #y
                self.rf_map[i, j] = self.get_rf_original(i, j)
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_conv_size(self, fineSize):
        critic = []
        for layer in list(self.D.trunk) + list(self.D.critic_branch):
            if isinstance(layer, nn.Conv2d):
                critic.append(layer)

        conv_size = [fineSize, ]
        in_size = conv_size[0]
        for conv_layer in critic:
            out_size = np.floor((in_size + 2*conv_layer.padding[0] - conv_layer.kernel_size[0])/conv_layer.stride[0] + 1)
            conv_size.append(int(out_size))
            in_size = out_size
        return conv_size, critic

    def receive_field(self, x, y, stride, padding, kernel_size):

        x_min = (x - 1) * stride + 1 - padding
        y_min = (y - 1) * stride + 1 - padding
        x_max = (x - 1) * stride - padding + kernel_size
        y_max = (y - 1) * stride - padding + kernel_size

        return x_min, y_min, x_max, y_max

    def get_rf_original(self, x, y):

        x_min, y_min, x_max, y_max = x, y, x, y
        for conv_layer, pre_feature_size in zip(self.critic, self.conv_size):
            x_min, y_min, _, _ = self.receive_field(x_min, y_min, conv_layer.stride[0], conv_layer.padding[0], conv_layer.kernel_size[0])
            _, _, x_max, y_max = self.receive_field(x_max, y_max, conv_layer.stride[0], conv_layer.padding[0], conv_layer.kernel_size[0])
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(pre_feature_size, x_max), min(pre_feature_size, y_max)

        return int(x_min), int(y_min), int(x_max), int(y_max)

    def get_Umap(self, input):
        underwater_index_batchmap = []
        for instance in input:
            # AorB = np.random.rand()
            underwater_index_map = np.zeros((1, self.final_size, self.final_size))
            image = cv2.normalize(instance.detach().cpu().float().numpy().transpose(1, 2, 0),
                                       None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
            # image = image_cat[: ,:, 3:] if AorB>0.5 else image_cat[: ,:, :3]
            image_lab = cv2.normalize(cv2.cvtColor(image, cv2.COLOR_RGB2Lab), None, alpha=0, beta=1,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC3)

            for i in range(self.final_size): #y
                for j in range(self.final_size) : #x

                    image_sub_l = image_lab[self.rf_map[j, i, 1]:self.rf_map[j, i, 3], self.rf_map[j, i, 0]:self.rf_map[j, i, 2], 0]
                    image_sub_a = image_lab[self.rf_map[j, i, 1]:self.rf_map[j, i, 3], self.rf_map[j, i, 0]:self.rf_map[j, i, 2], 1]
                    image_sub_b = image_lab[self.rf_map[j, i, 1]:self.rf_map[j, i, 3], self.rf_map[j, i, 0]:self.rf_map[j, i, 2], 2]
                    lab_bias = np.sqrt(np.sqrt((np.mean(image_sub_a) - 0.5) ** 2 + (np.mean(image_sub_b) - 0.5) ** 2) / (0.5 * np.sqrt(2)))
                    lab_var = (np.max(image_sub_a) - np.min(image_sub_a)) * (np.max(image_sub_b) - np.min(image_sub_b))
                    lab_light = np.mean(image_sub_l)
                    underwater_index_map[0, j, i] = lab_bias / (10*lab_var*lab_light)


            underwater_index_batchmap.append(underwater_index_map)
        with torch.no_grad():
            Umap = torch.from_numpy(np.array(underwater_index_batchmap)).type(torch.FloatTensor).to(self.device)
        return Umap

    def __call__(self, image, pred_critic, model='G'):
        Umap = self.Umap0.expand_as(pred_critic) if model == 'G' else self.get_Umap(image)
        return self.loss(pred_critic, Umap)