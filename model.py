import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class VIN(nn.Module):
    def __init__(self, config):
        super(VIN, self).__init__()
        self.config = config
        self.h = nn.Conv2d(
            in_channels=config.l_i,
            out_channels=config.l_h,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=True)
        self.r = nn.Conv2d(
            in_channels=config.l_h,
            out_channels=1,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=False)
        self.q = nn.Conv2d(
            in_channels=1,
            out_channels=config.l_q,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=False)
        self.fc = nn.Linear(in_features=302400, out_features=config.l_q, bias=False)
        self.w = Parameter(torch.zeros(config.l_q, 1, 3, 3), requires_grad=True)
        self.sm = nn.Softmax(dim=1)

    def forward(self, X, config):
        h = self.h(X)
        r = self.r(h)
        q = self.q(r)
        v, _ = torch.max(q, dim=1, keepdim=True)
        for i in range(0, config.k - 1):
            q = F.conv2d(
                torch.cat([r, v], 1),
                torch.cat([self.q.weight, self.w], 1),
                stride=1,
                padding=1)
            v, _ = torch.max(q, dim=1, keepdim=True)

        q = F.conv2d(
            torch.cat([r, v], 1),
            torch.cat([self.q.weight, self.w], 1),
            stride=1,
            padding=1)
        # print(X.size()[2])
        # print("q.size()",q.size())
        # slice_s1 = S1.long().expand(X.size(3), 1, config.l_q, q.size(0))
        # slice_s1 = slice_s1.permute(3, 2, 1, 0)
        # print("slice_s1.size()", slice_s1.size())
        # q_out = q.gather(2, slice_s1).squeeze(2)
        #
        # slice_s2 = S2.long().expand(1, config.l_q, q.size(0))
        # slice_s2 = slice_s2.permute(2, 1, 0)
        # q_out = q_out.gather(2, slice_s2).squeeze(2)
        q_out = q.view(q.size(0), -1).contiguous()
        # print(q_out.size())
        logits = self.fc(q_out)
        # logits = self.sm(logits)

        return logits #, self.sm(logits)
