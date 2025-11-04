'''
Classifier model is adapted from KataGos' v.16 model_pytorch.py, so layer ativations from KataGo
models are compatible with the classifier. 

Most of the changes from model_pytorch.py are to simplifiy the class, or to remove sections that handle old
versions of KataGo.
'''

import torch
import math
import engines.KataGo.python.katago.train.modelconfigs as modelconfigs
import numpy as np
from typing import Optional

'''
---description---
copied from model.py, removed call to compute_gain() since all norm is fixup, so compute_gain() isn't needed.

---inputs---
tensor: tensor: the tensor being intitalised 
scale: scaling factor for initialised weights
'''
def init_weights(tensor, scale, fan_tensor=None):
    gain = math.sqrt(2.0)

    if fan_tensor is not None:
        (fan_in, _) = torch.nn.init._calculate_fan_in_and_fan_out(fan_tensor)
    else:
        (fan_in, _) = torch.nn.init._calculate_fan_in_and_fan_out(tensor)

    target_std = scale * gain / math.sqrt(fan_in)
    # Multiply slightly since we use truncated normal
    std = target_std / 0.87962566103423978
    if std < 1e-10:
        tensor.fill_(0.0)
    else:
        torch.nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2.0*std, b=2.0*std)

class BiasMask(torch.nn.Module):
    def __init__(
        self,
        c_in,
        config: modelconfigs.ModelConfig,
        is_after_batchnorm: bool = False,
    ):
        super(BiasMask, self).__init__()
        self.c_in = c_in
        self.beta = torch.nn.Parameter(torch.zeros(1, c_in, 1, 1))
        self.is_after_batchnorm = is_after_batchnorm
        self.scale = None

    def set_scale(self, scale: Optional[float]):
        self.scale = scale
    
    def forward(self, x, mask, mask_sum: float):
        """
        Parameters:
        x: NCHW
        mask: N1HW
        mask_sum: scalar

        Returns: NCHW
        """
        if self.scale is not None:
            return (x * self.scale + self.beta) * mask
        else:
            return (x + self.beta) * mask
        

class KataValueHeadGPool(torch.nn.Module):
    def __init__(self):
        super(KataValueHeadGPool, self).__init__()

    def forward(self, x, mask, mask_sum_hw):
        """
        Parameters:
        x: NCHW
        mask: N1HW
        mask_sum_hw: N111

        Returns: NC11
        """
        mask_sum_hw_sqrt_offset = torch.sqrt(mask_sum_hw) - 14.0

        layer_mean = torch.sum(x, dim=(2, 3), keepdim=True, dtype=torch.float32) / mask_sum_hw

        out_pool1 = layer_mean
        out_pool2 = layer_mean * (mask_sum_hw_sqrt_offset / 10.0)
        out_pool3 = layer_mean * ((mask_sum_hw_sqrt_offset * mask_sum_hw_sqrt_offset) / 100.0 - 0.1)

        out = torch.cat((out_pool1, out_pool2, out_pool3), dim=1)
        return out


'''
---desciption---
Normalising layer from model.py, but simplified to only use fixup normalisation.

---parametres---
c_in: NCHW: previous layer output
config: the config file of the model used as an embedding space, not the model being defined
fixup_use_gamma: bool; whether or not the fixup norm should use a gamma value, which is a scalar
on x.
force_use_gamma: bool: can be maually toggled to force the use of gamma values
is_last_batcnorm: bool: well named 

---returns---
*: NCHW: output of normalisation layer
'''

class NormMask(torch.nn.Module):
    def __init__(
            self,
            c_in,
            config: modelconfigs.ModelConfig,
            fixup_use_gamma: bool,
            force_use_gamma = False,
            is_last_batchnorm: bool = False,
    ):
        super(NormMask, self).__init__()
        #models all used fixup so code was simplified to only use fixup
        self.epsilon = config["bnorm_epsilon"]
        self.running_avg_momentum = config["bnorm_running_avg_momentum"]
        self.fixup_use_gamma = fixup_use_gamma
        self.is_last_batchnorm = is_last_batchnorm
        self.use_gamma = True
        self.c_in = c_in

        self.scale = None
        self.gamma = None
        self.is_using_batchnorm = False #because we are always using fixup

        self.beta = torch.nn.Parameter(torch.zeros(1, c_in, 1, 1))
        if self.use_gamma:
            self.gamma = torch.nn.Parameter(torch.ones(1, c_in, 1, 1))

    def set_scale(self, scale: Optional[float]):
        self.scale = scale

    #The kataGo version of this takes in mask, which masks out the empty spaces on the board
    #Becuse we are training outside the KataGo Framework we can't access mask so its removed
    def apply_gamma_beta_scale_mask(self, x, mask):
        if self.scale is not None:
            if self.gamma is not None:
                return (x * (self.gamma * self.scale) + self.beta) * mask
            else:
                return (x * self.scale + self.beta) * mask
        else:
            if self.gamma is not None:
                return (x * self.gamma + self.beta) * mask
            else:
                return (x + self.beta) * mask
            
    def forward(self, x, mask):
        return self.apply_gamma_beta_scale_mask(x, mask)

'''
---description---
Global pooling from KataGo model.

---parametres---
x: NCHW: output from previous layer

---returns---
out: NCHW: pooling values
'''

class KataGPool(torch.nn.Module):
    def __init__(self):
        super(KataGPool, self).__init__()

    def forward(self, x, mask, mask_sum_hw):
        mask_sum_hw_sqrt_offset = torch.sqrt(mask_sum_hw) - 14.0

        layer_mean = torch.sum(x, dim=(2, 3), keepdim=True, dtype=torch.float32) / mask_sum_hw
        # All activation functions we use right now are always greater than -1.0, and map 0 -> 0.
        # So off-board areas will equal 0, and then this max is mask-safe if we assign -1.0 to off-board areas.
        (layer_max,_argmax) = torch.max((x+(mask-1.0)).view(x.shape[0],x.shape[1],-1).to(torch.float32), dim=2)
        layer_max = layer_max.view(x.shape[0],x.shape[1],1,1)

        out_pool1 = layer_mean
        out_pool2 = layer_mean * (mask_sum_hw_sqrt_offset / 10.0)
        out_pool3 = layer_max

        out = torch.cat((out_pool1, out_pool2, out_pool3), dim=1)
        return out


'''
---description---
Adapted from KataGos' model_pytorch.KataConvAndGPool()
Convolutional block that forms the backbone of the classifier. 
Made simpler as the models we are using all have the same config.
Block architecture:

    ┌------------------ rConv2d -------------------┐
in -┤                                              ⊕ - out
    └- gConv2D - FixUpNorm - ReLU - GPool - Linear ┘

---parametres---
name: str: a unique identifier for the block
config: modelconfigs.config: the config for the KataGo Model the layer activations originiate from
c_in: NCHW: number of input channels
c_out: tensor: number of output channels
c_gpool: KataGPool: number of global pooling channels

---returns--
out: NCHW: output of convr and convg concatenated together
'''
class ConvAndGPool(torch.nn.Module):
    def __init__(self, name, c_in, c_out, c_gpool, config):
        super(ConvAndGPool, self).__init__()
        self.name = name
        self.norm_kind = config["norm_kind"]
        self.activation = torch.nn.ReLU()
        self.conv1r = torch.nn.Conv2d(c_in, c_out, kernel_size=3, padding="same", bias=False)
        self.conv1g = torch.nn.Conv2d(c_in, c_gpool, kernel_size=3, padding="same", bias=False)
        self.normg = NormMask(
            c_gpool,
            config=config,
            fixup_use_gamma=False,
        )
        self.actg = torch.nn.ReLU(inplace = True)
        self.gpool = KataGPool()
        self.linear_g = torch.nn.Linear(3*c_gpool, c_out, bias=False)

    def initialize(self, scale):
        # Scaling so that variance on the r and g branches adds up to 1.0
        r_scale = 0.8
        g_scale = 0.6
        init_weights(self.conv1r.weight, scale=scale * r_scale)
        init_weights(self.conv1g.weight, scale=math.sqrt(scale) * math.sqrt(g_scale))
        init_weights(self.linear_g.weight, scale=math.sqrt(scale) * math.sqrt(g_scale))



    def forward(self, x, mask, mask_sum_hw):
        """
        Parameters:
        x: NCHW
        mask: N1HW
        mask_sum_hw: N111
        mask_sum: scalar

        Returns: NCHW
        """
        out = x
        outr = self.conv1r(out)
        outg = self.conv1g(out)

        outg = self.normg(outg, mask=mask)
        outg = self.actg(outg)
        outg = self.gpool(outg, mask=mask, mask_sum_hw=mask_sum_hw).squeeze(-1).squeeze(-1)
        outg = outg.float()
        outg = self.linear_g(outg).unsqueeze(-1).unsqueeze(-1)

        out = outr + outg
        return out

'''
---description---
Full advrsarial detection model. Architecture bsed off the Value output heads of KataGo.

---parametres---
kata_model_configs: ModelConfig: The config for the KataGo model neign used as an embedding
pos_len: int: the board size/ length of position array

--output---
out: model output
'''
class Model(torch.nn.Module):

    def __init__(self, kata_model_config: modelconfigs.ModelConfig, pos_len: int):

        super(Model, self).__init__()

        self.kata_model_config = kata_model_config
        self.pos_len = pos_len
        self.c_trunk = self.kata_model_config["trunk_num_channels"]
        self.c_gpool = kata_model_config["gpool_num_channels"]
        self.c_v1 = kata_model_config["v1_num_channels"]
        self.c_v2 = kata_model_config["v2_size"]
        self.num_total_blocks = len(kata_model_config['block_kind'])

        self.activation = torch.nn.ReLU(inplace = True)

        self.conv_block1 = ConvAndGPool('rconvDet', self.c_trunk, self.c_trunk, self.c_gpool, self.kata_model_config)
        self.conv1 = torch.nn.Conv2d(self.c_trunk, self.c_v1, kernel_size=1, padding="same", bias=False)
        self.bias1 = BiasMask(
            self.c_v1,
            config=kata_model_config,
            is_after_batchnorm=True,
        )
        self.act1 = torch.nn.ReLU(inplace=True)
        self.gPool1 = KataValueHeadGPool()
        self.lin1 = torch.nn.Linear(3*self.c_v1, self.c_v2, bias = False)
        self.act2 = torch.nn.ReLU(inplace=True)
        self.lin_final = torch.nn.Linear(self.c_v2, 2, bias = False)
        self.softmax = torch.nn.Softmax(dim=1)

    def initialize(self):
        super(Model, self).__init__()

        fixup_scale = 1.0 / math.sqrt(self.num_total_blocks)
        bias_scale = 0.2

        self.conv_block1.intialize(fixup_scale)

        self.conv1 = torch.nn.Conv2d(self.c_trunk, self.c_v1, kernel_size=1, padding="same", bias=False)
        self.bias1 = BiasMask(
            self.c_v1,
            config = self.kata_model_config,
            is_after_batchnorm=True,
        )
        self.act1 = torch.nn.ReLU(inplace=True)
        self.gpool = KataGPool()

        self.linear2 = torch.nn.Linear(3 * self.c_v1, self.c_v2, bias=True)
        self.act2 = torch.nn.ReLU(inplace=True)

    def forward(self, x):

        mask = np.ones([19,19], dtype=float)
        mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0)
        mask_sum_hw = torch.sum(mask,dim=(2,3),keepdim=True)
        mask_sum = torch.sum(mask)

        out = x

        out = self.conv_block1(out, mask, mask_sum_hw)
        out = self.conv1(out)
        out = self.bias1(out, mask = mask, mask_sum=mask_sum)
        out = self.act1(out)
        out = self.gPool1(out, mask, mask_sum_hw).squeeze(-1).squeeze(-1)
        out = out.float()
        out = self.lin1(out)
        out = self.act2(out)
        out = self.lin_final(out)

        return(out)





