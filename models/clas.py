import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register

from .mmcls.models import build_classifier

import os
import logging
logger = logging.getLogger(__name__)

from torch.nn.modules.utils import _pair

from src.utils import logging
loggers = logging.get_logger("visual_prompt")
import torch
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from models import register
from .mmcls.models import build_classifier, build_loss
from src.utils import logging as utils_logging

logger = logging.getLogger(__name__)
loggers = utils_logging.get_logger("visual_prompt")


class BinaryCrossEntropyLoss(nn.Module):
    '''
    Binary Cross Entropy Loss for binary classification
    '''

    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()

    def forward(self, pred, label):
        # 确保标签为浮点型
        label = label.float()

        # 如果 pred 是 [batch_size, 2]，则选择 logits 的第一列
        if pred.dim() == 2 and pred.size(1) == 2:
            pred = pred[:, 1]  # 选择第二类的 logits

        # 计算普通的二元交叉熵损失
        loss = F.binary_cross_entropy_with_logits(pred, label)

        return loss


class BinaryCrossEntropyWithWeight(nn.Module):
    '''
    Binary Cross Entropy with class weights for binary classification
    '''

    def __init__(self):
        super(BinaryCrossEntropyWithWeight, self).__init__()

    def compute_class_weights(self, label):
        eps = 1e-10
        count_pos = torch.bincount(label, minlength=2).float() + eps  # Ensure at least 2 classes
        total = label.size(0) + eps
        weights = total / (2 * count_pos)  # Return weights for both classes
        return weights.to(label.device)

    def forward(self, pred, label):
        eps = 1e-10
        label = label.long()
        count_pos = torch.sum(label) + eps
        count_neg = torch.sum(1. - label) + eps

        # 计算正负样本的权重
        w_pos = count_neg / (count_pos + count_neg)
        w_neg = count_pos / (count_pos + count_neg)

        # 创建 one-hot 编码的目标标签
        onehot_target = torch.eye(2, device=label.device)[label]
        # 确保权重与 label 相同的设备和类型
        weights = torch.zeros_like(onehot_target).float().to(label.device)
        weights[:, 0] = w_neg  # 对应于负类
        weights[:, 1] = w_pos  # 对应于正类
        # 逐样本计算损失
        loss_s = nn.functional.binary_cross_entropy_with_logits(
            pred, onehot_target, weight=weights, reduction='none'
        )  # 形状：[batch_size, num_classes]

        return loss_s  # 返回每个样本的损失


@register('clas')
class CLAS(nn.Module):
    def __init__(self, cfg, inp_size=None, encoder_mode=None, loss=None):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(encoder_mode)
        model_config = None
        # print(encoder_mode['name']=='evpcma')
        if 'vpt' in encoder_mode['name']:
            backbone = dict(type='VisionTransformerVPT', img_size=inp_size, pos_embed_interp=True, drop_rate=0.,
                            model_name='vit_base_patch16_384', embed_dim=768, depth=12, num_heads=12
                            )
        elif 'adaptformer' in encoder_mode['name']:
            backbone = dict(type='VisionTransformerAdaptFormer', img_size=inp_size, pos_embed_interp=True, drop_rate=0.,
                            model_name='vit_base_patch16_384', embed_dim=768, depth=12, num_heads=12
                            )
        elif 'evp' in encoder_mode['name']:
            backbone = dict(type='VisionTransformerEVP', img_size=inp_size, pos_embed_interp=True, drop_rate=0.1,
                            model_name='vit_base_patch16_224', embed_dim=768, depth=12, num_heads=12,
                            scale_factor=encoder_mode['scale_factor'], input_type=encoder_mode['input_type'],
                            freq_nums=encoder_mode['freq_nums'], prompt_type=encoder_mode['prompt_type'],
                            tuning_stage=encoder_mode['tuning_stage'],
                            handcrafted_tune=encoder_mode['handcrafted_tune'],
                            embedding_tune=encoder_mode['embedding_tune'], adaptor=encoder_mode['adaptor'], )
        elif 'LCDF' in encoder_mode['name']:
            print('**********ViTBAAACMA****')
            # vit_base_patch16_224_FMAG vit_base_patch16_224 beit_base_patch16_224
            backbone = dict(type='VisionTransformerBat', model_name='vit_base_patch16_224',
                            patch_size=encoder_mode['patch_size'], patch_size_prompt=encoder_mode['patch_size_prompt'],
                            num_classes=2, embed_dim=768, depth=12, num_heads=12,
                            search_size=_pair(encoder_mode['search_size']),
                            template_size=_pair(encoder_mode['search_size']), new_patch_size=224, adapter_type='bat',
                            adapter_dim=encoder_mode['adapter_dim'],
                            scale_factor=encoder_mode['scale_factor'], pretrain=encoder_mode['pretrain'])

        elif 'Swinbc' in encoder_mode['name']:
            # vit_base_patch16_224_FMAG vit_base_patch16_224 swin_base_patch4_window7_224_pre
            backbone = dict(type='SwinBatCMA', model_name='swin_base_patch4_window7_224_pre',
                            patch_size=encoder_mode['patch_size'], patch_size_prompt=encoder_mode['patch_size_prompt'],
                            num_classes=2, embed_dim=128, depth=12, num_heads=12,
                            search_size=_pair(encoder_mode['search_size']),
                            template_size=_pair(encoder_mode['search_size']), new_patch_size=224, adapter_type='bat',
                            adapter_dim=encoder_mode['adapter_dim'],
                            scale_factor=encoder_mode['scale_factor'], pretrain=encoder_mode['pretrain'])
            model_config = dict(
                type='ImageClassifier',
                backbone=backbone,
                neck=None,
                head=dict(
                    type='SwinHead',
                    in_channels=1024,
                    num_classes=2,
                    act_cfg=dict(type='tanh'),
                    init_cfg=dict(type='Constant', layer='Linear', val=0),
                    mlp_cfg=True,
                    mlp_act=nn.ReLU,
                    mlp_num=0,
                    loss=dict(
                        type='CrossEntropyLoss', use_sigmoid=False,
                        use_soft=False,
                        reduction='mean',
                        loss_weight=1.0),
                    hidden_dim=None
                ),
                train_cfg=dict(), )
        else:
            backbone = dict(type='VisionTransformer', img_size=inp_size, pos_embed_interp=True, drop_rate=0.,
                            model_name='vit_base_patch16_224', embed_dim=768, depth=12, num_heads=12
                            )
        print(backbone)
        if model_config is None:
            model_config = dict(
                type='ImageClassifier',
                backbone=backbone,
                neck=None,
                head=dict(
                    type='VisionTransformerClsHead',
                    in_channels=768,
                    num_classes=2,
                    act_cfg=dict(type='tanh'),
                    init_cfg=dict(type='Constant', layer='Linear', val=0),
                    mlp_cfg=True,
                    mlp_act=nn.ReLU,
                    mlp_num=0,
                    loss=dict(
                        type='CrossEntropyLoss', use_sigmoid=False,
                        use_soft=False,
                        reduction='mean',
                        loss_weight=1.0),
                    hidden_dim=None
                ),
                train_cfg=dict(),
            )

        loggers.info(f"Model Config: {model_config}")

        model = build_classifier(
            model_config,

        )

        self.encoder = model

        if 'vpt' in encoder_mode['name']:
            for k, p in self.encoder.named_parameters():
                if "prompt" not in k and "decode_head" not in k and "auxiliary_head" not in k:
                    p.requires_grad = False
        if 'adaptformer' in encoder_mode['name']:
            for k, p in self.encoder.named_parameters():
                if "adaptmlp" not in k and "decode_head" not in k and "auxiliary_head" not in k:
                    p.requires_grad = False
        if 'linear' in encoder_mode['name']:
            for k, p in self.encoder.named_parameters():
                if "decode_head" not in k and "auxiliary_head" not in k:
                    p.requires_grad = False
        if 'LCDF' in encoder_mode['name']:
            for k, p in self.encoder.named_parameters():
                if "patch_embed_prompt" in k:
                    p.requires_grad = encoder_mode['handcrafted_tune']
                if "patch_embed_prompt_rgb" in k:
                    p.requires_grad = encoder_mode['embedding_tune']
                elif "adapter" not in k and "head" not in k:
                    p.requires_grad = False
                logger.info('{}: {}'.format(k, p.requires_grad))
        if 'bacma2' in encoder_mode['name']:
            for k, p in self.encoder.named_parameters():
                if "patch_embed." in k:
                    p.requires_grad = False
                    # p.requires_grad = encoder_mode['embedding_tune']
                if "patch_embed_prompt" in k:
                    p.requires_grad = encoder_mode['handcrafted_tune']
                elif "adapter" not in k and "head" not in k :
                    p.requires_grad = False
        if 'Swinbc' in encoder_mode['name']:
            for k, p in self.encoder.named_parameters():
                if "patch_embed." in k:
                    p.requires_grad = False
                    # p.requires_grad = encoder_mode['embedding_tune']
                if "patch_embed_prompt" in k:
                    p.requires_grad = encoder_mode['handcrafted_tune']
                elif "adapter" not in k and "head" not in k:
                    p.requires_grad = False
        if 'evp' in encoder_mode['name']:
            for k, p in self.encoder.named_parameters():
                if "prompt" not in k and "head" not in k:
                    p.requires_grad = False
                logger.info('{}: {}'.format(k, p.requires_grad))

        model_total_params = sum(p.numel() for p in self.encoder.parameters())
        model_grad_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        # print('model_grad_params:' + str(model_grad_params),
        #       '\nmodel_total_params:' + str(model_total_params))
        logger.info(f"model_grad_params: {model_grad_params}\nmodel_total_params: {model_total_params}")

        self.loss_mode = loss
        if self.loss_mode == 'be':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()
        elif self.loss_mode == 'bce':
            self.criterionBCE = BinaryCrossEntropyLoss()
        elif self.loss_mode == 'bcew':
            self.criterionBCEW = BinaryCrossEntropyWithWeight()
        elif self.loss_mode == 'Sigm':
            self.cls_criterion = build_loss('softmax')

    def set_input(self, input, label, input2=None):
        # print("*****Set input*******")

        self.input = input.to(self.device)
        self.label = label.to(self.device)

        if input2 != None:
            self.input2 = input2.to(self.device)

    def forward_feature(self, input):
        self.pred_feature = self.encoder.extract_feat(self.input, self.input2, stage='backbone')
        return self.pred_feature

    def forward_feature_mul(self, input, input2=None):
        self.pred_feature = self.encoder.extract_multiple_feats(self.input, self.input2)
        self.pred_logit = self.pred_feature['pre_logits']
        return self.pred_feature

    def forward(self, input, input2=None):
        self.pred_logit = self.encoder.forward_dummy(self.input, self.input2)

        return self.pred_logit

    def backward_G(self, is_train, cls_weights):
        if self.loss_mode == 'Sigm':
            self.loss_G = self.cls_criterion(self.pred_logit, self.label)
        elif self.loss_mode == 'bcew':
            self.loss_G = self.criterionBCEW(self.pred_logit, self.label)
        elif self.loss_mode == 'bce':
            self.loss_G = self.criterionBCE(self.pred_logit, self.label)
        elif self.loss_mode == 'focal':
            self.loss_G = self.criterionFOCAL(self.pred_logit, self.label)
        if (is_train):
            self.loss_G.backward()

    def optimize_parameters(self, is_train, cls_weights):
        # self.forward()
        if is_train:
            self.backward_G(False, cls_weights)

        else:
            self.backward_G(False, cls_weights)


    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad