import torch
import torch.nn as nn
import torch.nn.functional as F


################################
#     Asymmetric Focal loss    #
################################
def asymmetric_focal_loss(delta=0.7, gamma=2.):
    def loss_function(y_true, y_pred):
        epsilon = 1e-6
        y_pred = F.sigmoid(y_pred)

        y_pred = torch.clamp(input=y_pred, min=epsilon, max=1. - epsilon)
        cross_entropy = -y_true * torch.log(y_pred)

        # calculate losses separately for each class, only suppressing background class
        back_ce = torch.pow(1 - y_pred[:, :, :, 0], gamma) * cross_entropy[:, :, :, 0]
        back_ce = (1 - delta) * back_ce

        fore_ce = cross_entropy[:, :, :, 1]
        fore_ce = delta * fore_ce

        loss = torch.mean(torch.sum(torch.cat([back_ce, fore_ce], dim=-1), dim=-1))

        return loss

    return loss_function


#################################
# Asymmetric Focal Tversky loss #
#################################
def asymmetric_focal_tversky_loss(delta=0.7, gamma=0.75):
    """This is the implementation for binary segmentation.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    """

    def loss_function(y_true, y_pred):
        # Clip values to prevent division by zero error

        epsilon = 1e-6

        y_pred = F.sigmoid(y_pred)
        y_pred = torch.clip(input=y_pred, min=epsilon, max=1. - epsilon)

        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = torch.sum(y_true * y_pred, dim=[0, 2, 3], keepdim=True)
        fn = torch.sum(y_true * (1 - y_pred), dim=[0, 2, 3], keepdim=True)
        fp = torch.sum((1 - y_true) * y_pred, dim=[0, 2, 3], keepdim=True)
        dice_class = (tp + epsilon) / (tp + delta * fn + (1 - delta) * fp + epsilon)

        # calculate losses separately for each class, only enhancing foreground class
        back_dice = (1 - dice_class[:, 0])
        fore_dice = (1 - dice_class[:, 1]) * torch.pow(1 - dice_class[:, 1], -gamma)

        # Average class scores
        loss = torch.mean(torch.cat([back_dice, fore_dice], dim=-1))
        return loss

    return loss_function


###########################################
#      Asymmetric Unified Focal loss      #
###########################################
def asym_unified_focal_loss(weight=0.5, delta=0.6, gamma=0.5
                            , y_true=None, y_pred=None):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to asymmetric Focal Tversky loss and asymmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    """

    def loss_function(y_true, y_pred):
        asymmetric_ftl = asymmetric_focal_tversky_loss(delta=delta, gamma=gamma)(y_true, y_pred)
        asymmetric_fl = asymmetric_focal_loss(delta=delta, gamma=gamma)(y_true, y_pred)
        if weight is not None:
            return (weight * asymmetric_ftl) + ((1 - weight) * asymmetric_fl)
        else:
            return asymmetric_ftl + asymmetric_fl

    return loss_function(y_true, y_pred)


# ----------------------------------#
# dice loss
# ----------------------------------#
def Dice_loss(pre, tar):
    assert pre.shape == tar.shape, "Compute dice loss error!"

    pre = F.softmax(pre, dim=1)

    union = (pre * tar).sum()

    A1 = pre.sum()
    A2 = tar.sum()

    loss = union / (A1 + A2 - union + 1e-5)
    return 1 - loss.sum()


# ----------------------------------#
# focal loss
# ----------------------------------#
def focal_loss(pre, tar, alpha=0.25, bate=2.):
    assert pre.shape == tar.shape, f"Compute focal loss error! pre:{pre.shape}  tar:{tar.shape}"

    b, c, h, w = pre.shape
    if c < 3:
        pre = F.sigmoid(pre)
    else:
        pre = F.softmax(pre)

    obj = (tar != 0).float()
    noobj = (tar == 0).float()

    obj_loss = (-1) * alpha * (1 - pre) ** bate * torch.log(pre) * obj

    noobj_loss = (-1) * (1 - alpha) * pre ** bate * torch.log(1 - pre) * noobj

    loss = obj_loss + noobj_loss

    return loss.mean()


# -------------------------------------------#
# score
# -------------------------------------------#
def score(pre, tar, thes=0.5):
    assert pre.shape == tar.shape, "Compute dice loss error!"
    pre = F.softmax(pre, dim=1)
    # ------------------------------#
    #
    # ------------------------------#
    pre = torch.ge(pre, thes)

    union = (pre * tar).sum()

    A1 = pre.sum()
    A2 = tar.sum()

    cost = union / (A1 + A2 - union + 1e-5)
    return cost.mean()


class PixContrasrt(nn.Module):
    def __init__(self,
                 thresh=5,
                 max_epoch=None):
        super(PixContrasrt, self).__init__()

        self.dist = nn.Parameter(torch.tensor(0.2,
                                              requires_grad=True))
        self.thresh = thresh

        self.max_epoch = max_epoch

    def forward(self, feature, label, iteration):
        # 前景激活层
        masked = feature * label

        # 背景激活层
        unmasked = feature * (1 - label)

        # 余弦对抗
        # s_neg = 0.5 * nn.CosineSimilarity()(masked,(unmasked ))
        # #loss = -torch.log2((s_neg + 1e-8))
        # loss = s_neg.mean()
        #
        # return loss

        # 以余弦距离来构建三元组损失

        undist = nn.CosineSimilarity()(label, unmasked).mean()

        maskdist = nn.CosineSimilarity()(label, masked).mean()

        distloss = torch.where(
            undist > maskdist + self.dist,
            1.5 * undist + 0.8 * maskdist,
            undist + 0.5 * maskdist
        )

        return torch.where(torch.tensor(iteration) < self.thresh,
                           distloss,
                           (m) / (1 - m + 1e-5) * distloss)


class Heatmaploss(nn.Module):
    def __init__(self, reduce='sum', alpha=2, bate=4):
        super(Heatmaploss, self).__init__()
        self.reduce = reduce

        self.alpha = alpha

        self.bate = bate

        self.loss = nn.BCELoss()

    def forward(self, input, target):
        # input = input.permute((0, 2, 3, 1))

        input = torch.sigmoid(input)

        # ---------------------------------------------------------------#
        #
        # ---------------------------------------------------------------#
        input = torch.clamp(input, 1e-6, 1 - 1e-6)

        pos = (target == 1).float()
        neg = (target < 1).float()

        pos_loss = self.loss(pos * input, pos * target)
        neg_loss = self.loss(neg * input, neg * target)

        N = pos.sum()

        loss = pos_loss.sum() + 0.5 * neg_loss.sum()

        return - (loss / N)





if __name__ == '__main__':
    label = torch.ones((1, 2, 256, 256))

    pred = torch.randn((1, 2, 256, 256))

    loss = asym_unified_focal_loss(y_true=label, y_pred=pred)


