from tqdm import tqdm
import torch
import cv2
import numpy as np
from nets.criterion import Dice_loss, score, \
    Heatmaploss, focal_loss, asym_unified_focal_loss
import torch.nn as nn
from utils.tools import get_lr
from utils.tools import prob2entropy
import torch.nn.functional as F
from utils.config import Config
from nets.Transim.sample import mse_loss
from nets.AHF.AHF_Fusion_U_Net import edl_digamma_loss

def epoch_fit(cur_epoch, total_epoch, save_step,
              model, optimizer, dataloader,
              device, logger, args, save_path,
              eval, local_rank):  # 评估函数是对测试集

    model = model.to(device)

    if torch.cuda.device_count() > 1:
        # -----------------------------------------------------#
        #                  训练的BS要能被显卡数整除
        # -----------------------------------------------------#
        # print(f'GPUS Numbers :{torch.cuda.device_count()}')

        # -------------------------------------------------------#
        #                         设置多卡训练
        # -------------------------------------------------------#
        model = nn.DataParallel(model)

    traindataloader, testdataloader, valdataloader = dataloader

    # 损失函数定义
    criteror = nn.BCEWithLogitsLoss()
    heatloss = Heatmaploss()  # TODO ? loc loss ?

    # 上采样
    up = nn.Upsample(size = args.input_shape,
                     mode = 'bilinear', align_corners = True)

    with tqdm(desc = f"Epoch: [{cur_epoch + 1}]/[{total_epoch}]",
              total = len(traindataloader), mininterval = 0.3, postfix = dict, colour = '#6DB9EF') as pb:

        model = model.train()

        # -------------------------------------#
        # 训练损失计算
        # -------------------------------------#
        total_loss = 0.
        total_cls_loss = 0.
        total_boundary_loss = 0.
        total_heat_loss = 0.
        total_kl_loss = 0.

        # 源域
        for ind, batch in enumerate(traindataloader):
            orimgs, image, label, (png, points, edges, recons, cdfs), (heat2, heat4), (
                (boundary1, boundary2), boundary4, boundary8, boundary16) = batch

            with torch.no_grad():
                # TODO 1
                image = image.to(device)
                label = label.to(device)
                png = png.to(device)

                # TODO 2 heatmap
                heat2 = heat2.to(device)
                heat4 = heat4.to(device)

                # TODO 3 boundary
                boundary1 = boundary1.to(device)

                boundary2 = boundary2.to(device)
                boundary4 = boundary4.to(device)
                boundary8 = boundary8.to(device)
                boundary16 = boundary16.to(device)

                # TODO?
                points = points.to(device)
                edges = edges.to(device)

                # TODO?
                recons = recons.to(device)
                cdfs = cdfs.to(device)

            if args.method == 'Ours':
                # ------------------------------------------------------------------------------------#
                h, w = args.input_shape[ 0 ] // 4, args.input_shape[ 1 ] // 4
                _label = F.interpolate(label, size = (h, w), mode = 'bilinear', align_corners = True)

                (feat_out8, feat_out16, feat_out32), \
                (feat_out_sp2, feat_out_sp4, feat_out_sp8, feat_out_sp16), \
                (pred_loc2, pred_loc4) = model(image, _label, *(boundary2, boundary4))

                # ------------------------------------------------------------------------------------#
                # TODO 1 seg
                # ------------------------------------------------------------------------------------#
                segloss8 = Dice_loss(feat_out8, label) + nn.CrossEntropyLoss()(feat_out8, png)

                # pos = (feat_out8 * label).float()
                # neg = (feat_out8 * (1 - label)).float()

                # _segloss = nn.CosineSimilarity()(pos,label).mean() + 0.5 * nn.CosineSimilarity()(neg,label).mean()
                # _segloss = criteror(feat_out8,png.unsqueeze(dim=1))

                segloss16 = Dice_loss(feat_out16, label) + nn.CrossEntropyLoss()(feat_out16, png)
                segloss32 = Dice_loss(feat_out32, label) + nn.CrossEntropyLoss()(feat_out32, png)

                # --------------------------------------------------------------------------------------#
                # TODO add seg
                # --------------------------------------------------------------------------------------#
                # _feat_out_sp2 = up(feat_out_sp2)
                # _feat_out8 = feat_out8 * (1 - _feat_out_sp2) + feat_out8 + _feat_out_sp2
                # _seg_loss = nn.KLDivLoss()(_feat_out8,label)

                # Kl
                _feat = up(feat_out_sp4)
                _kl_s = nn.MSELoss()(up(feat_out_sp2), _feat)

                with torch.no_grad():
                    alpha = (feat_out8 - label).mean()
                    alpha = alpha ** 2

                    # alpha1 = (feat_out16 - label).mean()
                    # alpha1 = alpha1 ** 2

                    # alpha2 = (feat_out32 - label).mean()
                    # alpha2 = alpha2 ** 2

                segloss = (alpha * segloss8 + segloss16 + segloss32)
                # --------------------------------------------------------------------------#
                # TODO 2 boundary
                # --------------------------------------------------------------------------#
                # boundaryloss2 = nn.KLDivLoss()((feat_out_sp2),boundary2) + nn.MSELoss()((feat_out_sp2),boundary2)
                # feat_out_sp1x = up(feat_out_sp2)
                # pos = (feat_out_sp1x * boundary1).float()
                # neg = (feat_out_sp1x * (1 - boundary1)).float()
                # boundaryloss2 = nn.CosineSimilarity()(pos,boundary1).mean() + 0.5 * nn.CosineSimilarity()(neg,boundary1).mean()
                # boundaryloss2 = criteror(feat_out_sp2,boundary1)
                with torch.no_grad():  # 170 开始
                    bate = (feat_out_sp2 - boundary2).mean() ** 2
                boundaryloss2 = heatloss(feat_out_sp2, boundary2)
                boundaryloss4 = heatloss(feat_out_sp4, boundary4)
                boundaryloss8 = criteror(feat_out_sp8, boundary8)

                boundaryloss = (bate * boundaryloss2 + boundaryloss4 + boundaryloss8)

                # ------------------------------------------------------------------------------------#
                # TODO 3 loc
                # ------------------------------------------------------------------------------------#
                loc2loss = focal_loss(pred_loc2, heat2)
                loc4loss = focal_loss(pred_loc4, heat4)
                locloss = (loc2loss + 0.1 * loc4loss)

                # TODO KL
                # kl = nn.KLDivLoss()(l2,feat_out8)

                Tloss = (boundaryloss + segloss + _kl_s + locloss)  # 目前就只做一个边界

                # -------------------------------------------------------------------------#
                # TODO
                # -------------------------------------------------------------------------#
                total_cls_loss += (segloss).item()
                total_boundary_loss += (boundaryloss).item()
                total_heat_loss += (locloss).item()

                total_kl_loss += (_kl_s).item()

                total_loss += (total_heat_loss + total_cls_loss + \
                               total_boundary_loss + total_kl_loss)

            elif (args.method == 'bisenetv1') | (args.method == 'bisenetv2') | (args.method == 'unetplus') \
                    | (args.method == 'resunetplus')|(args.method == 'dconnNet'):

                output = model(image)

                ds = [ ]  # dice loss
                cs = [ ]  # ce loss

                for o in output:
                    ds.append(Dice_loss(o, label))
                    cs.append(nn.CrossEntropyLoss()(o, png))

                total_cls_loss += (sum(ds).item() + sum(cs).item())
                total_loss += total_cls_loss

                Tloss = sum(ds) + sum(cs)

            elif (args.method == 'ddrnet') | (args.method == 'ffnet') | (args.method == 'lpsnet'):

                output = model(image)

                output = up(output)

                loss1 = Dice_loss(output, label)
                loss2 = nn.CrossEntropyLoss()(output, png)

                total_cls_loss += (loss1 + loss2).item()

                total_loss += total_cls_loss

                Tloss = (loss1 + loss2)

            elif ((args.method) == 'unet') | ((args.method) == 'resunet') \
                    | (args.method == 'transunet') | (args.method == 'swin_unet') \
                    | ((args.method == 'setr')) | (args.method == 'missformer') | (args.method == 'ftmnet') | (
                    args.method == 'funet') | (args.method == 'protounet') | (args.method == 'gcn')|\
                    (args.method == 'deeplabv3')|(args.method == 'pspnet')|(args.method == 'daeformer')|\
                    (args.method == 'Hiformer')|(args.method == 'ahfunet'):

                if ((args.method) == 'unet_trans'):
                    output, pred_edge, pred_point, recon = model(image)

                    loss1 = Dice_loss(output, label)
                    loss2 = nn.CrossEntropyLoss()(output, png)

                    loss3 = nn.MSELoss()(points[ :, None, :, : ], pred_point)
                    loss4 = criteror(pred_edge, edges[ :, None, :, : ].float())

                    # -------------------------------------------------#
                    # 第一阶段: style module
                    # -------------------------------------------------#
                    recon = F.sigmoid(recon)
                    loss5 = mse_loss(recon, cdfs)
                    loss5.backward()

                    model = model.eval()

                    _, _, _, output = model(image)
                    B = orimgs.shape[ 0 ]
                    images = [ ]  # 重构之后的图像

                    def _map(imgs, pred):
                        orim_hist = cv2.calcHist([ imgs ], [ 0 ], None,
                                                 [ 256 ], [ 0, 256 ])

                        orim_cdf = np.cumsum(orim_hist) / np.sum(orim_hist)

                        mapping = np.interp(orim_cdf, pred, range(256)).astype(np.uint8)

                        mapping = mapping[ imgs ]
                        return mapping

                    for b in range(B):
                        # print(output.dtype,output.shape)
                        out = output.cpu().detach().numpy()[ b ]

                        _x1 = cv2.cvtColor(orimgs[ b ], cv2.COLOR_RGB2LAB)
                        l, a, b = _x1[ :, :, 0 ], _x1[ :, :, 1 ], _x1[ :, :, 2 ]

                        newl = _map(l, out[ :, 0 ])
                        newa = _map(a, out[ :, 1 ])
                        newb = _map(b, out[ :, 2 ])
                        im = cv2.merge([ newl, newa, newb ])
                        im = cv2.cvtColor(im, cv2.COLOR_LAB2BGR)
                        im = np.transpose(im, (2, 0, 1))
                        images.append(im)

                    # 设置为可导
                    images = np.array(images)
                    images = images / 255.

                    images = torch.from_numpy(images).float()
                    images.requires_grad = False
                    images = images.to(device)

                    model = model.train()
                    output1, pred_edge1, pred_point1, _ = model(images)

                    loss6 = Dice_loss(output1, label) + nn.CrossEntropyLoss()(output1, png)
                    loss7 = nn.MSELoss()(points[ :, None, :, : ], pred_point1)
                    loss8 = criteror(pred_edge1, edges[ :, None, :, : ].float())

                    Tloss = (loss1 + loss2 + loss3 + loss4 + loss6 + loss7 + loss8)

                    total_cls_loss += (loss1 + loss2 + loss3 + loss4 + loss6 + loss7 + loss8 + loss5).item()

                    total_loss += total_cls_loss


                elif (args.method == 'funet') | (args.method == 'ftmnet'):
                    output, pred_edge, pred_point = model(image)

                    loss1 = Dice_loss(output, label)
                    loss2 = nn.CrossEntropyLoss()(output, png)

                    # loss3 = nn.MSELoss()(F.sigmoid(pred_point),points)
                    loss4 = criteror(pred_edge, edges[ :, None, :, : ].float())

                    # print((loss1 + loss2).item(),loss4.item(), loss3.item())

                    Tloss = ((loss1 + loss2) * loss4)

                    total_cls_loss += (Tloss).item()

                    total_loss += total_cls_loss


                elif (args.method == 'ahfunet'):
                    output = model(image)
                    loss1 = nn.BCEWithLogitsLoss()(output,label)
                    #loss2 = edl_digamma_loss(output,label,1,2,10,device = device)
                    loss2 = Dice_loss(output,label)

                    Tloss = (loss1 + 0.5 * loss2 )

                else:
                    output = model(image)
                    # print(output.shape,label.shape)

                    loss1 = Dice_loss(output, label)
                    loss2 = nn.CrossEntropyLoss()(output, png)

                    Tloss = (loss1 + loss2)

                    total_cls_loss += (Tloss).item()

                    total_loss += total_cls_loss

            elif (args.method == 'tsformer'):

                cseg, tseg, *c = model(image)

                # 分割损失
                segloss1 = Dice_loss(cseg, label) + nn.CrossEntropyLoss()(cseg, png) + \
                           asym_unified_focal_loss(y_true = label, y_pred = cseg)  #
                segloss2 = Dice_loss(tseg, label) + nn.CrossEntropyLoss()(tseg, png) + \
                           asym_unified_focal_loss(y_true = label, y_pred = tseg)

                # decoder sanemtci alignment
                k1 = 2.5 * klloss(cseg, tseg)

                ss = [ ]
                # encoder feature alignment
                cnns = c[ 0 ]

                trans = c[ 1 ]

                # alpha = [0.2,0.15,0.1,0.1]
                for i in range(len(cnns)):
                    ss.append(klloss(cnns[ i ], trans[ i ]))

                total_kl_loss += sum(ss).item() + k1.item()

                total_cls_loss += (segloss1 + segloss2).item()

                total_loss += (total_cls_loss + total_cls_loss)

                Tloss = sum(ss) + (segloss1 + segloss2) + k1

            elif (args.method == 'fhnet'):

                cseg, pred_boundary, tseg, *c = model(image)

                # 分割损失
                segloss1 = Dice_loss(cseg, label) + nn.CrossEntropyLoss()(cseg, png)
                segloss2 = Dice_loss(tseg, label) + nn.CrossEntropyLoss()(tseg, png)

                # decoder sanemtci alignment
                k1 = 2.5 * klloss(cseg, tseg)

                # decoder pixel boundary
                boundary_loss = criteror(pred_boundary, boundary2)

                # boundary_loss = nn.MSELoss()(F.sigmoid(pred_boundary),boundary2)

                ss = [ ]
                # encoder feature alignment
                cnns = c[ 0 ]

                trans = c[ 1 ]

                alpha = [ 5.0, 2.5, 1.5, 1.0 ]
                for i in range(len(cnns)):
                    ss.append(alpha[ i ] * klloss(cnns[ i ], trans[ i ]))

                total_kl_loss += sum(ss).item() + k1.item()

                total_cls_loss += (segloss1 + segloss2).item()

                total_boundary_loss += (boundary_loss).item()

                total_loss += (total_boundary_loss + total_cls_loss + total_cls_loss)

                Tloss = sum(ss) + torch.exp(boundary_loss) + torch.exp(segloss1 + segloss2) + k1


            elif (args.method == 'stdc') | (args.method == 'pidnet'):
                ds = [ ]  # dice loss
                cs = [ ]  # ce loss

                *output, boundary = model(image)

                for o in output:
                    if args.method == 'pidnet':
                        o = up(o)
                    ds.append(Dice_loss(o, label))
                    cs.append(nn.CrossEntropyLoss()(o, png))

                if args.method == 'stdc':
                    _boundaryloss = criteror(boundary, boundary8)
                else:
                    boundary = up(boundary)
                    _boundaryloss = criteror(boundary, boundary1)
                # stdc_bundaryloss = criteror(boundary,boundary8)
                total_boundary_loss += _boundaryloss.item()
                total_cls_loss += (sum(ds).item() + sum(cs).item())
                total_loss += (total_cls_loss + total_boundary_loss)

                Tloss = sum(ds) + sum(cs) + _boundaryloss

            Tloss = Tloss / args.accumulate

            # print(total_loss)
            # print(total_cls_loss)
            # print(total_kl)
            Tloss.backward()
            if ((ind + 1) % args.accumulate) == 0:
                optimizer.step()

                optimizer.zero_grad()

            pb.set_postfix(**{
                'total_loss': total_loss / (ind + 1),
                'total_cls_loss': total_cls_loss / (ind + 1),
                'total_boundary_loss': total_boundary_loss / (ind + 1),
                'total_heat_loss': total_heat_loss / (ind + 1),
                'total_kl_loss': total_kl_loss / (ind + 1)
            })
            # print(f'=============Loss: {total_loss}=======================')

            #   logger.info(f'total_loss: {total_loss/(ind + 1)} total_cls_loss: {total_cls_loss/(ind + 1)}')
            #   logger.info(f'total_boundary_loss: {total_boundary_loss/(ind + 1)}')
            #   logger.info(f'total_heat_loss: {total_heat_loss/(ind + 1)}')

            pb.update(1)

    # ----------------------------------------------------------------------------#
    # TODO freeze BN
    # ----------------------------------------------------------------------------#
    model = model.eval()

    if (cur_epoch % save_step) == 0:
        # Evaluatation
        with tqdm(desc = f"Epoch: [{cur_epoch + 1}]/[{total_epoch}]",
                  total = len(testdataloader), mininterval = 0.3, postfix = dict, colour = '#7E89EF') as pb:

            with torch.no_grad():
                # TODO
                for ind, batch in enumerate(testdataloader):
                    _, image, label, (png, points, edges, _, _), (_, _), ((boundary1, boundary2), *boundaryss) = batch

                    image = image.to(device)
                    boundary2 = boundary2.to(device)
                    boundary2 = torch.randn_like(boundary2)
                    _b = boundaryss[ 0 ].to(device)
                    _b = torch.randn_like(_b)

                    png = png.to(device).detach().cpu().numpy()

                    # ------------------------------------------------------------------------------------#
                    h, w = args.input_shape[ 0 ] // 4, args.input_shape[ 1 ] // 4
                    _label = F.interpolate(label, size = (h, w), mode = 'bilinear', align_corners = True)

                    if args.method == 'Ours':
                        (feat_out8, feat_out16, feat_out32), \
                        (feat_out_sp2, feat_out_sp4, feat_out_sp8, feat_out_sp16), \
                        (pred_loc2, pred_loc4) = model(image, _label, *(boundary2, _b))

                        OneHotlabel8 = F.softmax(feat_out8, dim = 1)
                        # OneHotlabel16 = F.softmax(feat_out16, dim = 1)
                        # OneHotlabel32 = F.softmax(feat_out32, dim = 1)

                        targetOneHotlabel8 = torch.argmax((OneHotlabel8), dim = 1).detach().cpu().numpy()  # 目标域
                        # targetOneHotlabel16 = torch.argmax((OneHotlabel16), dim = 1).detach().cpu().numpy()  # 目标域
                        # targetOneHotlabel32 = torch.argmax((OneHotlabel32), dim = 1).detach().cpu().numpy()  # 目标域
                        targetlabel = targetOneHotlabel8
                        # targetlabel = (targetOneHotlabel8 | targetOneHotlabel16 | targetOneHotlabel32)

                    elif (args.method == 'ddrnet') | (args.method == 'ffnet') | (args.method == 'lpsnet'):
                        output = model(image)

                        pre = F.softmax(up(output), dim = 1)

                        targetOneHotlabel8 = torch.argmax(pre, dim = 1).detach().cpu().numpy()

                        targetlabel = targetOneHotlabel8
                    elif (args.method == 'bisenetv1') | (args.method == 'bisenetv2')|(args.method == 'dconnNet'):

                        output = model(image)

                        pre = F.softmax((output[ 1 ]), dim = 1)

                        targetOneHotlabel8 = torch.argmax(pre, dim = 1).detach().cpu().numpy()

                        targetlabel = targetOneHotlabel8

                    elif (args.method == 'stdc') | (args.method == 'pidnet'):

                        *out, _ = model(image)
                        pre = F.softmax(up(out[ 0 ]) if (args.method == 'pidnet') else out[ 0 ], dim = 1)

                        targetOneHotlabel8 = torch.argmax(pre, dim = 1).detach().cpu().numpy()

                        targetlabel = targetOneHotlabel8

                    elif ((args.method) == 'unet') | ((args.method) == 'resunet') \
                            | (args.method == 'transunet') | (args.method == 'swin_unet') \
                            | ((args.method == 'setr')) | (args.method == 'missformer') | (args.method == 'ftmnet') | (
                            (args.method == 'funet')) | (args.method == 'protounet') | (args.method == 'gcn')|\
                            (args.method == 'deeplabv3')|(args.method == 'pspnet')|(args.method == 'daeformer')|\
                            (args.method == 'Hiformer')|(args.method == 'ahfunet'):

                        if (args.method == 'ftmnet') | ((args.method == 'funet')):
                            output, *_ = model(image)
                            pre = F.softmax((output), dim = 1)

                            targetOneHotlabel8 = torch.argmax(pre, dim = 1).detach().cpu().numpy()

                            targetlabel = targetOneHotlabel8
                        else:
                            output = model(image)
                            pre = F.softmax((output), dim = 1)

                            targetOneHotlabel8 = torch.argmax(pre, dim = 1).detach().cpu().numpy()

                            targetlabel = targetOneHotlabel8

                    elif (args.method == 'tsformer'):

                        cseg, tseg, *c = model(image)

                        pre = F.softmax((cseg), dim = 1)

                        targetOneHotlabel8 = torch.argmax(pre, dim = 1).detach().cpu().numpy()

                        targetlabel = targetOneHotlabel8


                    elif (args.method == 'fhnet'):

                        cseg, boundray, tseg, *c = model(image)

                        # TODO? 3D密度图
                        pre = F.softmax((cseg), dim = 1)

                        targetOneHotlabel8 = torch.argmax(pre, dim = 1).detach().cpu().numpy()

                        targetlabel = targetOneHotlabel8
                    elif (args.method == 'unetplus') \
                            | (args.method == 'resunetplus'):

                        output = model(image)
                        pre = F.softmax((output[ 0 ]), dim = 1)

                        targetOneHotlabel8 = torch.argmax(pre, dim = 1).detach().cpu().numpy()

                        targetlabel = targetOneHotlabel8

                    eval.init(targetlabel, png)

                    pb.update(1)

                # 评估结果
                eval.show()

                res = eval.get_res()

                save = res[ 'update' ]

                # if save:
                #     torch.save(model.state_dict(), f'{save_path}/best.pth')

                # logger.info(f'Epoch: {cur_epoch}  res: {res}')

    if ((cur_epoch + 1) % save_step == 0):
        torch.save(model.state_dict(), f'{save_path}/{(cur_epoch + 1)}.pth')
    if (cur_epoch + 1) == total_epoch:
        torch.save(model.state_dict(), f'{save_path}/last.pth')
