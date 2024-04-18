import torch
from time import time
# from tensorboardX import SummaryWriter
import numpy as np
import os
import sys
from tqdm import tqdm

sys.path.append('./')

import config.kitti_config as cfg
from config.train_config import parse_train_configs
from models.lss import compile_model, BevEncode
from utils.lss import SimpleLoss, get_batch_iou, get_val_info
from data_process.kitti_dataloader import create_train_dataloader, create_val_dataloader


def train(nepochs=10000,
          gpuid=0,

          H=900, W=1600,
          resize_lim=(0.193, 0.225),
          final_dim=(128, 352),
          bot_pct_lim=(0.0, 0.22),
          rot_lim=(-5.4, 5.4),
          rand_flip=True,
          ncams=5,
          max_grad_norm=5.0,
          pos_weight=2.13,
          logdir='./runs',

          xbound=[cfg.boundary["minX"], cfg.boundary["maxX"], cfg.DISCRETIZATION],
          ybound=[cfg.boundary["minY"], cfg.boundary["maxY"], cfg.DISCRETIZATION],
          zbound=[cfg.boundary["minZ"], cfg.boundary["maxZ"], 4.0],
          dbound=[4.0, 45.0, 1.0],  # ?

          bsz=4,
          nworkers=4,
          lr=1e-3,
          weight_decay=1e-7,
          ):
    configs = parse_train_configs()
    configs.distributed = False
    gpuid = configs.gpu_idx

    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
        # 'resize_lim': resize_lim,
        'final_dim': final_dim,
        # 'rot_lim': rot_lim,
        # 'H': H, 'W': W,
        # 'rand_flip': rand_flip,
        # 'bot_pct_lim': bot_pct_lim,
        # 'cams': ['P2', 'P3'],
        # 'Ncams': 2,
    }
    trainloader, _ = create_train_dataloader(configs)
    valloader = create_val_dataloader(configs)

    device = torch.device(
        'cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    # model = compile_model(grid_conf, data_aug_conf, outC=3)
    model = BevEncode(3, 3)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

    loss_fn = SimpleLoss(pos_weight).cuda(gpuid)

    # writer = SummaryWriter(logdir=logdir)
    # val_step = 1000 if version == 'mini' else 10000
    val_step = 10000

    model.train()
    counter = 0
    for epoch in range(nepochs):
        np.random.seed()
        for batchi, batch_data in enumerate(tqdm(trainloader)):
            cams, _, _ = batch_data
            # imgs, rots, trans, intrins, post_rots, post_trans, binimgs = cams
            imgs, binimgs = cams
            t0 = time()
            opt.zero_grad()
            # preds = model(imgs.to(device),
            #               rots.to(device),
            #               trans.to(device),
            #               intrins.to(device),
            #               post_rots.to(device),
            #               post_trans.to(device),
            #               )
            preds = model(imgs.to(device))
            binimgs = binimgs.to(device)
            loss = loss_fn(preds, binimgs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            counter += 1
            t1 = time()

            if counter % 10 == 0:
                print(counter, loss.item())

            if counter % 50 == 0:
                _, _, iou = get_batch_iou(preds, binimgs)

            if counter % val_step == 0:
                val_info = get_val_info(model, valloader, loss_fn, device)
                print('VAL', val_info)

            if counter % val_step == 0:
                model.eval()
                mname = os.path.join(logdir, "model{}.pt".format(counter))
                print('saving', mname)
                torch.save(model.state_dict(), mname)
                model.train()


if __name__ == '__main__':
    train()
