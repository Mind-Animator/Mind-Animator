import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from torch.optim import lr_scheduler
from Mind_Animator_Models.modules import Semantic_Decoder
from Mind_Animator_Models.Dataset import Train_Semantic_dataset
from Mind_Animator_Models.loss_function import Contrastive_loss, Semantic_SimpleLossCompute

scaler = StandardScaler()
device = torch.device('cuda:2')


def make_model(in_dim=4500, out_dim=19 * 768, h=512, n_blocks=3, norm_type='ln', act_first=False):
    model = Semantic_Decoder(in_dim=in_dim, out_dim=out_dim, h=h, n_blocks=n_blocks, norm_type=norm_type,
                         act_first=act_first)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


def run_epoch(mode, data_iter, model, loss_compute, norm=1, epoch=None, opt=None, scheduler=None):
    train_loss = []
    total_loss = 0
    for i, batch in tqdm(enumerate(data_iter)):
        loss_reg = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                loss_reg += torch.sum(torch.abs(param))

        trn_src, trn_proj, trn_tgt_CLIP_text, trn_tgt_CLIP_picture = batch
        x, y, logits_per_fMRI_t, logits_per_fMRI_i = model.forward(x=trn_src.squeeze(1),
                                                                   y_CLIP_t=trn_tgt_CLIP_text.squeeze(1),
                                                                   y_CLIP_i=trn_tgt_CLIP_picture.squeeze(1))
        loss = loss_compute(mode=mode, epoch=epoch, y_pred_mid=x, y_CLIP_t=trn_tgt_CLIP_text.squeeze(1), y_pred=y,
                            y_tgt=trn_proj.squeeze(1),
                            logits_per_fMRI_t=logits_per_fMRI_t, logits_per_fMRI_i=logits_per_fMRI_i, norm=norm)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.data.item()
        train_loss.append(loss.data.item())
        if i % 10 == 0:
            print('batch:{}，total_loss:{}'.format(i, loss.data.item()))

    if scheduler is not None:
        scheduler.step()
        print('lr:{}'.format(opt.state_dict()['param_groups'][0]['lr']))
    return train_loss


def main(args, f_MRI_data, image_data):
    criterion1 = Contrastive_loss()
    criterion2 = nn.MSELoss()
    model = make_model(in_dim=4500, out_dim=19 * 768, h=512, n_blocks=args.n_blocks, norm_type='ln',act_first=False).to(device)

    model_opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(model_opt, gamma=0.999)

    Train_loss = []

    for epoch in tqdm(range(args.epoch + 1)):
        trainset = Train_Semantic_dataset(f_MRI_data, image_data, args.train_caption_root)

        model.train()
        train_data = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        train_loss, _ = run_epoch(mode='train', data_iter=train_data, model=model,
                                     loss_compute=Semantic_SimpleLossCompute(criterion1=criterion1, criterion2=criterion2,
                                                                   k1=args.k1, k2=args.k2 ),
                                      norm=1, epoch=epoch,opt=model_opt, scheduler=scheduler)
        Train_loss.append(np.mean(train_loss))
        if epoch in [20 ,30,40,50, 60,70, 80, 100]:
            torch.save(model.state_dict(), os.path.join(args.model_dir,'/Sub_{}/'.format(args.subj_ID) , 'Semantic_Decoder_{}.pth'.format(epoch)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Semantic decoder')
    parser.add_argument('--model_dir', help='model saved path', default=None, type=str)
    parser.add_argument('--fMRI_data_path', help='fMRI data saved path', default=None, type=str)
    parser.add_argument('--image_data_path', help='image_data saved path', default=None, type=str)
    parser.add_argument('--train_caption_root', help='train_caption_root', default=None, type=str)
    parser.add_argument('--batch_size', help='batch size of dnn training', default=64, type=int)
    parser.add_argument('--epoch', help='epoch', default=100, type=int)
    parser.add_argument('--subj_ID', help='subj_ID', default=1, type=int)
    parser.add_argument('--n_blocks', help='n_mlp_blocks', default=3, type=float)
    parser.add_argument('--k1', help='ratio of loss_CLIP', default=0.01, type=float)
    parser.add_argument('--k2', help='ratio of loss_raw', default=0.5, type=float)
    parser.add_argument('--lr', help='learning rate', default=0.0002, type=float)
    parser.add_argument('--warm_up', help='warm_up', default=0, type=float)
    args = parser.parse_args()
    image_data = torch.tensor(np.load(args.image_data_path + '/Train_images.npy'), dtype=torch.float32).to(device)
    fMRI_data = np.load(args.fMRI_data_path+ '/Sub_{}/'.format(args.subj_ID) +  '/masked4500_trn_data.npy')


    main(args, fMRI_data, image_data)
