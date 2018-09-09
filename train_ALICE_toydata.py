"""https://github.com/ChunyuanLI/ALICE/blob/master/toy_data/ALICE_A.py
Toy dataset.
"""
import argparse
import datetime
import json
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')  # NOQA
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import tqdm

sys.path.append('.')  # NOQA
import gmm as gmm_module
from networks import Generator
from networks import Inference
from networks import Discriminator
from networks import DiscriminatorXX
from networks import DiscriminatorZZ
from utils import forward


def main():
    parser = argparse.ArgumentParser()
    # Training
    parser.add_argument('--adv', default=False, action='store_true')
    parser.add_argument('--easy', default=False, action='store_true')
    parser.add_argument('--num_epochs', '-N', type=int, default=100)
    parser.add_argument('--batch_size', '-B', type=int, default=128)
    parser.add_argument('--gpu_id', '-G', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=820)
    # Log
    parser.add_argument('--results_dir', '-R', type=str, default='results')
    # Dataset -- X
    parser.add_argument('--X_std', type=float, default=0.04)
    parser.add_argument('--X_trn_sz', type=int, default=512 * 4)
    parser.add_argument('--X_val_sz', type=int, default=512 * 2)
    # Dataset -- Z
    parser.add_argument('--Z_std', type=float, default=1.0)
    parser.add_argument('--Z_trn_sz', type=int, default=512 * 4)
    parser.add_argument('--Z_val_sz', type=int, default=512 * 2)
    # Optimizer
    parser.add_argument('--n_gen', type=int, default=5)
    parser.add_argument('--n_dis', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=1e-3)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    # Models
    parser.add_argument('--in_features', '-in', type=int, default=2)
    parser.add_argument('--latent_features', '-lf', type=int, default=2)
    parser.add_argument('--noise_features', '-nf', type=int, default=2)
    parser.add_argument('--gen_num_layers', '-gnl', type=int, default=2)
    parser.add_argument('--gen_hidden_features', '-ghf', type=int, default=256)
    parser.add_argument('--gen_out_features', '-gof', default=None)
    parser.add_argument('--inf_num_layers', '-inl', type=int, default=2)
    parser.add_argument('--inf_hidden_features', '-ihf', type=int, default=256)
    parser.add_argument('--inf_out_features', '-iof', default=None)
    parser.add_argument('--dis_num_layers', '-dnl', type=int, default=2)
    parser.add_argument('--dis_hidden_features', '-dhf', type=int, default=256)
    parser.add_argument('--dis_out_features', '-dof', type=int, default=1)
    args = parser.parse_args()
    if args.gen_out_features is None:
        args.gen_out_features = args.in_features
    if args.inf_out_features is None:
        args.inf_out_features = args.latent_features

    if args.adv:
        _name = "_ALICE_toydata_unsupervised_adversarial_reconstruction"
    else:
        _name = "_ALICE_toydata_unsupervised_MSE_reconstruction"
    args.results_dir = os.path.join(
        args.results_dir,
        datetime.datetime.now().strftime('%y%m%d-%H%M%S') + _name
    )
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)
    with open(os.path.join(args.results_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    if args.gpu_id > -1:
        if torch.cuda.is_available():
            torch.cuda.set_device(args.gpu_id)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        'cuda' if args.gpu_id > -1 and torch.cuda.is_available() else 'cpu'
    )

    # Prepare dataset X
    rt2 = math.sqrt(2)
    means = {
        'easy': [[0, 0], [5, 5], [-5, 5], [-5, -5], [5, -5]],
        'difficult': [
            [5 * rt2, 0], [5, 5], [0, 5 * rt2], [-5, 5],
            [-5 * rt2, 0], [-5, -5], [0, -5 * rt2], [5, -5]]
    }
    key = 'easy' if args.easy else 'difficult'
    means_x = list(map(
        lambda x: torch.tensor(x, dtype=torch.float), means[key]
    ))
    variances_x = [torch.eye(2) * args.X_std for _ in means_x]
    x_trn = gmm_module.GMMData(
        args.X_trn_sz, means_x, variances_x, seed=args.seed)
    x_trn_loader = torch.utils.data.DataLoader(
        x_trn, args.batch_size,
        pin_memory=(args.gpu_id > -1 and torch.cuda.is_available())
    )
    # Prepare dataset Z
    means_z = list(map(
        lambda x: torch.tensor(x, dtype=torch.float),
        [[0, 0]]
    ))
    variances_z = [torch.eye(2) * args.Z_std for _ in means_z]
    z_trn = gmm_module.GMMData(
        args.Z_trn_sz, means_z, variances_z, seed=args.seed)
    z_trn_loader = torch.utils.data.DataLoader(
        z_trn, args.batch_size,
        pin_memory=(args.gpu_id > -1 and torch.cuda.is_available())
    )
    # Prepare models
    gen = Generator(
        args.gen_num_layers, args.in_features, args.noise_features,
        args.gen_hidden_features, args.gen_out_features).to(device)
    inf = Inference(
        args.gen_num_layers, args.in_features, args.noise_features,
        args.inf_hidden_features, args.inf_out_features).to(device)
    dis = Discriminator(
        args.gen_num_layers, args.in_features, args.noise_features,
        args.dis_hidden_features, args.dis_out_features).to(device)
    if args.adv:
        dis_x = DiscriminatorXX(
            args.gen_num_layers, args.in_features, args.noise_features,
            args.dis_hidden_features, args.dis_out_features).to(device)
        dis_z = DiscriminatorZZ(
            args.gen_num_layers, args.in_features, args.noise_features,
            args.dis_hidden_features, args.dis_out_features).to(device)
    else:
        dis_x, dis_z = None, None
    opt_gen_inf = torch.optim.Adam(
        list(gen.parameters()) + list(inf.parameters()),
        args.alpha, (args.beta1, args.beta2)
    )
    _params = list(dis.parameters())
    if args.adv:
        _params += list(dis_x.parameters()) + list(dis_z.parameters())
    opt_dis = torch.optim.Adam(_params, args.alpha, (args.beta1, args.beta2))

    # Save figures
    x_gmm_samples = x_trn.samples.numpy()
    z_gmm_samples = z_trn.samples.numpy()
    figure, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
    ax.scatter(x_gmm_samples[:, 0], x_gmm_samples[:, 1],
               label='X', marker='.', alpha=0.3,
               c=matplotlib.cm.Set1(x_trn.labels.numpy().reshape((-1,))/args.in_features/2.0))
    ax.scatter(z_gmm_samples[:, 0], z_gmm_samples[:, 1],
               label='Z', marker='.', alpha=0.1)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    plt.legend()
    plt.savefig(os.path.join(args.results_dir, 'dataset.png'))
    plt.close('all')
    torch.save(x_trn, os.path.join(args.results_dir, 'x_trn.pkl'))
    torch.save(z_trn, os.path.join(args.results_dir, 'z_trn.pkl'))

    for epoch in tqdm.tqdm(range(args.num_epochs)):
        _mom, epoch_dis_loss, epoch_gen_loss = .9, 0, 0
        for i, (x_batch, z_batch) in tqdm.tqdm(enumerate(zip(x_trn_loader, z_trn_loader))):
            x, *_ = x_batch
            z, *_ = z_batch
            x, z = x.to(device), z.to(device)

            iter_dis_loss = .0
            for j in range(args.n_dis):
                dis_loss_opt, _ = forward(
                    device, args.adv, x, z, gen, inf, dis,
                    dis_x, dis_z, False
                )
                opt_dis.zero_grad()
                dis_loss_opt.backward()
                opt_dis.step()
                iter_dis_loss += dis_loss_opt.item() / args.n_dis
            epoch_dis_loss = epoch_dis_loss * (1 - _mom) + iter_dis_loss * _mom

            iter_gen_loss, iter_x, iter_z = .0, .0, .0
            for j in range(args.n_gen):
                _, gen_loss_opt, cost_x, cost_z = forward(
                    device, args.adv, x, z, gen, inf, dis,
                    dis_x, dis_z, True
                )
                opt_gen_inf.zero_grad()
                gen_loss_opt.backward()
                opt_gen_inf.step()
                iter_gen_loss += gen_loss_opt.item() / args.n_gen
                iter_x += cost_x.item() / args.n_gen
                iter_z += cost_z.item() / args.n_gen
            epoch_gen_loss = epoch_gen_loss * (1 - _mom) + iter_gen_loss * _mom

            if (i + 1) % 8 == 0:
                _fmt = "epoch {}/{}, iter {}, dis: {:.05f}, gen: {:.05f}, x: {:.05f}, z: {:.05f}"
                tqdm.tqdm.write(
                    _fmt.format(epoch, args.num_epochs, i + 1, iter_dis_loss, iter_gen_loss, iter_x, iter_z))

        if (epoch + 1) % 10 == 0:
            gen.eval()
            inf.eval()
            x_trn_samples = x_trn.samples.to(device)
            z_trn_samples = z_trn.samples.to(device)
            with torch.no_grad():
                p_x = gen(z_trn_samples)
                q_z = inf(x_trn_samples)
                x_rec = gen(q_z).cpu().numpy()
                z_rec = inf(p_x).cpu().numpy()
            tqdm.tqdm.write('Epoch {}/{}, z_rec| mean: {}, var: {}'.format(
                epoch + 1, args.num_epochs,
                z_rec.mean(axis=0), z_rec.var(axis=0, ddof=1)))
            if (epoch + 1) % 10 == 0:
                figure, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
                ax.scatter(x_rec[:, 0], x_rec[:, 1], label='X_reconstructed',
                           marker='.', alpha=0.3)
                ax.scatter(z_rec[:, 0], z_rec[:, 1], label='Z_reconstructed',
                           marker='.', alpha=0.1)
                ax.set_xlim(-10, 10)
                ax.set_ylim(-10, 10)
                plt.legend()
                plt.savefig(
                    os.path.join(args.results_dir,
                                 'reconstructed_{}.png'.format(epoch + 1)))
                plt.close('all')
    _ckpt = {
        'opt_gen_inf': opt_gen_inf.state_dict(),
        'opt_dis': opt_dis.state_dict(),
        'gen': gen.state_dict(), 'inf': inf.state_dict(),
        'dis': dis.state_dict(),
    }
    if args.adv:
        _ckpt.update({
            'dis_x': dis_x.state_dict(),
            'dis_z': dis_z.state_dict()})
    torch.save(_ckpt, os.path.join(args.results_dir, 'ckpt.pth.tar'))


if __name__ == '__main__':
    main()
