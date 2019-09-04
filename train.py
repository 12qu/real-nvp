"""Train Real NVP on CIFAR-10.

Train script adapted from: https://github.com/kuangliu/pytorch-cifar/
"""
import argparse
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import util

from models import RealNVP, RealNVPLoss
from tqdm import tqdm

from oos.models import get_conv_realnvp_density
from oos.densities import BijectionDensity
from oos.bijections import LogitTransformBijection

from tensorboardX import SummaryWriter

use_ours = False
print(("U" if use_ours else "NOT u") + "sing our model")
channels = None
height = None
width = None
writer = SummaryWriter(comment=f"_{'our-model' if use_ours else 'baseline'}")

def main(args):
    device = 'cuda' if torch.cuda.is_available() and len(args.gpu_ids) > 0 else 'cpu'
    start_epoch = 0

    # Note: No normalization applied, since RealNVP expects inputs in (0, 1).
    transform_train = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    # trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
    trainset = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform_train)
    x_train = trainset.data.to(torch.get_default_dtype()).view(-1, 1, 28, 28)
    x_train += torch.rand_like(x_train)
    y_train = trainset.targets
    trainset = data.TensorDataset(x_train, y_train)
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
    testset = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transform_test)
    x_test = testset.data.to(torch.get_default_dtype()).view(-1, 1, 28, 28)
    x_test += torch.rand_like(x_test)
    y_test = testset.targets
    testset = data.TensorDataset(x_test, y_test)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    global channels, height, width
    channels, height, width = trainset[0][0].shape

    # Model
    print('Building model..')

    if use_ours:
        net = get_conv_realnvp_density(1, (1, 28, 28))
        net = BijectionDensity(
            prior=net,
            bijection=LogitTransformBijection(
                input_shape=(1, 28, 28),
                lam=1e-6
            )
        )
    else:
        net = RealNVP(num_scales=1, in_channels=channels, mid_channels=64, num_blocks=8)

    net = net.to(device)
    # if device == 'cuda':
    #     net = torch.nn.DataParallel(net, args.gpu_ids)
    #     cudnn.benchmark = args.benchmark

    if args.resume:
        # Load checkpoint.
        print('Resuming from checkpoint at ckpts/best.pth.tar...')
        assert os.path.isdir('ckpts'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('ckpts/best.pth.tar')
        net.load_state_dict(checkpoint['net'])
        global best_loss
        best_loss = checkpoint['test_loss']
        start_epoch = checkpoint['epoch']

    loss_fn = RealNVPLoss()
    param_groups = util.get_param_groups(net, args.weight_decay, norm_suffix='weight_g')
    optimizer = optim.Adam(param_groups, lr=args.lr)

    test(-1, net, testloader, device, loss_fn, args.num_samples)
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        train(epoch, net, trainloader, device, optimizer, loss_fn, args.max_grad_norm)
        test(epoch, net, testloader, device, loss_fn, args.num_samples)


step = 0
def train(epoch, net, trainloader, device, optimizer, loss_fn, max_grad_norm):
    global step
    print('\nEpoch: %d' % epoch)
    net.train()
    loss_meter = util.AverageMeter()
    with tqdm(total=len(trainloader.dataset)) as progress_bar:
        for x, _ in trainloader:
            x = x.to(device)
            optimizer.zero_grad()

            if use_ours:
                result = net.elbo(x)
                loss = -result["elbo"].mean()
            else:
                z, sldj = net(x, reverse=False)
                loss = loss_fn(z, sldj)

            loss_meter.update(loss.item(), x.size(0))
            loss.backward()
            util.clip_grad_norm(optimizer, max_grad_norm)
            optimizer.step()


            writer.add_scalar("mnist/train-loss", loss.item(), global_step=step)
            step += 1

            progress_bar.set_postfix(loss=loss_meter.avg,
                                     bpd=util.bits_per_dim(x, loss_meter.avg))
            progress_bar.update(x.size(0))


def sample(net, batch_size, device):
    """Sample from RealNVP model.

    Args:
        net (torch.nn.DataParallel): The RealNVP model wrapped in DataParallel.
        batch_size (int): Number of samples to generate.
        device (torch.device): Device to use.
    """
    z = torch.randn((batch_size, channels, height, width), dtype=torch.float32, device=device)
    x, _ = net(z, reverse=True)
    x = torch.sigmoid(x)

    return x


def test(epoch, net, testloader, device, loss_fn, num_samples):
    global best_loss
    net.eval()
    loss_meter = util.AverageMeter()
    with torch.no_grad():
        with tqdm(total=len(testloader.dataset)) as progress_bar:
            for x, _ in testloader:
                x = x.to(device)
                if use_ours:
                    result = net.elbo(x)
                    loss = -result["elbo"].mean()
                else:
                    z, sldj = net(x, reverse=False)
                    loss = loss_fn(z, sldj)
                loss_meter.update(loss.item(), x.size(0))
                progress_bar.set_postfix(loss=loss_meter.avg,
                                         bpd=util.bits_per_dim(x, loss_meter.avg))
                progress_bar.update(x.size(0))

    # Save checkpoint
    if loss_meter.avg < best_loss:
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'test_loss': loss_meter.avg,
            'epoch': epoch,
        }
        os.makedirs('ckpts', exist_ok=True)
        torch.save(state, 'ckpts/best.pth.tar')
        best_loss = loss_meter.avg

    writer.add_scalar("mnist/log-prob/mnist", -loss_meter.avg, global_step=epoch)
    writer.add_scalar("mnist/bpd/mnist", util.bits_per_dim(x, loss_meter.avg),
            global_step=epoch)

    # Save samples and data
    if use_ours:
        images = net.sample(num_samples) / 256
    else:
        images = sample(net, num_samples, device)
    os.makedirs('samples', exist_ok=True)
    images_concat = torchvision.utils.make_grid(images, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
    torchvision.utils.save_image(images_concat, 'samples/epoch_{}.png'.format(epoch))

    writer.add_image("mnist/samples", images_concat / 256, global_step=epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RealNVP on CIFAR-10')

    parser.add_argument('--batch_size', default=100, type=int, help='Batch size')
    parser.add_argument('--benchmark', action='store_true', help='Turn on CUDNN benchmarking')
    parser.add_argument('--gpu_ids', default='[0]', type=eval, help='IDs of GPUs to use')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=100., help='Max gradient norm for clipping')
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs to train')
    parser.add_argument('--num_samples', default=64, type=int, help='Number of samples at test time')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader threads')
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--weight_decay', default=5e-5, type=float,
                        help='L2 regularization (only applied to the weight norm scale factors)')

    best_loss = 0

    main(parser.parse_args())
