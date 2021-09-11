"""Main program for PyTorch distributed training.

Adapted from: https://github.com/narumiruna/pytorch-distributed-example
"""

import argparse
import time
import datetime
import os
import sys
import logging
import socket
from retrying import retry
import torch
from torch import distributed
from torch import nn
import torch.nn.functional as F
from torch.utils import data
from torchvision import datasets
from torchvision import transforms


def distributed_is_initialized():
    return distributed.is_available() and distributed.is_initialized()


class Average(object):
    """Rolling average counter, updates during training epochs."""

    def __init__(self):
        self.sum = 0
        self.count = 0

    def __str__(self):
        return '{:.6f}'.format(self.average)

    @property
    def average(self):
        return self.sum / self.count

    def update(self, value, number):
        self.sum += value * number
        self.count += number


class Accuracy(object):
    """Rolling accuracy counter, updates during training epochs."""

    def __init__(self):
        self.correct = 0
        self.count = 0

    def __str__(self):
        return '{:.2f}%'.format(self.accuracy * 100)

    @property
    def accuracy(self):
        return self.correct / self.count

    @torch.no_grad()
    def update(self, output, target):
        pred = output.argmax(dim=1)
        correct = pred.eq(target).sum().item()

        self.correct += correct
        self.count += output.size(0)


class Trainer(object):
    """Trainer class to encapsulate the entire train-and-evaluate loop."""

    def __init__(self, model, optimizer, train_loader, test_loader, device):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def fit(self, epochs):
        """Iterate for desired number of epochs in train-and-evaluate loop."""
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train()
            test_loss, test_acc = self.evaluate()

            print(
                'Epoch: {}/{},'.format(epoch, epochs),
                'train loss: {}, train acc: {},'.format(train_loss, train_acc),
                'test loss: {}, test acc: {}.'.format(test_loss, test_acc),
            )

    def train(self):
        """Train the model for a single epoch."""
        # Sets the model to run in 'training' mode.
        self.model.train()

        train_loss = Average()
        train_acc = Accuracy()

        # Iterate through training dataset.
        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward propagation
            output = self.model(inputs)
            loss = F.cross_entropy(output, targets)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            train_loss.update(loss.item(), inputs.size(0))
            train_acc.update(output, targets)

        return train_loss, train_acc

    @torch.no_grad()
    def evaluate(self):
        """Evaluate the model on test set."""
        # Sets the model to run in 'evaluation' mode.
        self.model.eval()

        test_loss = Average()
        test_acc = Accuracy()

        # Iterate through test dataset.
        for inputs, targets in self.test_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward propagation
            output = self.model(inputs)
            loss = F.cross_entropy(output, targets)

            # Update metrics
            test_loss.update(loss.item(), inputs.size(0))
            test_acc.update(output, targets)

        return test_loss, test_acc


# Simple network consisting of a single fully connected layer (a linear model).
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


class MNISTDataLoader(data.DataLoader):
    """Data loader encapsulation for the MNIST dataset.

    Contains data loading, transformation, and sampling steps.
    """

    def __init__(self, root, batch_size, train=True):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        dataset = datasets.MNIST(
            root, train=train, transform=transform, download=True)

        sampler = None
        if train and distributed_is_initialized():
            sampler = data.DistributedSampler(dataset)

        super(MNISTDataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
        )


def run(args):
    """Set up and run the training loop."""
    # Run on GPU if possible
    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Net()
    if distributed_is_initialized():
        model.to(device)
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank)
    else:
        model = nn.DataParallel(model)
        model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    root = os.path.join(args.root, str(args.local_rank))

    train_loader = MNISTDataLoader(root, args.batch_size, train=True)
    test_loader = MNISTDataLoader(root, args.batch_size, train=False)

    trainer = Trainer(model, optimizer, train_loader, test_loader, device)
    trainer.fit(args.epochs)


def make_torch_tcp_store(master_addr, master_port, world_size, start_daemon):
    return distributed.TCPStore(master_addr, master_port, world_size, start_daemon)


def barrier(make_tcp_store):
    """Blocks until master and all workers have joined.

    Args:
      make_tcp_store: Cosntructs a TCP store.

    Raises:
      EnvironmentError: when required environment variables are missing.
    """
    world_size = os.environ.get('WORLD_SIZE', None)
    # if this is not a distributed PyTorch run, don't block on barrier.
    if world_size is None:
        return

    world_size = int(world_size)
    if world_size == 1:
        return

    rank = os.environ.get('RANK', None)
    if rank is None:
        raise EnvironmentError('missing RANK environment variable')
    rank = int(rank)

    master_addr = os.environ.get('MASTER_ADDR', None)
    if master_addr is None:
        raise EnvironmentError('missing MASTER_ADDR environment variable')

    master_port = os.environ.get('MASTER_PORT', None)
    if master_port is None:
        raise EnvironmentError('missing MASTER_PORT environment variable')
    master_port = int(master_port)

    logging.info('Starting rank %s', rank)
    logging.info('My master is %s:%s', master_addr, master_port)
    logging.info('World size = %s', world_size)

    barrier_with_retry(world_size, rank, master_addr,
                       master_port, make_tcp_store)


@retry(stop_max_attempt_number=4, wait_exponential_multiplier=1000)
def barrier_with_retry(
        world_size, rank, master_addr, master_port, make_tcp_store):
    """Blocks until master and all workers have joined with multiple retries.

    Args:
      world_size: Total number of nodes. Including master.
      rank: My rank in the world.
      master_addr: Master node network address.
      master_port: Master node listening port.
      make_tcp_store: constructor for TCP store object.
    """
    # Only start the server daemon on rank 0 (master)
    start_daemon = rank == 0
    store = None
    try:
        # This could take up to 2 minutes before timing out. If it does, we retry.
        logging.info('%d# Connecting to barrier...', rank)
        store = make_tcp_store(master_addr, master_port,
                               world_size, start_daemon)
        logging.info('%d# Barrier connected successuflly.', rank)
        logging.info('%d# Waiting for all processes to join.', rank)
        for i in range(world_size):
            if rank == i:
                store.set(str(rank), '')
            else:
                store.wait([str(i)], datetime.timedelta(minutes=5))
        logging.info('%d# All barrier processes joined successuflly.', rank)
        time.sleep(5)
        del store
    except Exception as e:  # pylint: disable=broad-except
        logging.info(
            'Rank#(%d) Error: %s. Backing off then retrying ...', rank, e)
        if store is not None:
            del store
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--backend', type=str, default='nccl', help='Name of the backend to use.')
    parser.add_argument(
        '-i',
        '--init-method',
        type=str,
        default='env://',
        help='URL specifying how to initialize the package.')
    parser.add_argument(
        '-r',
        '--local_rank',
        type=int,
        default=os.environ.get('LOCAL_RANK', 0),
        help='Local rank of the current process.')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    print(args)
    print('\n'.join([f'{k}={v}' for k, v in os.environ.items()]))

    myrank = os.environ.get('RANK', None)
    myname = socket.gethostname()
    if myrank == '0':
        logging.info('Updating master address to local address %s', myname)
        os.environ['MASTER_ADDR'] = myname
    barrier(make_torch_tcp_store)

    if myrank != '0':
      time.sleep(5)

    print('Initializing distributed backend.')
    distributed.init_process_group(
        backend=args.backend,
        init_method=args.init_method,
    )
    run(args)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s',
                        stream=sys.stdout,
                        level=logging.INFO)
    main()
