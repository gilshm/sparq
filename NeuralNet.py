import torch
import time
import os
import Config as cfg
from StatsLogger import StatsLogger


class NeuralNet:
    def __init__(self, arch, dataset, model_chkp=None, pretrained=True):
        """
        NeuralNet class wraps a model architecture and adds the functionality of training and testing both
        the entire model and the prediction layers.

        :param arch: a string that represents the model architecture, e.g., 'alexnet'
        :param dataset: a string that represents the dataset, e.g., 'cifar100'
        :param model_chkp: a model checkpoint path to be loaded (default: None)
        :param pretrained: whether to load PyTorch pretrained parameters, used for ImageNet (default: True)
        """
        cfg.LOG.write('__init__: arch={}, dataset={}, model_chkp={}, pretrained={}'
                      .format(arch, dataset, model_chkp, pretrained))

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            cfg.LOG.write('WARNING: Found no valid GPU device - Running on CPU')

        self.arch = '{}_{}'.format(arch, dataset)

        # The 'pretrained' argument is used for PyTorch pretrained ImageNet models, whereas the model_chkp is
        # intended to load user checkpoints. At this point model_chkp and pretrained are not supported together,
        # although it is possible.
        if model_chkp is None:
            self.model = cfg.MODELS[self.arch](pretrained=pretrained)
        else:
            self.model = cfg.MODELS[self.arch]()

        # self.parallel_model = torch.nn.DataParallel(self.model).cuda(self.device)
        # Disabled parallel model. Statistics collection does not support GPU parallelism.
        self.parallel_model = self.model.cuda(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().cuda(self.device)

        self.optimizer = None
        self.lr_plan = None
        self.best_top1_acc = 0
        self.next_train_epoch = 0

        if model_chkp is not None:
            self._load_state(model_chkp)

        self.stats = StatsLogger()

    def test(self, test_gen, iterations=None):
        batch_time = self.AverageMeter('Time', ':6.3f')
        losses = self.AverageMeter('Loss', ':.4e')
        top1 = self.AverageMeter('Acc@1', ':6.2f')
        top5 = self.AverageMeter('Acc@5', ':6.2f')
        progress = self.ProgressMeter(len(test_gen), batch_time, losses, top1, top5, prefix='Test: ')

        self.model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(test_gen):
                input = input.cuda(self.device, non_blocking=True)
                target = target.cuda(self.device, non_blocking=True)

                # Compute output
                output = self.parallel_model(input)
                loss = self.criterion(output, target)

                # Measure accuracy and record loss
                acc1, acc5 = self._accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(acc1[0], input.size(0))
                top5.update(acc5[0], input.size(0))

                # Measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # Print to screen
                if i % 10 == 0:
                    progress.print(i)

                if iterations is not None:
                    if i == iterations:
                        break

            # TODO: this should also be done with the ProgressMeter
            cfg.LOG.write(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

        return top1.avg

    def train(self, train_gen, test_gen, epochs, lr=0.0001, lr_plan=None, momentum=0.9, wd=5e-4,
              desc=None, iterations=None):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
        self.lr_plan = lr_plan

        cfg.LOG.write('train_pred: epochs={}, lr={}, lr_plan={}, momentum={}, wd={}, batch_size={}, optimizer={}'
                      .format(epochs, lr, lr_plan, momentum, wd, cfg.BATCH_SIZE, 'SGD'))

        for epoch in range(self.next_train_epoch, epochs):
            self._adjust_lr_rate(self.optimizer, epoch, lr_plan)

            # A precautions that when the learning rate is 0 then no parameters are updated
            lr = self.optimizer.param_groups[0]['lr']
            if lr == 0:
                cfg.LOG.write('lr=0, running train steps with no_grad()')
                with torch.no_grad():
                    self._train_step(train_gen, epoch, self.optimizer, iterations=iterations)
            else:
                self._train_step(train_gen, epoch, self.optimizer, iterations=iterations)

            torch.cuda.empty_cache()
            top1_acc = self.test(test_gen).item()

            if top1_acc > self.best_top1_acc:
                self.best_top1_acc = top1_acc
                self._save_state(epoch, desc=desc)

    def _train_step(self, train_gen, epoch, optimizer, iterations=None, bn_train=True):
        self.model.train()

        # BN to gather running mean and var (does not require backprop!)
        if not bn_train:
            for m in self.model.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()

        batch_time = self.AverageMeter('Time', ':6.3f')
        data_time = self.AverageMeter('Data', ':6.3f')
        losses = self.AverageMeter('Loss', ':.4e')
        top1 = self.AverageMeter('Acc@1', ':6.2f')
        top5 = self.AverageMeter('Acc@5', ':6.2f')
        progress = self.ProgressMeter(len(train_gen), batch_time, data_time, losses, top1,
                                      top5, prefix="Epoch: [{}]".format(epoch))

        end = time.time()
        for i, (input, target) in enumerate(train_gen):
            # measure data loading time
            data_time.update(time.time() - end)

            input = input.cuda(self.device, non_blocking=True)
            target = target.cuda(self.device, non_blocking=True)

            # Compute output
            output = self.parallel_model(input)
            loss = self.criterion(output, target)

            # Measure accuracy and record loss
            acc1, acc5 = self._accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # Compute gradient and do SGD step
            # Bypass when the learning rate is zero
            if self.optimizer.param_groups[0]['lr'] != 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.print(i)

            if iterations is not None:
                if i == iterations:
                    break

    def _adjust_lr_rate(self, optimizer, epoch, lr_dict):
        if lr_dict is None:
            return

        for key, value in lr_dict.items():
            if epoch == key:
                cfg.LOG.write("=> New learning rate set to {}".format(value))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = value

    def _save_state(self, epoch, desc=None):
        if desc is None:
            filename = '{}_epoch-{}_top1-{}.pth'.format(self.arch, epoch, round(self.best_top1_acc, 2))
        else:
            filename = '{}_epoch-{}_{}_top1-{}.pth'.format(self.arch, epoch, desc, round(self.best_top1_acc, 2))
        path = '{}/{}'.format(cfg.LOG.path, filename)

        state = {'arch': self.arch,
                 'epoch': epoch + 1,
                 'state_dict': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'lr_plan': self.lr_plan,
                 'best_top1_acc': self.best_top1_acc}

        torch.save(state, path)

    def _load_state(self, path):
        if os.path.isfile(path):
            chkp = torch.load(path)

            # Load class variables from checkpoint
            self.next_train_epoch = chkp['epoch'] if 'epoch' in chkp else 0

            # Assuming parameters are either in state_dict or in model keys
            state_dict =  chkp['state_dict'] if 'state_dict' in chkp else chkp['model']
            try:
                self.model.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                cfg.LOG.write('Loading model state warning, please review')
                cfg.LOG.write('{}'.format(e))

            self.optimizer_state = chkp['optimizer']
            self.lr_plan = chkp['lr_plan'] if 'lr_plan' in chkp else None
            self.best_top1_acc = chkp['best_top1_acc'] if 'best_top1_acc' in chkp else 0

            cfg.LOG.write("Checkpoint best top1 accuracy is {} @ epoch {}"
                          .format(round(self.best_top1_acc, 2), self.next_train_epoch - 1))
        else:
            cfg.LOG.write("Unable to load model checkpoint from {}".format(path))
            raise RuntimeError

    def print_stats(self, only_cfg=False, t1_analysis=False):
        raise NotImplementedError

    @staticmethod
    def _accuracy(output, target, topk=(1,)):
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    @staticmethod
    class AverageMeter(object):
        def __init__(self, name, fmt=':f'):
            self.name = name
            self.fmt = fmt
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

        def __str__(self):
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
            return fmtstr.format(**self.__dict__)

    @staticmethod
    class ProgressMeter(object):
        def __init__(self, num_batches, *meters, prefix=""):
            self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
            self.meters = meters
            self.prefix = prefix

        def print(self, batch):
            entries = [self.prefix + self.batch_fmtstr.format(batch)]
            entries += [str(meter) for meter in self.meters]
            cfg.LOG.write('\t'.join(entries))

        def _get_batch_fmtstr(self, num_batches):
            num_digits = len(str(num_batches // 1))
            fmt = '{:' + str(num_digits) + 'd}'
            return '[' + fmt + '/' + fmt.format(num_batches) + ']'
