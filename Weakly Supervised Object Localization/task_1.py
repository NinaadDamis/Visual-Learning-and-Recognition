import argparse
import os
import shutil
import time

import sklearn
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import wandb

from AlexNet import localizer_alexnet, localizer_alexnet_robust
from voc_dataset import *
from utils import *
from matplotlib import pyplot as plt

CLASS_NAMES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
    'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

USE_WANDB = True  # use flags, wandb is not convenient for debugging
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', default='localizer_alexnet_robust')
parser.add_argument(
    '-j',
    '--workers',
    default=4,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 4)')
parser.add_argument(
    '--epochs',
    default=46,
    type=int,
    metavar='N',
    help='number of total epochs to run')
parser.add_argument(
    '--start-epoch',
    default=0,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')
parser.add_argument(
    '-b',       
    '--batch-size',
    default=32,
    type=int,
    metavar='N',
    help='mini-batch size (default: 256)')
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=0.1,
    type=float,
    metavar='LR',
    help='initial learning rate')
parser.add_argument(
    '--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument(
    '--weight-decay',
    '--wd',
    default=1e-4,
    type=float,
    metavar='W',
    help='weight decay (default: 1e-4)')
parser.add_argument(
    '--print-freq',
    '-p',
    default=10,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--eval-freq',
    default=2,
    type=int,
    metavar='N',
    help='print frequency (default: 10)')
parser.add_argument(
    '--resume',
    default='',
    type=str,
    metavar='PATH',
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '-e',
    '--evaluate',
    dest='evaluate',
    action='store_true',
    help='evaluate model on validation set')
parser.add_argument(
    '--pretrained',
    dest='pretrained',
    action='store_false',
    help='use pre-trained model')
parser.add_argument(
    '--world-size',
    default=1,
    type=int,
    help='number of distributed processes')
parser.add_argument(
    '--dist-url',
    default='tcp://224.66.41.62:23456',
    type=str,
    help='url used to set up distributed training')
parser.add_argument(
    '--dist-backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--vis', action='store_true')

best_prec1 = 0

sigmoid_layer = nn.Sigmoid()

columns = ["Epoch","Class","Original Image","Heatmap"]
def main():
    global args, best_prec1
    args = parser.parse_args()
    args.distributed = args.world_size > 1

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'localizer_alexnet':
        print("Normal architecture used")
        model = localizer_alexnet(pretrained=args.pretrained)
    elif args.arch == 'localizer_alexnet_robust':
        print("Robust architecture used")
        model = localizer_alexnet_robust(pretrained=args.pretrained)
    print(model)

    # model.features = torch.nn.DataParallel(model.features)
    model.to(device)
    # TODO (Q1.1): define loss function (criterion) and optimizer from [1]
    # also use an LR scheduler to decay LR by 10 every 30 epochs
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code

    # TODO (Q1.1): Create Datasets and Dataloaders using VOCDataset
    # Ensure that the sizes are 512x512
    # Also ensure that data directories are correct
    # The ones use for testing by TAs might be different
    train_dataset = VOCDataset(split='trainval', top_n=10, image_size=512)
    val_dataset = VOCDataset(split='test', top_n=10, image_size=512)
    train_sampler = None

#   Add collatre function, batch size > 1
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn = train_dataset.collate_fn,
        sampler=train_sampler,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn = val_dataset.collate_fn,
        drop_last=True)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! VAL EVALUATE = ", args.evaluate)

    # TODO (Q1.3): Create loggers for wandb.
    # Ideally, use flags since wandb makes it harder to debug code.
    if USE_WANDB:
        wandb.init(project="Plot45_Q1.7 ", reinit=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        print("##################################################################################### Epoch is = ", epoch)
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            m1, m2 = validate(val_loader, model, criterion, epoch)

            score = m1 * m2
            # remember best prec@1 and save checkpoint
            is_best = score > best_prec1
            best_prec1 = max(score, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)
        scheduler.step()
        # validate(val_loader, model, criterion)


# TODO: You can add input arguments if you wish
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (data) in enumerate(train_loader):
        optimizer.zero_grad()
        # measure data loading time
        data_time.update(time.time() - end)
        # TODO (Q1.1): Get inputs from the data dict
        # Convert inputs to cuda if training on GPU
        target = data["label"].to(device)
        input = data["image"].to(device)
        # TODO (Q1.1): Get output from model
        imoutput = model(input)
        # print("imoutput shape = ", imoutput.shape)

        # TODO (Q1.1): Perform any necessary operations on the output
        if args.arch == 'localizer_alexnet_robust':
            adaptivePoolLayer = nn.AvgPool2d(kernel_size=5)
            logits = adaptivePoolLayer(imoutput)
            adaptiveMaxPoolLayer = nn.AdaptiveMaxPool2d(output_size=(1,1))

            # Alternatively replace avg ppol and adaptive max pool with adaptive avg pool            
            # adaptiveAvgPoolLayer = nn.AdaptiveAvgPool2d(output_size=(1,1))

            logits = adaptiveMaxPoolLayer(logits)
            output = torch.squeeze(logits)

        if args.arch == 'localizer_alexnet':
            adaptivePoolLayer = nn.AdaptiveMaxPool2d(output_size=(1,1))
            logits = adaptivePoolLayer(imoutput)
            output = torch.squeeze(logits)

        # TODO (Q1.1): Compute loss using ``criterion``
        loss = criterion(output,target)

        # measure metrics and record loss
        m1 = metric1(output.data, target)
        m2 = metric2(output.data, target)
        losses.update(loss.item(), input.size(0))
        avg_m1.update(m1)
        avg_m2.update(m2)

        # TODO (Q1.1): compute gradient and perform optimizer step
        # optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))

        # TODO (Q1.3): Visualize/log things as mentioned in handout at appropriate intervals
        if USE_WANDB:
            wandb.log({'train/loss': losses.avg})
            wandb.log({'train/metric1': avg_m1.avg})
            wandb.log({'train/metric2': avg_m2.avg})

            if epoch == 0 or epoch == 1 or epoch==15 or epoch==30 or epoch ==45:
                if i == 25 or i == 50:
                    # Logging the first image of the given iterations
                    original_image = input[0,:,:,:] # original image
                    # print("###################################################Image shape before resizze = ", original_image.shape)
                    # imoutput = sigmoid_layer(imoutput)
                    resized_out = transforms.Resize((512, 512))(imoutput[0,:,:,:])
                    # print("Drezied out shape = ", resized_out.shape)
                    # heat_map = resized_out[0,:,:]

                    #Get label of the first image , idx = 0
                    label_vec = target[0,:]
                    index = label_vec.argmax()
                    class_id = index.cpu().detach().item()

                    # Select the channel corresponding to the class index. Obtain 2D shape.
                    heat_map = resized_out[class_id,:,:]

                    # Save and reload image to convert it into jet colormap.
                    plt.imsave('temp.png', heat_map.cpu().detach().numpy(), cmap='jet')
                    load_img = Image.open('temp.png').resize((512,512))

                    # Add image to table.
                    test_table = wandb.Table(columns=columns)
                    table_name = "Table" + str(epoch) + "-" + str(i)
                    test_table.add_data(epoch,CLASS_NAMES[class_id], wandb.Image(original_image),wandb.Image(load_img))
                    wandb.log({table_name: test_table})    

        # End of train()


def validate(val_loader, model, criterion, epoch=0):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_m1 = AverageMeter()
    avg_m2 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    print("*******************************************************************************************************************")
    end = time.time()
    for i, (data) in enumerate(val_loader):

        # TODO (Q1.1): Get inputs from the data dict
        # Convert inputs to cuda if training on GPU
        target = data["label"].to(device)
        input = data["image"].to(device)
        # TODO (Q1.1): Get output from model
        imoutput = model(input)

        # TODO (Q1.1): Perform any necessary functions on the output
        if args.arch == 'localizer_alexnet_robust':
            adaptivePoolLayer = nn.AvgPool2d(kernel_size=5)
            logits = adaptivePoolLayer(imoutput)
            adaptiveMaxPoolLayer = nn.AdaptiveMaxPool2d(output_size=(1,1))

            # Alternatively replace avg ppol and adaptive max pool with adaptive avg pool            
            # adaptiveAvgPoolLayer = nn.AdaptiveAvgPool2d(output_size=(1,1))

            logits = adaptiveMaxPoolLayer(logits)
            output = torch.squeeze(logits)
        if args.arch == 'localizer_alexnet':
            # output = maxPoolOutput(outputModel)
            adaptivePoolLayer = nn.AdaptiveMaxPool2d(output_size=(1,1))
            logits = adaptivePoolLayer(imoutput)
            output = torch.squeeze(logits)
        # TODO (Q1.1): Compute loss using ``criterion``
        loss = criterion(output,target)

        # measure metrics and record loss
        m1 = metric1(output.data, target)
        m2 = metric2(output.data, target)
        losses.update(loss.item(), input.size(0))
        avg_m1.update(m1)
        avg_m2.update(m2)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Metric1 {avg_m1.val:.3f} ({avg_m1.avg:.3f})\t'
                  'Metric2 {avg_m2.val:.3f} ({avg_m2.avg:.3f})'.format(
                      i,
                      len(val_loader),
                      batch_time=batch_time,
                      loss=losses,
                      avg_m1=avg_m1,
                      avg_m2=avg_m2))

        # TODO (Q1.3): Visualize things as mentioned in handout
        # TODO (Q1.3): Visualize at appropriate intervals
        if USE_WANDB:
            wandb.log({'validation/loss': losses.avg})
            wandb.log({'validation/metric1': avg_m1.avg})
            wandb.log({'validation/metric2': avg_m2.avg})

        # Visualize after the last epoch 
        if(epoch == 45):
            # Every 10th iteration
            if i % 10 == 0: 
                original_image = input[0,:,:,:] # original image
                # imoutput = sigmoid_layer(imoutput)
                resized_out = transforms.Resize((512, 512))(imoutput[0,:,:,:])
                # print("Drezied out shape = ", resized_out.shape)
                
                #Get label of the first image , idx = 0   
                label_vec = target[0,:] 
                index = label_vec.argmax()
                class_id = index.cpu().detach().item()

                # Select the channel corresponding to the class index. Obtain 2D shape.
                heat_map = resized_out[class_id,:,:]

                # Save and reload image to convert it into jet colormap.
                plt.imsave('heat_map.png', heat_map.cpu().detach().numpy(), cmap='jet')
                load_img = Image.open('heat_map.png').resize((512,512))

                #Add image to table.
                test_table = wandb.Table(columns=columns)
                table_name = "Table" + str(epoch) + "-" + str(i)
                test_table.add_data(epoch,CLASS_NAMES[class_id]+"Validation", wandb.Image(original_image),wandb.Image(load_img))
                wandb.log({table_name: test_table})    

    print(' * Metric1 {avg_m1.avg:.3f} Metric2 {avg_m2.avg:.3f}'.format(
        avg_m1=avg_m1, avg_m2=avg_m2))

    return avg_m1.avg, avg_m2.avg


# TODO: You can make changes to this function if you wish (not necessary)
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


def metric1(output, target):
    # TODO (Q1.5): compute metric1
    # Precision = TP / (TP + FP) == TP / Predicted positives
    output     = sigmoid_layer(output)
    # print("Metric 1 Shapes = ", output.shape, target.shape) (32,20)
    # output = output.numpy().astype('float32')
    # target = target.numpy().astype('float32')
    output_arr = output.detach().cpu().numpy()
    target_arr = target.detach().cpu().numpy()
    num_classes = output.shape[1]
    classwise_ap = []
    for cls in range(num_classes):
        # Dont append if there are no gt in the class
        # if(np.amax(target_arr[:,cls]) == 0):
        all_zeros = np.all(target_arr[:,cls] == 0)
        if (all_zeros != True):
            ap = sklearn.metrics.average_precision_score(target_arr[:,cls],output_arr[:,cls])
            classwise_ap.append(ap)
    classwise_ap = np.array(classwise_ap)
    return np.mean(classwise_ap)


def metric2(output, target):
    # TODO (Q1.5): compute metric2

    # Recall = TP / TP + FN == TP / Total num of ground truth positives

    # print("Metric 2 - Types  = ", type(output), type(target))
    # print("Metric 2 - Shapes = ", output.shape, target.shape)
    # Logits to (0,1)
    output = sigmoid_layer(output)

    # Convert to np array (detach = removes gradient, cpu = moves to cpu memory, numpy converts to array)
    # output_arr = output.numpy().astype('float32')
    # target_arr = target.numpy().astype('float32')
    output_arr = output.detach().cpu().numpy()
    target_arr = target.detach().cpu().numpy()
    # print("Type output arr = ", type(output_arr))

    # Assume a threshold of 0.5. If > 0.5, assumed to be true else false
    output_arr = np.where(output_arr > 0.5, 1, 0)

    # Use zero division instead of looping over to check zero denom.
    r = sklearn.metrics.recall_score(target_arr, output_arr, zero_division = 1, average = 'samples')
    return r


if __name__ == '__main__':
    
    cuda = torch.cuda.is_available()
    print("Version torch : " , torch.__version__)
    print("Arch list : " , torch.cuda.get_arch_list())
    print("Is CUDA AVAILABLE : ", cuda)
    device = torch.device("cuda")
    main()
