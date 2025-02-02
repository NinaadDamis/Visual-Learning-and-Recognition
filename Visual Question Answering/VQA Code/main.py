"""Base experiment runner class for VQA experiments."""
import argparse
import os

import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import BaselineNet, TransformerNet
from vqa_dataset import VQADataset
from matplotlib import pyplot as plt
import numpy as np

class Trainer:
    """Train/test models on manipulation."""

    def __init__(self, model, data_loaders, args):
        self.model = model
        self.data_loaders = data_loaders
        self.args = args

        self.writer = SummaryWriter('runs/' + args.tensorboard_dir)
        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.optimizer = Adam(
            model.parameters(), lr=args.lr, betas=(0.0, 0.9), eps=1e-8
        )

        self._id2answer = {
            v: k
            for k, v in data_loaders['val'].dataset.answer_to_id_map.items()
        }
        self._id2answer[len(self._id2answer)] = 'Other'
        # print("ID 2 ANSWER = ", self._i)

    def run(self):
        # Set
        start_epoch = 0
        val_acc_prev_best = -1.0

        # Load
        if os.path.exists(self.args.ckpnt):
            start_epoch, val_acc_prev_best = self._load_ckpnt()

        # Eval?
        if self.args.eval or start_epoch >= self.args.epochs:
            print("USing .pt file ---> Doing just eval.")
            self.model.eval()
            self.train_test_loop('val')
            return self.model

        # Go!
        for epoch in range(start_epoch, self.args.epochs):
            print("Epoch: %d/%d" % (epoch + 1, self.args.epochs))
            self.model.train()
            # Train
            self.train_test_loop('train', epoch)
            # Validate
            print("\nValidation")
            self.model.eval()
            with torch.no_grad():
                val_acc = self.train_test_loop('val', epoch)

            # Store
            if val_acc >= val_acc_prev_best:
                print("Saving Checkpoint")
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_acc": val_acc
                }, self.args.ckpnt)
                val_acc_prev_best = val_acc
            else:
                print("Updating Checkpoint")
                checkpoint = torch.load(self.args.ckpnt)
                checkpoint["epoch"] += 1
                torch.save(checkpoint, self.args.ckpnt)

        return self.model

    def _load_ckpnt(self):
        ckpnt = torch.load(self.args.ckpnt)
        self.model.load_state_dict(ckpnt["model_state_dict"], strict=False)
        self.optimizer.load_state_dict(ckpnt["optimizer_state_dict"])
        start_epoch = ckpnt["epoch"]
        val_acc_prev_best = ckpnt['best_acc']
        return start_epoch, val_acc_prev_best

    def train_test_loop(self, mode='train', epoch=1000):
        n_correct, n_samples = 0, 0
        predicted_frequencies = torch.zeros(5217) # n_answers
        gt_frequencies = torch.zeros(5217) # n_answers
        for step, data in tqdm(enumerate(self.data_loaders[mode])):
            # print("Input image data shape = ", data['image'].shape)
            # Forward pass
            # if step == 25 : break
            scores = self.model(
                data['image'].to(self.args.device),
                data['question']
            )
            answers = data['answers'].to(self.args.device)

            # Losses
            # Uncomment these if you want to assign less weight to 'other'
            # pos_weight = torch.ones_like(answers[0])
            # pos_weight[-1] = 0.1  # 'Other' has lower weight
            # and use the pos_weight argument
            # ^OPTIONAL: the expected performance can be achieved without this
            loss = self.criterion(scores,answers)

            # Update
            if mode == 'train':
                # optimize loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Accuracy
            n_samples += len(scores)
            found = (
                F.one_hot(scores.argmax(1), scores.size(1))
                * answers
            ).sum(1)
            # print("Answers shape = ", answers.shape)
            # print("Scores shape = ", scores.shape)
            # print("Scores argmax shape", scores.argmax(1).shape )
            # print("one hot shape = ", F.one_hot(scores.argmax(1), scores.size(1)).shape)
            # print("mult shape = ", (F.one_hot(scores.argmax(1), scores.size(1))* answers).shape)
            n_correct += (
                F.one_hot(scores.argmax(1), scores.size(1))
                * answers
            ).sum().item()  # checks if argmax matches any ground-truth
            # prediction_idx   = torch.argmax(scores[5]).item()
            # print("Scores i shape a = {} b = {}  ".format(prediction_idx ))

            pred = F.one_hot(scores.argmax(1), scores.size(1)
            ).sum(0) # Sum over all rows to get count.
            # Detach to prevent runtime error
            # RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
            predicted_frequencies += pred.detach().cpu()
            # gt_frequencies += answers.sum(0).detach().cpu()

            # Logging
            self.writer.add_scalar(
                'Loss/' + mode, loss.item(),
                epoch * len(self.data_loaders[mode]) + step
            )
            if mode == 'val' and step == 0:  # change this to show other images
                _n_show = 125  # how many images to plot
                for i in range(_n_show):
                    self.writer.add_image(
                        'Image%d' % i, data['orig_img'][i].cpu().numpy(),
                        epoch * _n_show + i, dataformats='CHW'
                    )
                    prediction_idx   = torch.argmax(scores[i]).item()
                    ground_truth_idx = torch.argmax(answers[i]).item()
                    prediction_str   = self._id2answer[prediction_idx]
                    gt_str           = self._id2answer[ground_truth_idx] 
                    # add code to show the question
                    self.writer.add_text('Image %d' % i, "Question : " + data['question'][i] + " \n " + " Ground truth : " + gt_str +
                    " \n " + "Prediction : " + prediction_str + " \n ", global_step=epoch * _n_show + i, walltime=None)
                    # the gt answer
                    # self.writer.add_text('Ground truth %d' % i, data['question'][i], global_step=epoch * _n_show + i, walltime=None)
                    # # and the predicted answer
                    # self.writer.add_text('Prediction%d' % i, data['question'][i], global_step=epoch * _n_show + i, walltime=None)
            # add code to plot the current accuracy
            acc_piter = n_correct/n_samples
            self.writer.add_scalar(
                'Accuracy/' + mode, acc_piter,
                epoch * len(self.data_loaders[mode]) + step
            )


        # topk = torch.topk(gt_frequencies, 10,largest=True, sorted=True)
        # print("The 10 most frquently gt classes are :", len(topk[0]),len(topk[1]))
        # topk_idx = topk.indices.tolist()
        # topk_vals= topk.values.tolist()
        # for p in range(10):
        #     cls = self._id2answer[topk_idx[p]]
        #     val = topk_vals[p]
        #     print("Class : {} ; Frequency = {}".format(cls,val))

        # torch.topk(input, k, dim=None, largest=True, sorted=True, *, out=None) Output = tuple(values,indices)
        # sorted_predictions = sorted_predictions.numpy()
        sorted_predictions = np.sort(predicted_frequencies.numpy())
        sorted_predictions = sorted_predictions[::-1]
        print("len sorted = ", len(sorted_predictions))
        print(sorted_predictions)
        print("type srted = ", type(sorted_predictions[0]))
        x_bar = np.arange(len(sorted_predictions))
        # y_bar = np.arange(len(sorted_predictions)) * 100
        np.savetxt("x.csv", x_bar, delimiter=",")
        np.savetxt("y.csv", sorted_predictions, delimiter=",")
        plt.bar(x_bar,sorted_predictions)
        plt.savefig("histogram_simple.jpg")
        plt.show()

        topk = torch.topk(predicted_frequencies, 10,largest=True, sorted=True)
        print("The 10 most frquently predicted classes are :", len(topk[0]),len(topk[1]))
        topk_idx = topk.indices.tolist()
        topk_vals= topk.values.tolist()
        for p in range(10):
            cls = self._id2answer[topk_idx[p]]
            val = topk_vals[p]
            print("Class : {} ; Frequency = {}".format(cls,val))


        acc = n_correct / n_samples
        print(" {} Accuracy = {}".format(mode,acc))
        return acc


def main():
    """Run main training/test pipeline."""
    # Feel free to add more args, or change/remove these.
    parser = argparse.ArgumentParser(description='Load VQA.')
    parser.add_argument('--model', type=str, default='simple')
    parser.add_argument('--tensorboard_dir', type=str, default=None)
    parser.add_argument('--ckpnt', type=str, default=None)
    parser.add_argument('--data_path', type=str, default='./')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()
    data_path = args.data_path

    # Other variables
    args.train_image_dir = data_path + 'train2014/'
    args.train_q_path = data_path + 'OpenEnded_mscoco_train2014_questions.json'
    args.train_anno_path = data_path + 'mscoco_train2014_annotations.json'
    args.test_image_dir = data_path + 'val2014/'
    args.test_q_path = data_path + 'OpenEnded_mscoco_val2014_questions.json'
    args.test_anno_path = data_path + 'mscoco_val2014_annotations.json'
    if args.tensorboard_dir is None:
        args.tensorboard_dir = args.model
    if args.ckpnt is None:
        args.ckpnt = args.model + '.pt'
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Loaders
    train_dataset = VQADataset(
        image_dir=args.train_image_dir,
        question_json_file_path=args.train_q_path,
        annotation_json_file_path=args.train_anno_path,
        image_filename_pattern="COCO_train2014_{}.jpg"
    )
    val_dataset = VQADataset(
        image_dir=args.test_image_dir,
        question_json_file_path=args.test_q_path,
        annotation_json_file_path=args.test_anno_path,
        image_filename_pattern="COCO_val2014_{}.jpg",
        answer_to_id_map=train_dataset.answer_to_id_map
    )
    print(len(train_dataset), len(val_dataset))
    data_loaders = {
        mode: DataLoader(
            train_dataset if mode == 'train' else val_dataset,
            batch_size=args.batch_size,
            shuffle=mode == 'train',
            drop_last=mode == 'train',
            num_workers=4
        )
        for mode in ('train', 'val')
    }

    # Models
    if args.model == "simple":
        model = BaselineNet()
    elif args.model == "transformer":
        model = TransformerNet()
    else:
        raise ModuleNotFoundError()

    trainer = Trainer(model.to(args.device), data_loaders, args)
    trainer.run()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
