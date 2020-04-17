import argparse
import os
import random
import string
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data

from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from losses.coral import CORAL
from seqda_model import Model
from test import validation
from utils import AttnLabelConverter, Averager, load_char_dict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def coral_loss(source_context_history, source_prediction,
               target_context_history, target_prediction):
    feature_dim = source_context_history.size()[-1]

    source_feature = source_context_history.reshape(-1, feature_dim)
    target_feature = target_context_history.reshape(-1, feature_dim)

    # print(type(pred_class),pred_class)
    _, source_pred_class = source_prediction.max(-1)
    _, target_pred_class = target_prediction.max(-1)
    source_valid_char_index = (source_pred_class.reshape(-1, ) != 1).nonzero().reshape(-1, )
    source_valid_char_feature = source_feature.reshape(-1, feature_dim).index_select(0,
                                                                                     source_valid_char_index)
    target_valid_char_index = (target_pred_class.reshape(-1, ) != 1).nonzero().reshape(-1, )
    target_valid_char_feature = target_feature.reshape(-1, feature_dim).index_select(0,
                                                                                     target_valid_char_index)

    similarity_loss = CORAL(
        source_valid_char_feature,
        target_valid_char_feature)
    return similarity_loss


class trainer(object):
    def __init__(self, opt):

        opt.src_select_data = opt.src_select_data.split('-')
        opt.src_batch_ratio = opt.src_batch_ratio.split('-')
        opt.tar_select_data = opt.tar_select_data.split('-')
        opt.tar_batch_ratio = opt.tar_batch_ratio.split('-')

        """ vocab / character number configuration """
        if opt.sensitive:
            # opt.character += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

        if opt.char_dict is not None:
            opt.character = load_char_dict(opt.char_dict)[3:-2]  # 去除Attention 和 CTC引入的一些特殊符号

        """ model configuration """
        self.converter = AttnLabelConverter(opt.character)
        opt.num_class = len(self.converter.character)

        if opt.rgb:
            opt.input_channel = 3
        self.opt = opt
        print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel,
              opt.output_channel,
              opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation,
              opt.FeatureExtraction,
              opt.SequenceModeling, opt.Prediction)
        self.save_opt_log(opt)

        self.build_model(opt)

    def dataloader(self, opt):
        src_train_data = opt.src_train_data
        src_select_data = opt.src_select_data
        src_batch_ratio = opt.src_batch_ratio
        src_train_dataset = Batch_Balanced_Dataset(opt, src_train_data, src_select_data,
                                                   src_batch_ratio)

        tar_train_data = opt.tar_train_data
        tar_select_data = opt.tar_select_data
        tar_batch_ratio = opt.tar_batch_ratio
        tar_train_dataset = Batch_Balanced_Dataset(opt, tar_train_data, tar_select_data,
                                                   tar_batch_ratio)

        AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)

        valid_dataset = hierarchical_dataset(root=opt.valid_data, opt=opt)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=opt.batch_size,
            shuffle=True,  # 'True' to check training progress with validation function.
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_valid, pin_memory=True)
        return src_train_dataset, tar_train_dataset, valid_loader

    def _optimizer(self, opt):
        # filter that only require gradient decent
        filtered_parameters = []
        params_num = []
        for p in filter(lambda p: p.requires_grad, self.model.parameters()):
            filtered_parameters.append(p)
            params_num.append(np.prod(p.size()))
        print('Trainable params num : ', sum(params_num))
        # setup optimizer
        if opt.adam:
            self.optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
        else:
            self.optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho,
                                            eps=opt.eps)

        print("Optimizer:")
        print(self.optimizer)

    def build_model(self, opt):
        print('-' * 80)

        """ Define Model """
        self.model = Model(opt)

        self.weight_initializer()

        self.model = torch.nn.DataParallel(self.model).to(device)

        """ Define Loss """
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
        self.D_criterion = torch.nn.CrossEntropyLoss().to(device)

        """ Trainer """
        self._optimizer(opt)

    def train(self, opt):
        # Add custom dataset add cfgs from da-faster-rcnn
        # Make sure you change the imdb_name in factory.py
        """
        Dummy format:

        args.src_dataset == '$YOUR_DATASET_NAME'
        args.src_imdb_name = '$YOUR_DATASET_NAME_2007_trainval'
        args.src_imdbval_name = '$YOUR_DATASET_NAME_2007_test'
        args.set_cfgs = [...]
        """

        # src, tar dataloaders
        src_dataset, tar_dataset, valid_loader = self.dataloader(opt)
        src_dataset_size = src_dataset.total_data_size
        tar_dataset_size = tar_dataset.total_data_size
        train_size = max([src_dataset_size, tar_dataset_size])

        self.model.train()

        start_iter = 0

        if opt.continue_model != '':
            self.load(opt.continue_model)
            print(" [*] Load SUCCESS")
            # if opt.decay_flag and start_iter > (opt.num_iter // 2):
            #     self.d_image_opt.param_groups[0]['lr'] -= (opt.lr / (opt.num_iter // 2)) * (
            #             start_iter - opt.num_iter // 2)
            #     self.d_inst_opt.param_groups[0]['lr'] -= (opt.lr / (opt.num_iter // 2)) * (
            #             start_iter - opt.num_iter // 2)

        # loss averager
        cls_loss_avg = Averager()
        sim_loss_avg = Averager()
        loss_avg = Averager()

        # training loop
        print('training start !')
        start_time = time.time()
        best_accuracy = -1
        best_norm_ED = 1e+6

        for step in range(start_iter, opt.num_iter + 1):

            src_image, src_labels = src_dataset.get_batch()
            src_image = src_image.to(device)
            src_text, src_length = self.converter.encode(src_labels,
                                                         batch_max_length=opt.batch_max_length)

            tar_image, tar_labels = tar_dataset.get_batch()
            tar_image = tar_image.to(device)
            tar_text, tar_length = self.converter.encode(tar_labels,
                                                         batch_max_length=opt.batch_max_length)

            # Set gradient to zero...
            self.model.zero_grad()

            # Attention # align with Attention.forward
            src_preds, src_global_feature, src_local_feature = self.model(src_image,
                                                                          src_text[:, :-1])
            target = src_text[:, 1:]  # without [GO] Symbol
            src_cls_loss = self.criterion(src_preds.view(-1, src_preds.shape[-1]),
                                          target.contiguous().view(-1))

            src_local_feature = src_local_feature.view(-1, src_local_feature.shape[-1])
            # TODO 
            tar_preds, tar_global_feature, tar_local_feature = self.model(tar_image,
                                                                          tar_text[:, :-1],
                                                                          is_train=False)

            tar_local_feature = tar_local_feature.view(-1, tar_local_feature.shape[-1])

            d_inst_loss = coral_loss(src_local_feature, src_preds,
                                     tar_local_feature, tar_preds)
            # Add domain loss
            loss = src_cls_loss.mean() + 0.1 * d_inst_loss.mean()
            loss_avg.add(loss)
            cls_loss_avg.add(src_cls_loss)
            sim_loss_avg.add(d_inst_loss)

            # frcnn backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           opt.grad_clip)  # gradient clipping with 5 (Default)
            # frcnn optimizer update
            self.optimizer.step()

            # validation part
            if step % opt.valInterval == 0:

                elapsed_time = time.time() - start_time
                print(
                    f'[{step}/{opt.num_iter}] Loss: {loss_avg.val():0.5f} CLS_Loss: {cls_loss_avg.val():0.5f} SIMI_Loss: {sim_loss_avg.val():0.5f} elapsed_time: {elapsed_time:0.5f}')
                # for log
                with open(f'./saved_models/{opt.experiment_name}/log_train.txt', 'a') as log:
                    log.write(
                        f'[{step}/{opt.num_iter}] Loss: {loss_avg.val():0.5f} elapsed_time: {elapsed_time:0.5f}\n')
                    loss_avg.reset()
                    cls_loss_avg.reset()
                    sim_loss_avg.reset()

                    self.model.eval()
                    with torch.no_grad():
                        valid_loss, current_accuracy, current_norm_ED, preds, labels, infer_time, length_of_data = validation(
                            self.model, self.criterion, valid_loader, self.converter, opt)

                    self.print_prediction_result(preds, labels, log)

                    valid_log = f'[{step}/{opt.num_iter}] valid loss: {valid_loss:0.5f}'
                    valid_log += f' accuracy: {current_accuracy:0.3f}, norm_ED: {current_norm_ED:0.2f}'
                    print(valid_log)
                    log.write(valid_log + '\n')

                    self.model.train()

                    # keep best accuracy model
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                        save_name = f'./saved_models/{opt.experiment_name}/best_accuracy.pth'
                        self.save(opt, save_name)
                    if current_norm_ED < best_norm_ED:
                        best_norm_ED = current_norm_ED
                        save_name = f'./saved_models/{opt.experiment_name}/best_norm_ED.pth'
                        self.save(opt, save_name)

                    best_model_log = f'best_accuracy: {best_accuracy:0.3f}, best_norm_ED: {best_norm_ED:0.2f}'
                    print(best_model_log)
                    log.write(best_model_log + '\n')

            # save model per 1e+5 iter.
            if (step + 1) % 1e+5 == 0:
                save_name = f'./saved_models/{opt.experiment_name}/iter_{step+1}.pth'
                self.save(opt, save_name)

    def load(self, saved_model):
        params = torch.load(saved_model)

        if 'model' not in params:
            self.model.load_state_dict(params)
        else:
            self.model.load_state_dict(params['model'])

    def save(self, opt, save_name):
        params = {}
        params['model'] = self.model.state_dict()
        # for training
        params['optimizer'] = self.optimizer.state_dict()
        torch.save(params, save_name)
        print('Successfully save model: {}'.format(save_name))

    def weight_initializer(self):
        # weight initialization
        for name, param in self.model.named_parameters():
            if 'localization_fc2' in name:
                print(f'Skip {name} as it is already initialized')
                continue
            try:
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)
            except Exception as e:  # for batchnorm.
                if 'weight' in name:
                    param.data.fill_(1)
                continue

    def save_opt_log(self, opt):
        """ final options """
        # print(opt)
        with open(f'./saved_models/{opt.experiment_name}/opt.txt', 'a') as opt_file:
            opt_log = '------------ Options -------------\n'
            args = vars(opt)
            for k, v in args.items():
                opt_log += f'{str(k)}: {str(v)}\n'
            opt_log += '---------------------------------------\n'
            print(opt_log)
            opt_file.write(opt_log)

    def print_prediction_result(self, preds, labels, fp_log):
        """
         fp-logwenjian
        :param preds:
        :param labels:
        :param fp_log:
        :return:
        """
        for pred, gt in zip(preds[:5], labels[:5]):
            if 'Attn' in opt.Prediction:
                pred = pred[:pred.find('[s]')]
                gt = gt[:gt.find('[s]')]
            print(f'{pred:20s}, gt: {gt:20s},   {str(pred == gt)}')
            fp_log.write(f'{pred:20s}, gt: {gt:20s},   {str(pred == gt)}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', help='Where to store logs and models')
    parser.add_argument('--src_train_data', required=True, help='path to training dataset')
    parser.add_argument('--tar_train_data', required=True, help='path to training dataset')
    parser.add_argument('--valid_data', required=True, help='path to validation dataset')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=300000,
                        help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=2000,
                        help='Interval between each validation')
    parser.add_argument('--continue_model', default='', help="path to model to continue training")

    parser.add_argument('--adam', action='store_true',
                        help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1,
                        help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--decay_flag', action='store_true', help='for learning rate decay')
    parser.add_argument('--use_tfboard', action='store_true', help='use_tfboard')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.9')
    # parser.add_argument('--weight_decay', type=float, default=0.9, help='weight_decay for adam. default=0.9')
    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')

    parser.add_argument('--rho', type=float, default=0.95,
                        help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5,
                        help='gradient clipping value. default=5')

    """ Data processing """
    parser.add_argument('--src_select_data', type=str, default='MJ-ST',
                        help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    parser.add_argument('--src_batch_ratio', type=str, default='0.5-0.5',
                        help='assign ratio for each selected data in the batch')
    parser.add_argument('--tar_select_data', type=str, default='real_data',
                        help='select training data (default is real_data, which means MJ and ST used as training data)')
    parser.add_argument('--tar_batch_ratio', type=str, default='1',
                        help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--char_dict', type=str, default=None,
                        help="path to char dict: dataset/iam/char_dict.txt")
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz',
                        help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--filtering_special_chars', action='store_true',
                        help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true',
                        help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true',
                        help='for data_filtering_off mode')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True,
                        help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True,
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True,
                        help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20,
                        help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    if not opt.experiment_name:
        opt.experiment_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
        opt.experiment_name += f'-Seed{opt.manualSeed}'
    else:
        experiment_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
        experiment_name += f'-Seed{opt.manualSeed}'
        opt.experiment_name = experiment_name + opt.experiment_name
        # print(opt.experiment_name)

    os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)

    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    if opt.num_gpu > 1:
        print('------ Use multi-GPU setting ------')
        print('if you stuck too long time with multi-GPU setting, try to set --workers 0')
        opt.workers = opt.workers * opt.num_gpu

        """ previous version
        print('To equlize batch stats to 1-GPU setting, the batch_size is multiplied with num_gpu and multiplied batch_size is ', opt.batch_size)
        opt.batch_size = opt.batch_size * opt.num_gpu
        print('To equalize the number of epochs to 1-GPU setting, num_iter is divided with num_gpu by default.')
        If you dont care about it, just commnet out these line.)
        opt.num_iter = int(opt.num_iter / opt.num_gpu)
        """
    train = trainer(opt)
    train.train(opt)
