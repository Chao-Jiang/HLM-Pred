from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_conditions import TrainConditions


def main():
    parser = argparse.ArgumentParser(description='Run the Table Tennis Ball Prediction algorithm.')
    parser.add_argument('--train_condition', '-tc', type=str, choices=['toy_test',
                                                                       'coordinate_input',
                                                                       'center_pixel_input',
                                                                       'real_time_play',
                                                                       'multiframe_pred'],
                        default='center_pixel_input', help='train_condition: \'toy_test\', '
                                                         '\'coordinate_input\', \'center_pixel_input\','
                                                           '\'multiframe_pred\' or \'real_time_play\'')
    parser.add_argument('--train_dir', type=str, default='./../train',
                        help='path to the training directory.')
    parser.add_argument('--restore_training', '-rt', action='store_true',
                        help='restore training')
    parser.add_argument('--restore_step', '-rs', type=int, default=None, help='checkpoint file to restore')
    parser.add_argument('--test', action='store_true',
                        help='run algorithm in testing mode')
    parser.add_argument('--rnn_cell', type=str, choices=['lstm', 'gru'],
                        default='lstm', help='type of rnn cell: \'lstm\' or \'gru\'')
    parser.add_argument('--num_cells', type=int, default=3, help='number of rnn cells')
    parser.add_argument('--num_mixtures', type=int, default=3, help='number of mixtures for MDN')
    parser.add_argument('--seq_length', '-sl', type=int, default=6, help='sequence length')
    parser.add_argument('--gaussian_dim', '-gd', type=int, default=3, help='dimentionality of gaussian distribution')
    parser.add_argument('--features_dim', '-fd', type=int, default=3, help='dimentionality of input features')
    parser.add_argument('--weight_decay', '-wd', type=float, default=0.005, help='scale factor of regularization of neural network')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=3000, help='number of epochs')
    parser.add_argument('--keep_prob', type=float, default=0.6, help='dropout keep probability')
    parser.add_argument('--grad_clip', type=float, default=10., help='clip gradients at this value')
    parser.add_argument('--random_seed', type=int, default=0, help='random seed')
    parser.add_argument('--ckpt_interval', type=int, default=800, help='checkpoint file save interval')
    parser.add_argument('--predicted_step', '-ps', type=int,
                        default=17, help='which time step after the '
                                         'last sequence input to make prediction ')


    args = parser.parse_args()
    np.random.seed(args.random_seed)

    inst = TrainConditions(args=args)
    inst.run()

if __name__ == "__main__":
    main()

 # 3 Threshold max: step = 58400
 #       32.83 %    66.83 %    83.17 %
# -fd 4 --num_cells=2 -wd=0.012 -sl=7 --keep_prob=0.8 --rnn_cell=gru