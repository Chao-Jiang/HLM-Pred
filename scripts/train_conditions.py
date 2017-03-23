import numpy as np
import rospy
from Model_SF import Model_SF
from Model_MF import Model_MF
from pingpang_control.srv import *
from sklearn.model_selection import train_test_split
from utility import build_toy_dataset, \
    get_coordinates, plot_3d, get_left_right_center_pixel, find_best_ckpt

class TrainConditions(object):
    def __init__(self, args):
        conditions = ['toy_test',
                      'coordinate_input',
                      'center_pixel_input',
                      'real_time_play',
                      'multiframe_pred']
        config = {conditions[0]: self._toy_test,
                  conditions[1]: self._coordinate_input_train,
                  conditions[2]: self._center_pixel_input_train,
                  conditions[3]: self._real_time_play,
                  conditions[4]: self._center_pixel_input_multiframe_output}
        self.func = config[args.train_condition]
        self.args = args
        if args.train_condition == 'multiframe_pred':
            self.model = Model_MF(args)
        elif args.train_condition == 'real_time_play' and args.multi_pred:
            self.model = Model_MF(args)
        else:
            self.model = Model_SF(args)


    def run(self):
        self.func(args=self.args)


    def _toy_test(self, args):
        num_traj_train = 6000
        num_traj_test = 100
        traj_time_steps = 20
        seq_length = args.seq_length
        assert seq_length < traj_time_steps, 'Sequence length should be less than the total time steps of the trajectory'
        data = np.zeros((num_traj_train + num_traj_test, traj_time_steps, 3))
        for i in range(num_traj_train + num_traj_test):
            speed = 4 + np.random.rand() * 1
            alpha = np.random.rand() * np.pi / 12 + np.pi / 4
            beta = (np.random.rand() - 0.5) * np.pi / 6
            coord = [np.random.rand() * 0.1, np.random.rand() * 0.1, np.random.rand() * 0.1 + 0.1]
            data[i, :, :] = build_toy_dataset(speed, alpha, beta, coord, traj_time_steps)
        train_data = data[:num_traj_train]
        test_data = data[num_traj_train:]
        X_train = train_data[:, 1:, :]
        y_train = train_data[:, args.seq_length + args.predicted_step - 1, :]
        X_test = test_data[:, 1:, :]
        y_test = test_data[:, args.seq_length + args.predicted_step - 1, :]
        if not args.test:
            self.model.fit(X_train, y_train)
        else:
            self.model.restore_model()
        # model.predict(X_test)
        y_preds, _ = self.model.test(X_test, y_test)
        plot_3d([X_test, y_preds])

    def _coordinate_input_train(self, args):
        features, labels = get_coordinates(args)
        train_data, test_data, _, _ = train_test_split(features, labels,
                                                       test_size=0.1,
                                                       random_state=args.random_seed)
        print("Traning data groups: %d" % train_data.shape[0])
        print("Testing data groups: %d" % test_data.shape[0])
        X_train = train_data[:, 1:, :]
        y_train = train_data[:, args.seq_length + args.predicted_step - 1, :]
        X_test = test_data[:, 1:, :]
        y_test = test_data[:, args.seq_length + args.predicted_step - 1, :]
        if not args.test:
            self.model.fit(X_train, y_train)
        else:
            train_num = X_test.shape[0]
            find_best_ckpt(args,
                           self.model,
                           X_train[:train_num],
                           y_train[:train_num],
                           X_test,
                           y_test,
                           restore=False)
            self.model.restore_model()
        # model.predict(X_test)
        print("\nTraining data testing\n---------------------")
        train_y_preds, _ = self.model.test(X_train, y_train, name='Train')
        print("\nTesting data testing\n---------------------")
        test_y_preds, _ = self.model.test(X_test, y_test, name='Test')
        draw_num = 1
        start = np.random.randint(0, X_test.shape[0] - draw_num)
        plot_3d([X_train[start:start + draw_num, :args.seq_length + args.predicted_step, :],
                 train_y_preds[start:start + draw_num]],
                title='Train', draw_now=False, seq_length=args.seq_length)
        plot_3d([X_test[start:start + draw_num, :args.seq_length + args.predicted_step, :],
                 test_y_preds[start:start + draw_num]],
                title='Test', draw_now=True, seq_length=args.seq_length)


    def _predict(self, pixel_centers):
        pixel_centers = np.asarray(pixel_centers.inputs).reshape(self.model.args.seq_length, self.model.args.features_dim)
        assert len(pixel_centers.shape) == 2, 'Request from client should contain two-dimensional data'
        assert pixel_centers.shape[0] == self.model.args.seq_length, \
            'Sequence length %d received does not match that in neural network' % pixel_centers.shape[0]
        y_preds = self.model.predict(np.expand_dims(pixel_centers, axis=0))
        if isinstance(self.model, Model_SF):
            pred_num = self.args.gaussian_dim
        else:
            pred_num = self.args.pred_frames_num * self.args.gaussian_dim
        outputs = np.zeros(pred_num + 1)
        outputs[:pred_num] = y_preds[0].reshape(-1)
        outputs[pred_num] = self.model.args.predicted_step
        return Table_TennisResponse(outputs.tolist())


    def _real_time_play(self, args):
        self.model.restore_model()
        rospy.init_node('Real_time_playing')
        rospy.Service('prediction_interface', Table_Tennis, self._predict)
        rospy.spin()


    def _center_pixel_input_train(self, args):
        features, coords, labels = get_left_right_center_pixel(args, restore=True, save=True)
        features_train, features_test, coords_train, coords_test = train_test_split(features,
                                                                                    coords,
                                                                                    test_size=0.1,
                                                                                    random_state=args.random_seed)
        start_steps = [1, 2, 3]
        for idx, value in enumerate(start_steps):
            if idx == 0:
                X_train = features_train[:, value: value + args.seq_length, :]
                y_train = coords_train[:, value + args.seq_length + args.predicted_step - 1, :]
                X_test = features_test[:, value: value + args.seq_length, :]
                y_test = coords_test[:, value + args.seq_length + args.predicted_step - 1, :]
            else:
                X_train = np.concatenate((X_train,
                                          features_train[:, value: value + args.seq_length, :]))
                y_train = np.concatenate((y_train,
                                          coords_train[:, value + args.seq_length + args.predicted_step - 1, :]))
                X_test = np.concatenate((X_test,
                                         features_test[:, value: value + args.seq_length, :]))
                y_test = np.concatenate((y_test,
                                         coords_test[:, value + args.seq_length + args.predicted_step - 1, :]))
        print("Traning data groups: %d" % X_train.shape[0])
        print("Testing data groups: %d" % X_test.shape[0])
        if not args.test:
            self.model.fit(X_train, y_train)
        else:
            train_num = X_test.shape[0]
            find_best_ckpt(args,
                           self.model,
                           X_train[:train_num],
                           y_train[:train_num],
                           X_test,
                           y_test,
                           restore=False)
            self.model.restore_model()
        # model.predict(X_test)
        print("\nTraining data testing\n---------------------")
        train_y_preds, _ = self.model.test(X_train, y_train, draw=False, name='Train')
        print("\nTesting data testing\n---------------------")
        test_y_preds, _ = self.model.test(X_test, y_test, draw=False, name='Test')
        draw_num = 1
        start = np.random.randint(0, coords_test.shape[0] - draw_num)
        for i in xrange(len(start_steps)):
            plot_3d([coords_train[start:start + draw_num, :start_steps[i]+args.seq_length + args.predicted_step, :],
                     train_y_preds[coords_train.shape[0] * i + start: coords_train.shape[0] * i + start + draw_num]],
                    title='Train - start_step=%d'%start_steps[i], draw_now=False, seq_length=args.seq_length, start=start_steps[i])
            if i == len(start_steps) - 1:
                draw_now = True
            else:
                draw_now = False
            plot_3d([coords_test[start:start + draw_num, :start_steps[i]+args.seq_length + args.predicted_step, :],
                     test_y_preds[coords_test.shape[0] * i + start: coords_test.shape[0] * i + start + draw_num]],
                    title='Test - start_step=%d' % start_steps[i], draw_now=draw_now, seq_length=args.seq_length, start=start_steps[i])


    def _center_pixel_input_multiframe_output(self, args):
        features, coords, labels = get_left_right_center_pixel(args, restore=True, save=True)
        features_train, features_test, coords_train, coords_test = train_test_split(features,
                                                                                    coords,
                                                                                    test_size=0.1,
                                                                                    random_state=args.random_seed)
        start_steps = [1, 2, 3]
        for idx, value in enumerate(start_steps):
            if idx == 0:
                X_train = features_train[:, value: value + args.seq_length, :]
                y_start = value + args.seq_length + args.predicted_step - args.pred_frames_num
                y_train = coords_train[:, y_start: y_start + args.pred_frames_num, :]
                X_test = features_test[:, value: value + args.seq_length, :]
                y_test = coords_test[:, y_start: y_start + args.pred_frames_num, :]
            else:
                X_train = np.concatenate((X_train,
                                          features_train[:, value: value + args.seq_length, :]))
                y_start = value + args.seq_length + args.predicted_step - args.pred_frames_num
                y_train = np.concatenate((y_train,
                                          coords_train[:, y_start: y_start + args.pred_frames_num, :]))
                X_test = np.concatenate((X_test,
                                         features_test[:, value: value + args.seq_length, :]))
                y_test = np.concatenate((y_test,
                                         coords_test[:, y_start: y_start + args.pred_frames_num, :]))
        print("Traning data groups: %d" % X_train.shape[0])
        print("Testing data groups: %d" % X_test.shape[0])
        if not args.test:
            self.model.fit(X_train, y_train)
        else:
            train_num = X_test.shape[0]
            # find_best_ckpt(args,
            #                self.model,
            #                X_train[:train_num],
            #                y_train[:train_num],
            #                X_test,
            #                y_test,
            #                restore=False)
            self.model.restore_model()
        # model.predict(X_test)
        print("\nTraining data testing\n---------------------")
        train_y_preds, _ = self.model.test(X_train, y_train, draw=False, name='Train')
        print("\nTesting data testing\n---------------------")
        test_y_preds, _ = self.model.test(X_test, y_test, draw=False, name='Test')
        draw_num = 2
        start = np.random.randint(0, coords_test.shape[0] - draw_num)
        for i in xrange(len(start_steps)):
            plot_3d([coords_train[start:start + draw_num, :start_steps[i] + args.seq_length + args.predicted_step, :],
                     train_y_preds[coords_train.shape[0] * i + start: coords_train.shape[0] * i + start + draw_num]],
                    title='Train - start_step=%d' % start_steps[i], draw_now=False, seq_length=args.seq_length,
                    start=start_steps[i])
            if i == len(start_steps) - 1:
                draw_now = True
            else:
                draw_now = False
            plot_3d([coords_test[start:start + draw_num, :start_steps[i] + args.seq_length + args.predicted_step, :],
                     test_y_preds[coords_test.shape[0] * i + start: coords_test.shape[0] * i + start + draw_num]],
                    title='Test - start_step=%d' % start_steps[i], draw_now=draw_now, seq_length=args.seq_length,
                    start=start_steps[i])