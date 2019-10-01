import os
import sys
import pickle
import argparse
import numpy as np

from hmdata import TotalCapture


def action2label(action):
    action = action[:-1]
    if action == 'acting':
        return 0
    elif action == 'freestyle':
        return 1
    elif action == 'rom':
        return 2
    elif action == 'walking':
        return 3


def gendata(data: TotalCapture, out_path, cues, part, pace, max_frame):
    samples = []
    labels = []
    for sub in data.subjects:
        for act in data.actions:
            label = action2label(act)
            sub_samples = []
            temp = data.data_dict[sub][act][cues]
            length = len(temp) % pace
            temp = temp[length:, :, :]
            temp = temp.transpose(2, 0, 1)

            for i in range(max_frame, temp.shape[1], pace):
                sample = temp[:, i-max_frame:i, :]
                sub_samples.append(sample)
                labels.append(label)

            sub_samples = np.array(sub_samples)
            if samples == []:
                samples = sub_samples
            else:
                samples = np.concatenate((samples, sub_samples), axis=0)
    samples = samples[:, :, :, :, np.newaxis]
    np.save(os.path.join(out_path, part+'_data.npy'), samples)
    with open(os.path.join(out_path, part+'_label.pkl'), 'wb') as f:
        pickle.dump(labels, f)


if __name__ == '__main__':
    # a = np.load('data/TotalCapture/vicon/train_data.npy')
    # with open('data/TotalCapture/imu/train_label.pkl', 'rb') as f:
    #     label = pickle.load(f)
    import ptvsd
    ptvsd.enable_attach(address=('localhost'))
    ptvsd.wait_for_attach()

    parser = argparse.ArgumentParser(description='TotalCapture Data Converter.')
    parser.add_argument(
        '--data_path', default='/media/ywj/Data/totalcapture/totalcapture')
    parser.add_argument('--out_folder', default='data/TotalCapture/vicon')
    parser.add_argument('--pace', default=100)
    parser.add_argument('--max_frame', default=500)
    arg = parser.parse_args()

    tp_train_data = TotalCapture(arg.data_path, cues='vicon', mode='train')
    gendata(data=tp_train_data, out_path=arg.out_folder,
            cues='vicon', part='train', pace=arg.pace, max_frame=arg.max_frame)
    print('==')
