"""
Helper script for extracting frames from the UCF-101 dataset
"""

import av
import glob
import os
import time
import tqdm
import datetime
import argparse
import pickle
import numpy as np


def extract_frames(video_path):
    frames = []
    video = av.open(video_path)
    for frame in video.decode(0):
        yield frame.to_image()


prev_time = time.time()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="videos-instance", help="Path to video dataset")
    parser.add_argument("--annot_path", type=str, default="daly1.1.0.pkl", help="Path to frames")

    opt = parser.parse_args()
    print(opt)

    with open(opt.annot_path, "rb") as f:
        daly = pickle.load(f, encoding='latin1')
        
    time_left = 0
    video_paths = glob.glob(os.path.join(opt.dataset_path, "*", '*'))
    files = []
    for f in video_paths:
        f = f.split('/')[-1][:11]
        if f not in files:
            files.append(f)

    for i, video_name in enumerate(files):
        video_id = video_name + '.mp4'
        video_types = daly['annot'][video_id]['annot'].keys()
        sequence_dct = {}
        lag_time = []
        for vid_type in video_types:
            n_instance = len(daly['annot'][video_id]['annot'][vid_type])
            if not os.path.exists(os.path.join('videos-pose', vid_type)):
                os.makedirs(os.path.join('videos-pose', vid_type))
            for ii in range(n_instance):
                beginTime = daly['annot'][video_id]['annot'][vid_type][ii]['beginTime']
                endTime = daly['annot'][video_id]['annot'][vid_type][ii]['endTime']
                begin = beginTime*daly['annot'][video_id]['annot'][vid_type][ii]['keyframes'][0]['frameNumber']/daly['annot'][video_id]['annot'][vid_type][ii]['keyframes'][0]['time']
                end = endTime*daly['annot'][video_id]['annot'][vid_type][ii]['keyframes'][0]['frameNumber']/daly['annot'][video_id]['annot'][vid_type][ii]['keyframes'][0]['time']
                begin = int(begin)
                end = int(end)
                instance_name = video_name+str(ii)
                n_keys = len(daly['annot'][video_id]['annot'][vid_type][ii]['keyframes'])
                pose = []
                for k in range(n_keys):
                    pose.append(daly['annot'][video_id]['annot'][vid_type][ii]['keyframes'][k]['pose'])
                sequence_path = os.path.join('videos-pose', vid_type, instance_name+'.npy')
                if not os.path.exists(sequence_path):
                    sequence_dct[begin] = {'end':end, 'type':vid_type, 'seq_path':sequence_path, 'pose':pose}
                    lag_time.append(begin) 

        lag_time.sort()
        for t in lag_time:
            np.save(sequence_dct[t]['seq_path'], sequence_dct[t]['pose'])
            #file = open(sequence_dct[t]['seq_path'], 'w')
            #file.write(str(sequence_dct[t]['pose']))
            #file.close()

        
        
