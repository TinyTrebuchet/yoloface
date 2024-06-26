import os
import math
import argparse
import os.path as osp

import cv2
import torch
import numpy as np

from face_detector import YoloDetector

##################################################

torch.distributed.init_process_group(backend='nccl')

rank = int(os.environ['RANK'])
world_size = int(os.environ['WORLD_SIZE'])
print(f"[+] Rank {rank} : Worker initialized!")

if rank == 0:
    print(f"World Size : {world_size}")

parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
parser.add_argument('--config', type=str)
parser.add_argument('--weights', type=str)
parser.add_argument('--input', type=str)
parser.add_argument('--output', type=str)
args = parser.parse_args()

model = YoloDetector(args.weights, args.config, target_size=640, device="cuda:0", min_face=5)

in_dir = args.input
out_dir = args.output
os.makedirs(out_dir, exist_ok=True)
pids = set(os.listdir(in_dir))
done_pids = set(os.listdir(out_dir))
rem_pids = sorted(list(pids - done_pids))
work = math.ceil(len(rem_pids) / world_size)
local_pids = rem_pids[work*rank : work*(rank+1)]

for i,pid in enumerate(local_pids):
    print(f"Rank {rank} | Processing subject {pid} ({i+1}/{len(local_pids)})")

    for phase in sorted(os.listdir(osp.join(in_dir, pid))):
        phase = phase.lower()
        if phase not in ['phase_1', 'phase_2']:
            print("Skipping invalid value:", pid, phase)
            continue

        for vid in sorted(os.listdir(osp.join(in_dir, pid, phase))):
            try:
                view = int(vid.split('.')[0].split('_')[1])
                if view not in [0, 45, 90, 135, 180, 225, 270, 315]:
                    raise ValueError
                view = f'{view:03d}'
            except (ValueError, IndexError):
                print("Skipping invalid value:", pid, phase, vid)
                continue
            save_dir = osp.join(out_dir, pid, phase, view)
            os.makedirs(save_dir, exist_ok=True)

            cap = cv2.VideoCapture(osp.join(in_dir, pid, phase, vid))
            while True:
                ret, frame = cap.read()
                if not ret:
                    cap.release()
                    break
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    bboxes, points = model.predict(frame)
                    crops = model.align(frame, points[0])
                    if len(crops):
                        cv2.imwrite(osp.join(save_dir, f'{i:03d}.png'), cv2.cvtColor(crops[0], cv2.COLOR_RGB2BGR))
                except cv2.error:
                    pass
            