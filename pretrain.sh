#!/bin/bash

#ESC50_path=/share/workhorse3/muqiaoy/Datasets/
ESC50_path=$1

python data/generate_json.py --dpath $ESC50_path --mode train --n_init_cls 50 --n_cls_a_task 50 --n_tasks 1
python data/generate_json.py --dpath $ESC50_path --mode test --n_init_cls 50 --n_cls_a_task 50 --n_tasks 1
python main.py --dataset ESC-50 --data_root ./collection --mode finetune --epoch 100 --n_init_cls 50 --n_cls_a_task 50 --n_tasks 1


