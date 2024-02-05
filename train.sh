#!/bin/bash

#Audioset_path=/workhorse3/Datasets/Audioset_balanced/

Audioset_path=$1

python data/generate_json.py --dataset Audioset --dpath $Audioset_path --mode train --n_init_cls 5 --n_cls_a_task 5 --n_tasks 10 --bal_train  /media/konan/DataDrive/ANKIT/ASC_FRESH/audio_files_DO_NOT_DELETE/original_audio_set_files/balanced_train_segments.csv  --audiosetclass /media/konan/DataDrive/ANKIT/ASC_FRESH/audio_files_DO_NOT_DELETE/original_audio_set_files/class_labels_indices.csv 

python data/generate_json.py --dataset Audioset --dpath $Audioset_path --mode test --n_init_cls 5 --n_cls_a_task 5 --n_tasks 10 --eval_path /media/konan/DataDrive/ANKIT/ASC_FRESH/audio_files_DO_NOT_DELETE/original_audio_set_files/eval_segments.csv --audiosetclass /media/konan/DataDrive/ANKIT/ASC_FRESH/audio_files_DO_NOT_DELETE/original_audio_set_files/class_labels_indices.csv

#python main.py --dataset Audioset --data_root ./collection --mode replay --n_init_cls 5 --n_cls_a_task 5 --n_tasks 10

# sed -i 's/\/home\/muqiaoy\/workhorse3\/Datasets\/Audioset_balanced\/eval_segments_original_segmented_downloaded\//\/share\/workhorse2\/ankit\/public\/datasets\/Audioset_balanced\/eval_segments_original_segmented_downloaded\//g' *.json
python main.py --dataset Audioset --data_root ./collection --mode replay --n_init_cls 5 --n_cls_a_task 5 --n_tasks 10 --num_pretrain_class 50 --load_from_ckpt  ./workspace/ESC-50/disjoint/save_models/last.pt
