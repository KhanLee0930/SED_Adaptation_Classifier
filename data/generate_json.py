import argparse
import json
import random
import os
from glob import glob

import pandas as pd
from soundata.datasets import tau2019uas
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Input optional guidance for training")
    parser.add_argument("--dpath", default="/media/konan/DataDrive/ANKIT/ASC_FRESH", type=str,
                        help="The path of dataset")
    parser.add_argument("--save_path", default="./collection", type=str,
                        help="The path to save generated jsons")
    parser.add_argument("--seed", type=int, default=5, help="Random seed number.")
    parser.add_argument("--dataset", type=str, default="ESC-50", help="[ESC-50, Audioset]")
    parser.add_argument("--n_tasks", type=int, default=5, help="The number of tasks")
    parser.add_argument("--n_cls_a_task", type=int, default=2, help="The number of class of each task")
    parser.add_argument("--n_init_cls", type=int, default=2, help="The number of classes of initial task")
    parser.add_argument("--exp_name", type=str, default="disjoint", help="[disjoint, blurry]")
    parser.add_argument("--mode", type=str, default="train", help="[train, test]")
    parser.add_argument("--audiosetclass", type=str,help="The class label indices from audioset")
    parser.add_argument("--bal_train", type=str,help="The balanced train csv from audioset")
    parser.add_argument("--eval_path", type=str,help="The eval set csv from audioset")
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    print(f"[0] Start to generate the {args.n_tasks} tasks of {args.dataset}.")
    if args.dataset == "Audioset":
        
        num_class = args.n_init_cls + args.n_cls_a_task * (args.n_tasks - 1)
        #class_df = pd.read_csv(
        #            f"/share/workhorse2/ankit/public/datasets/audioset/audio_files_DO_NOT_DELETE/original_audio_set_files/class_labels_indices.csv", 
        #            sep=',',
        #        ).head(num_class)
        class_df = pd.read_csv(args.audiosetclass,sep=',').head(num_class)

        display_to_machine_mapping = dict(zip(class_df['display_name'], class_df['mid']))
        machine_to_display_mapping = dict(zip(class_df['mid'], class_df['display_name']))
        selected_classes = list(machine_to_display_mapping.keys())

        #metadata = pd.read_csv(
        #            f"/share/workhorse2/ankit/public/datasets/audioset/audio_files_DO_NOT_DELETE/original_audio_set_files/balanced_train_segments.csv" \
        #                if args.mode == 'train' else f"/share/workhorse2/ankit/public/datasets/audioset/audio_files_DO_NOT_DELETE/original_audio_set_files/eval_segments.csv", 
        #            sep=', ', 
        #            skiprows=3,
        #            header=None,
        #            names=['YTID', 'start_seconds', 'end_seconds', 'positive_labels'],
        #            engine='python'
        #        )

        metadata = pd.read_csv( args.bal_train
                        if args.mode == 'train' else args.eval_path,
                    sep=', ', 
                    skiprows=3,
                    header=None,
                    names=['YTID', 'start_seconds', 'end_seconds', 'positive_labels'],
                    engine='python'
                )
        metadata = metadata.sort_values(by='YTID', ascending=True)
        # remove " in the labels
        metadata['positive_labels'] = metadata['positive_labels'].apply(lambda x: x.replace('"', '').split(",")[0])
        metadata = metadata.reset_index(drop=True)
        metadata = metadata[metadata['positive_labels'].isin(selected_classes)]

        random.seed(args.seed)
        random.shuffle(selected_classes)
        total_list = []
        for i in range(args.n_tasks):
            if i == 0:
                t_list = []
                for j in range(args.n_init_cls):
                    t_list.append(selected_classes[j])
                total_list.append(t_list)
            else:
                t_list = []
                for j in range(args.n_cls_a_task):
                    t_list.append((selected_classes[j + args.n_init_cls + (i - 1) * args.n_cls_a_task]))
                total_list.append(t_list)

        print(total_list)
        for i in range(len(total_list)):
            class_list = total_list[i]
            if args.mode == 'train':
                collection_name = "{dataset}_{mode}_{exp}_rand{rnd}_cls{n_cls}_task{iter}.json".format(
                    dataset=args.dataset, mode='train', exp=args.exp_name, rnd=args.seed, n_cls=args.n_cls_a_task,
                    iter=i
                )
            else:
                collection_name = "{dataset}_test_rand{rnd}_cls{n_cls}_task{iter}.json".format(
                    dataset=args.dataset, rnd=args.seed, n_cls=args.n_cls_a_task, iter=i
                )
            dataset_list = []
            for _, row in metadata.iterrows():
                ytid = row['YTID']
                start_seconds = str("{:.3f}".format(float(row['start_seconds'])))
                end_seconds = str("{:.3f}".format(float(row['end_seconds'])))
                tag = row['positive_labels']
                if tag in class_list:
                    subdir = "balanced_original_segmented_downloaded" if args.mode == 'train' else "eval_segments_original_segmented_downloaded"

                    filename = os.path.join(args.dpath, subdir, "Y{}_{}_{}.wav".format(ytid, start_seconds, end_seconds))
                    # print(filename)

                    if os.path.exists(filename):
                        dataset_list.append([filename, machine_to_display_mapping.get(tag), selected_classes.index(tag)])
            res = [{"tag": item[1], "audio_name": item[0], "label": item[2]} for item in dataset_list]
            print("Task ID is {}".format(i))
            print("Total samples are {}".format(len(res)))
            f = open(os.path.join(args.save_path, collection_name), 'w')
            f.write(json.dumps(res))
            f.close()
    elif args.dataset == "ESC-50":
        data_list = []
        meta = pd.read_csv(os.path.join(args.dpath, 'ESC-50-master/meta/esc50.csv'))

        test_fold_num = 1
        if args.mode == 'train':
            # data_list = meta[meta['fold'] != test_fold_num]
            data_list = meta
        elif args.mode == 'test':
            data_list = meta[meta['fold'] == test_fold_num]
        print(f'ESC-50 {args.mode} set using fold {test_fold_num} is creating, using sample rate {44100} Hz ...')
        class_list = sorted(data_list["category"].unique())
        random.seed(args.seed)
        random.shuffle(class_list)
        total_list = []
        for i in range(args.n_tasks):
            if i == 0:
                t_list = []
                for j in range(args.n_init_cls):
                    t_list.append(class_list[j])
                total_list.append(t_list)
            else:
                t_list = []
                for j in range(args.n_cls_a_task):
                    t_list.append((class_list[j + args.n_init_cls + (i - 1) * args.n_cls_a_task]))
                total_list.append(t_list)

            print(total_list)
            label_list = []
            for i in range(len(total_list)):
                class_list = total_list[i]
                label_list = label_list + class_list
                if args.mode == 'train':
                    collection_name = "{dataset}_{mode}_{exp}_rand{rnd}_cls{n_cls}" \
                                      "_task{iter}_{test_fold_num}.json".format(dataset=args.dataset, mode='train',
                                                                                exp=args.exp_name, rnd=args.seed,
                                                                                n_cls=args.n_cls_a_task,
                                                                                iter=i, test_fold_num=test_fold_num
                                                                                )

                else:
                    collection_name = "{dataset}_test_rand{rnd}_cls{n_cls}_task{iter}" \
                                      "_{test_fold_num}.json".format(dataset=args.dataset, rnd=args.seed,
                                                                     n_cls=args.n_cls_a_task, iter=i,
                                                                     test_fold_num=test_fold_num
                                                                     )
                f = open(os.path.join(args.save_path, collection_name), 'w')
                class_encoding = {category: index for index, category in enumerate(label_list)}
                dataset_list = []

                for index in tqdm(range(len(data_list))):
                    row = data_list.iloc[index]
                    file_path = os.path.join(args.dpath, 'ESC-50-master', 'audio', row["filename"])
                    if row['category'] in class_list:
                        dataset_list.append([file_path, row['category'], class_encoding.get(row['category'])])
                res = [{"tag": item[1], "audio_name": item[0], "label": item[2]} for item in dataset_list]
                print("Task ID is {}".format(i))
                print("Total samples are {}".format(len(res)))
                f.write(json.dumps(res))
                f.close()

    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
