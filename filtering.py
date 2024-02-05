import pandas as pd
import csv
import os
from glob import glob
from collections import defaultdict

unbalanced_csv = "/share/workhorse2/ankit/public/datasets/audioset/audio_files_DO_NOT_DELETE/original_audio_set_files/unbalanced_train_segments.csv"
class_labels_indices_csv = "/share/workhorse2/ankit/public/datasets/audioset/audio_files_DO_NOT_DELETE/original_audio_set_files/class_labels_indices.csv"
audioset_unbalanced_path = "/share/workhorse3/aps1/DATA/audioset_download/download-2022-10-14/"

metadata = pd.read_csv(unbalanced_csv, 
                sep=', ', 
                skiprows=3,
                header=None,
                names=['YTID', 'start_seconds', 'end_seconds', 'positive_labels'],
                engine='python',
                )
metadata = metadata[metadata['positive_labels'].apply(lambda x: len(x.split(',')) == 1)]
metadata['positive_labels'] = metadata['positive_labels'].apply(lambda x: x.replace('"', '').split(",")[0])
# print(metadata)


num_samples_per_label = 500
num_class = 50

# filter classes with num_samples >= 500
label_counts = metadata['positive_labels'].value_counts()
print(len(label_counts))
label_counts = label_counts[label_counts >= num_samples_per_label].index.tolist()  # filter class where num_samples >= 500
print(len(label_counts))
metadata = metadata[metadata['positive_labels'].isin(label_counts)]
# print(metadata)


class_df = pd.read_csv(
            class_labels_indices_csv, 
            sep=',',
        )
total_list = class_df['mid'].tolist()
# print(total_list)
metadata = metadata[metadata['positive_labels'].isin(total_list)]
# print(metadata)

index_to_remove = []
class_count = defaultdict(int)
for index, row in metadata.iterrows():
    # print(index)
    ytid = row['YTID']
    start_seconds = str(int(row['start_seconds']))
    end_seconds = str(int(row['end_seconds']))
    tag = row['positive_labels']

    if class_count[tag] >= num_samples_per_label:
        index_to_remove.append(index)
    else:
        subdir = "**/processed"
        filename = os.path.join(audioset_unbalanced_path, subdir, "Y{}_{}_{}.wav".format(ytid, start_seconds, end_seconds))
        # print(filename)

        matching_paths = glob(filename)
        if not matching_paths:
            index_to_remove.append(index)
        else:
            class_count[tag] += 1

metadata = metadata.drop(index_to_remove)


# filter classes with num_samples >= 500, considering whether the path actually exists
label_counts = metadata['positive_labels'].value_counts()
label_counts = label_counts[label_counts >= num_samples_per_label].index.tolist()  # filter class where num_samples >= 500
print(label_counts)
metadata = metadata[metadata['positive_labels'].isin(label_counts)]

sampled_data = metadata.groupby('positive_labels', group_keys=False).apply(lambda x: x.sample(n=num_samples_per_label, random_state=42))

sampled_data.to_csv('./filtered_unbalanced_500_singlelabel.csv', index=False, sep=',')
print(sampled_data)
