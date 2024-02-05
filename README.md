### Sound Event Detection Adaptation Classifier

This codebase is developed for Sound Event Detection where Sound event detection is the automated identification or classification of specific audio events within a larger audio signal.

We first pretrain a base classifier which is pretrained classifier on ESC-50 dataset. 
The base classifier trained is incrementally increasing the number of classes being added each iteration to the new task. 
During training the classes are continually added based on number of tasks defined in the train.sh script. 

### Environment Setup

```
bash install_env.sh
```

```
conda activate asc_final
```

### Data Download links 

#### ESC-50 Dataset
```
Link to download ESC-50 - https://github.com/karoldvl/ESC-50/archive/master.zip
```

#### Audioset Dataset
```
Link to download Audioset - https://research.google.com/audioset/download.html

Here - place the csv from the dataset into a given path - lets call it original_audioset_csvs 
The original audioset csv folder should contain:

1. class_labels_indices.csv
2. eval_segments.csv
3. unbalanced_train_segments.csv
4. balanced_train_segments.csv
5. qa_true_counts.csv
```

Due to GitHub filesize limitations these files cannot be copied to the repository.


### Code Description

This code is first training ESC-50 (50 classes) as a pre-initialized model. Then it is fine-tuning on Audioset. Default setting is 5 classes per task and 10 tasks (so 50 new classes in total).

1. Save ESC-50 and Audioset to a certain path.

2. Train ESC-50 (50 classes) as a pre-initialized model. Change ```dpath``` in ```pretrain.sh``` accordingly.

3. The finetuning process needs the data path, initialized model and then use the partions and below pretraining receipt works well

Make sure that you launch the commands from asc_final conda environment
```
bash pretrain.sh <path to ESC-50 location>
```
4. Select certain classes from Audioset. Update ```unbalanced_csv```,  ```class_labels_indices_csv```, and ```audioset_unbalanced_path``` to correct paths accordingly.
```
python filtering.py
```

Currently, we want to select 50 new classes out of the Audioset unbalanced version, with 500 samples per class. So the ```filtering.py``` is taking out all classes with more than or equal to 500 samples. This processing may take a long time, maybe a few hours.

5. Finetune on Audioset. Change ```dpath``` in ```train.sh``` accordingly. Change the ```n_init_cls```, ```n_cls_a_task```, ```n_tasks``` if necessary.

Here -
`n_init_cls` is the number of initialized classes 
`n_cls_a_task` is the number of classes being added in each iteration of the task
`n_tasks` is the number of tasks that we have split the model into. 

Command to launch the finetuning is as follows:
```
bash train.sh <PATH to Audioset location>
```

### Expected performance post training

```
Evaluation accuracy - 0.45 after classification with 10 tasks
```
