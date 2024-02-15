from tqdm import tqdm

from data_loader_test import get_train_datalist
from data_loader_test import get_dataloader
import argparse
import pandas as pd

def datalist_test():
    parser = argparse.ArgumentParser(description='Example of parser. ')
        # Data root.
    parser.add_argument("--data_root", type=str, default='/root/SED_Adaptation_Classifier/Datasets/ESC-50-master/collection')
    parser.add_argument('--exp_name', type=str, default='disjoint')
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model_name', type=str, default='BC-ResNet')  # 'baseline' | 'BC-ResNet'
    parser.add_argument("--load_from_ckpt", type=str, default=None)
    parser.add_argument('--dataset', type=str, default='ESC-50')  # 'ESC-50' |
    parser.add_argument("--mode", type=str, default="replay", help="CIL methods [finetune, replay]", )
    parser.add_argument(
        "--mem_manage",
        type=str,
        default='uncertainty',
        help="memory management [random, uncertainty, reservoir, prototype]",
    )
    parser.add_argument("--num_pretrain_class", type=int, default=None, help="The number of classes in pretrained model")
    parser.add_argument("--n_tasks", type=int, default=1, help="The number of tasks")
    parser.add_argument(
        "--n_cls_a_task", type=int, default=50, help="The number of class of each task"
    )
    parser.add_argument(
        "--n_init_cls",
        type=int,
        default=50,
        help="The number of classes of initial task",
    )
    parser.add_argument("--rnd_seed", type=int, default=5, help="Random seed number.")
    parser.add_argument(
        "--memory_size", type=int, default=10000, help="Episodic memory size"
    )
    # Uncertain
    parser.add_argument(
        "--uncert_metric",
        type=str,
        default="noisytune",
        choices=["shift", "noise", "mask", "combination", "noisytune"],
        help="A type of uncertainty metric",
    )
    parser.add_argument("--metric_k", type=int, default=6, choices=[2, 4, 6],
                        help="The number of the uncertainty metric functions")
    parser.add_argument("--noise_lambda", type=float, default=0.2,
                        help="The number of the uncertainty metric functions")
    # Debug
    parser.add_argument("--debug", action="store_true", help="Turn on Debug mode")
    args = parser.parse_args()
    print(args)
    # type(pd.read_json('/root/SED_Adaptation_Classifier/Datasets/ESC-50-master/collection/ESC-50_train_disjoint_rand5_cls50_task0_1.json'))
    train_list = get_train_datalist(args,0)[0]

    train_loader = get_dataloader(pd.DataFrame(train_list), 'ESC-50', split='train', batch_size=128, num_class=50,
                                      num_workers=8)
    print(train_loader)
    for batch_data_dict in train_loader:
        print(batch_data_dict)

if __name__ == "__main__":
    datalist_test()