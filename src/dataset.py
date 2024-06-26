
from data_utils.all_datasets import MultiPDE

from logging import getLogger
import os

logger = getLogger()


def get_dataset(params, symbol_env, split, skip = 0):


    if split == "train":

        return MultiPDE(params,symbol_env,split = split, skip= skip)
    else:
        datasets = {}
        if    params.data.eval_types == -1:
            types = [name for name in os.listdir(params.data.directory) if
                              os.path.isdir(os.path.join(params.data.directory, name))]
        else:
            types =  params.data.eval_types if split == "eval" else params.data.train_types
            types = [types] if isinstance(types,str) else types
        for t in   types:
            ds = MultiPDE(params,symbol_env,split = split,types=t, skip= skip)
            datasets[t] = ds

        return datasets


if __name__ == "__main__":
    import hydra
    from torch.utils.data import DataLoader
    import logging
    import sys
    from data_utils.collate import custom_collate
    from symbol_utils.environment import SymbolicEnvironment
    from itertools import cycle
    import torch

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    def print_sample(sample):
        for k, v in sample.items():
            if isinstance(v, dict):
                print(f"{k}")
                print_sample(v)
            else: #k in ["data","tree", "t", "data_mask"]:
                if isinstance(v, torch.Tensor):
                    print(f"{k}: {v.size()}, {v.dtype}")
                else:
                    print(f"{k}: {[len(e)for e in v]}")
            # else:
            #     print(f"{k}: {v}")
        print()

    @hydra.main(version_base=None, config_path="configs", config_name="main")
    def test(params):
        params.base_seed = 0
        params.n_gpu_per_node = 1
        params.local_rank = 0
        params.global_rank = 0
        params.num_workers = 4
        params.batch_size = 10
        params.eval_size=200
        params.data.eval_types = -1

        symbol_env = SymbolicEnvironment(params.symbol)
        dataset:dict = get_dataset(params,symbol_env,split="eval")
        print(dataset.__len__())
        loader = {k:cycle(DataLoader(
            v,
            batch_size=params.batch_size_eval,
            num_workers=params.num_workers,
            collate_fn=custom_collate(params.data.max_output_dimension,symbol_env),
            shuffle=True
        )) for k,v in dataset.items()}

        data_iter = iter(loader["burgers"])
        data = next(data_iter)
        print_sample(data)  # (bs, t_num, x_num, x_num, max_output_dimension)
        print_sample(next(data_iter))
        print_sample(next(data_iter))

    test()