import sys

# sys.path.append("src/generator")
# sys.path.append("src/symbol_utils")

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
from tqdm import tqdm
import numpy as np

from collections import defaultdict

import h5py


from symbol_utils.environment import SymbolicEnvironment
import torch

from torch.utils.data import Dataset, DataLoader

import hydra

import copy

import io

import json





def zip_dic(lst):
    dico = {}
    for d in lst:
        for k in d:
            if k not in dico:
                dico[k] = []
            dico[k].append(d[k])
    for k in dico:
        if isinstance(dico[k][0], dict):
            dico[k] = zip_dic(dico[k])
    return dico

class generateDataset(Dataset):

    def __init__(self, config, env, size=100000):

        self.config = config

        self.seed = config.seed

        self.size = size

        self.errors = defaultdict(int)

        self.env = env

        self.remaining_data = 0

        self.items = None

        self.rng = None

        self.type = config.data.types

        self.generator = env.generator






    def __len__(self):

        return self.size


    def init_rng(self):

        """

        Initialize random generator for generation.

        """

        if self.rng is not None:

            return

        worker_id = self.get_worker_id()

        self.worker_id = worker_id

        seed = [worker_id, self.seed]

        self.rng = np.random.RandomState(seed)

        print(f"Initialized random generator for worker {worker_id}, with seed {seed} (base seed={self.seed}).")

    def get_worker_id(self):

        worker_info = torch.utils.data.get_worker_info()

        return 0 if worker_info is None else worker_info.id

    def collate_fn(self, elements):
        """
        Collate samples into a batch.
        """
        samples = zip_dic(elements)
        errors = copy.deepcopy(self.errors)
        self.errors = defaultdict(int)
        return samples, errors




    def __getitem__(self, idx):

        if self.remaining_data == 0:
            self.init_rng()

            self.item, errors = self.env.gen_expr(self.rng)
            for error, count in errors.items():
                self.errors[error] += count

            self.remaining_data = len(self.item["data"])



        self.remaining_data -= 1

        data = self.item["data"][-self.remaining_data]



        sample = dict()
        sample["type"] = self.item["type"]
        sample["tree"] = self.item["tree"]
        sample["data"] = data
        sample["tree_encoded"] = self.item["tree_encoded"]

        return sample



    def generate_samples(self, rng):


        item = self.generator.generate_one_sample(rng)

        return item



@hydra.main(version_base=None, config_path="configs/", config_name="generator")
def main(params):

    type = params.data.types

    folder = os.path.join(params.directory, type)

    os.makedirs(folder, exist_ok=True)

    print("file_name",params.file_name)
    if params.file_name is not None:
        export_path_prefix = os.path.join(folder,  type+"_"  + str(params.IC_per_param) +"_" +params.file_name + ".prefix")
        export_path_data = os.path.join(folder, type+"_"  + str(params.IC_per_param) +"_" + params.file_name + "_data.h5")
    else:
        export_path_prefix = os.path.join(folder, type +"_" + str(params.IC_per_param) + ".prefix")
        export_path_data = os.path.join(folder,  type+"_"  + str(params.IC_per_param) +"_data.h5")



    file_handler_prefix = io.open(export_path_prefix, mode="a", encoding="utf-8")


    # set random seed for reproducibility

    if params.seed < 0:

        params.seed = np.random.randint(0, 1000000000)

    total_lines = params.size

    t_num = params.data.t_num

    x_num = params.data.x_num

    dim = getattr(params.data, type).dim

    env = SymbolicEnvironment(params)



    dataset = generateDataset(params, env, size=total_lines)

    dataloader = DataLoader(

        dataset,

        batch_size=params.batch_size,

        num_workers=params.num_workers,

        collate_fn=dataset.collate_fn,

    )

    data_iter = iter(dataloader)


    print(f"Data will be saved in: {os.path.abspath(export_path_data)}")



    data_matrix = np.zeros((total_lines, t_num, x_num , dim), dtype=np.single)


    cur_line = 0



    pbar = tqdm(total=total_lines)

    while cur_line < total_lines:

        # get data

        try:

            samples,_ = next(data_iter)

        except StopIteration:

            print("Reached end of dataloader, restart...")

            data_iter = iter(dataloader)

            samples,_ = next(dataloader)



        cur_size = len(samples["data"])

        if cur_size > 0:

            for i in range(cur_size):
                outputs = dict()

                outputs["type"] = samples["type"][i]

                outputs["tree_encoded"] = samples["tree_encoded"][i]

                cur_data = samples["data"][i]

                cur_data = cur_data.numpy().reshape(params.data.t_num, env.generator.x_grid_size, dim)

                data_matrix[cur_line, ..., :dim] = cur_data


                outputs["dim"] = dim

                cur_line += 1

                file_handler_prefix.write(json.dumps(outputs) + "\n")
                file_handler_prefix.flush()



                if cur_line >= total_lines:

                    cur_size = i + 1

                    break



            pbar.update(cur_size)



    ax = tuple(range(0, data_matrix.ndim - 1))

    mean = np.mean(data_matrix, axis=ax, keepdims=True, dtype=np.float64).astype(np.single)

    std = np.std(data_matrix, axis=ax, keepdims=True, dtype=np.float64).astype(np.single)



    with h5py.File(export_path_data, "w") as hf:

        hf.create_dataset("data", data=data_matrix, maxshape=(None, t_num, x_num, dim))

        hf.create_dataset("seed", data=params.seed)

        hf.create_dataset("mean", data=mean)

        hf.create_dataset("std", data=std)



    print("Done.")


if __name__ == "__main__":


    main()
