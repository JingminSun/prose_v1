import os
import numpy as np
from logging import getLogger
from collections import defaultdict
import copy

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import get_dataset
from utils.misc import sync_tensor
from utils.metrics import compute_metrics
from utils.plot import plot_2d_pde, plot_1d_pde
from data_utils.collate import custom_collate
from tabulate import tabulate
from utils.misc import to_cuda
import wandb
# np.seterr(all="raise")
np.seterr(divide="raise", under="ignore", over="raise", invalid="raise")

logger = getLogger()

metric_to_header = {
    "_l2_error": "rel l2",
    "_mse": "mse",
    "_rmse": "rmse",
    "_l2_error_first_half": "rel l2 1st_half",
    "_l2_error_second_half": "rel l2 2nd_half",
}


def data_loss_fn(data_output, data_label, data_mask, weight=None):
    # copy of trainer data_loss_fn, by batch
    loss = F.mse_loss(data_output, data_label, reduction="none")
    pred_mask = data_mask.expand_as(loss)
    if weight is None:
        # no re-weighting, loss is just regular MSE
        loss = (loss * pred_mask).flatten(1).sum(1) / (pred_mask.flatten(1).sum(1))
    else:
        # reweight by weight
        weight = weight.expand_as(loss)
        loss = ((loss * pred_mask) * weight).flatten(1).sum(1)
    return loss.tolist()  # (bs, )


class Evaluator(object):

    def __init__(self, trainer, symbol_env):
        self.trainer = trainer
        self.modules = trainer.modules
        self.params = trainer.params
        self.symbol_env = symbol_env

        self.skip = self.params.train_size
        self.datasets: dict = get_dataset(self.params, self.symbol_env, split="eval", skip = self.skip)
        self.dataloaders = {
            k: DataLoader(
                v,
                batch_size=self.params.batch_size_eval,
                num_workers=self.params.num_workers,
                # num_workers=1,
                # pin_memory=True,
                collate_fn=custom_collate(self.params.data.max_output_dimension,symbol_env),
            )
            for k, v in self.datasets.items()
        }
        self.iteration = {
            k: iter(self.dataloaders[k]) for k in self.dataloaders.keys()
        }

        self.types = self.datasets.keys()

        self.validation_metrics = self.params.validation_metrics_print.split(",")

    @torch.no_grad()
    def evaluate(self):

        params = self.params

        model = self.modules["model"]
        model.eval()

        if params.print_outputs:
            save_folder = os.path.join(params.eval_dump_path, "figures/")
            if not os.path.isdir(save_folder):
                os.makedirs(save_folder)

        if params.log_eval_plots > 0:
            plot_folder = os.path.join(params.eval_dump_path, f"epoch_{self.trainer.epoch}_{self.params.local_rank}")
            if not os.path.isdir(plot_folder):
                os.makedirs(plot_folder)

        all_results = {}

        for type, loader in self.dataloaders.items():
            eval_size = 0
            num_plotted = 0
            results = defaultdict(list)

            for idx, samples in enumerate(loader):

                bs = len(samples["data"])
                eval_size += bs
                dict = (
                    self.trainer.prepare_data(samples,train=False)
                )

                data_label = dict["data_label"]
                with torch.cuda.amp.autocast(enabled=bool(params.amp), dtype=torch.bfloat16):
                    output_dict = model(
                        "generate",
                        data_input=dict["data_input"],
                        input_times=dict["input_times"][..., None],
                        output_times=dict["output_times"][..., None],
                        symbol_input=dict["symbol_input"],
                        query_space_grid=dict["spatial_grid"][..., None] if params.model.data_decoder.full_tx else None,
                        symbol_padding_mask=dict["symbol_mask"]
                    )  # (bs, output_len, x_num, x_num, data_dim)
                    data_output = output_dict["data_output"]
                    data_output = data_output* dict["data_mask"]
                    data_loss= data_loss_fn(data_output, data_label,
                                                       dict["data_mask"], dict["loss_weight"])
                    results["data_loss"].extend(data_loss)



                if self.params.normalize:
                    # denormalize data
                    eps = 1e-6

                    data_output = data_output * (dict["std"] + eps) + dict["mean"]
                    data_label = dict["data_label"] * (dict["std"] + eps) + dict["mean"]


                cur_result = compute_metrics(
                    data_output, data_label, metrics=params.validation_metrics_print, batched=True
                )

                for k in cur_result.keys():
                    keys = k
                    results[keys].extend(cur_result[k])
                if params.print_outputs:
                    # plot all outputs

                    data_output = data_output.float().numpy(
                        force=True)  # (bs, output_len//output_step, x_num,data_dim)
                    data_all = samples["data"].numpy(
                        force=True)  # (bs, input_len + output_len, x_num,  data_dim)
                    for i in range(bs):
                        index = idx * params.batch_size_eval + i
                        plot_title = "Type {} | Idx {} | zero {:.4f}".format(type, index,
                                                                        cur_result["_l2_error"][i])


                        plot_1d_pde(
                            data_loss[i],
                            None,
                            samples["t"][i],
                            samples["x"][i],
                            data_all[i],
                            params.input_len,
                            plot_title,
                            filename=f"{type}_plot_{index}",
                            folder=save_folder,
                            dim=params.data[type].dim,
                            input_step = params.input_step,
                            output_step = params.output_step,
                            output_start = params.input_len if params.output_step == 1 else params.input_len + 1
                        )

                if params.log_eval_plots > 0 and num_plotted < params.log_eval_plots:
                    # only plot the first element

                    output_zero_shot = data_output[0]
                    cur_data = data_all[0]

                    index = idx * params.batch_size_eval
                    plot_title = "Type {} | Idx {} | zero {:.4f} ".format(type,index,cur_result[ "_l2_error"][ 0])
                    path = plot_1d_pde(
                        output_zero_shot,
                        None,
                        samples["t"][0],
                        samples["x"][0],
                        cur_data,
                        params.input_len,
                        plot_title,
                        filename=f"{type}_plot_{index}",
                        folder=plot_folder,
                        dim=params.data[type].dim,
                        input_step=params.input_step,
                        output_step=params.output_step,
                        output_start=params.input_len if params.output_step == 1 else params.input_len + 1
                    )

                    if params.use_wandb:
                        wandb.log(
                            {"val": {"epoch": self.trainer.epoch,
                                     f"{type}_plot_{num_plotted}": wandb.Image(path)}}
                        )

                    num_plotted += 1

                if params.eval_size > 0 and eval_size >= params.eval_size:
                    break

            for k, v in results.items():
                results[k] = np.sum(np.array(v))
            results["size"] = eval_size
            all_results[type] = results

        if params.multi_gpu:
            # sync results on all gpus
            sorted_keys = None
            for type, results in all_results.items():

                if sorted_keys is None:
                    sorted_keys = sorted(results.keys())

                stats = torch.Tensor([results[k] for k in sorted_keys])
                stats = sync_tensor(stats)
                results = {k: stats[i].item() for i, k in enumerate(sorted_keys)}
                results["size"] = int(results["size"])
                all_results[type] = results

        # aggregate results and compute averages

        total_size = 0
        results_per_type = {}
        stats = defaultdict(float)
        for type, results in all_results.items():
            res_mean_type = {}
            for k, v in results.items():
                if k == "size":
                    res_mean_type[k] = v
                    total_size += v
                elif k == "_mse":
                    # rescale mse due to padding dimensions
                    ratio = self.params.data.max_output_dimension / self.params.data[type].dim
                    res_mean_type[k] = v / results["size"] * ratio
                    stats[k] += v * ratio
                elif k == "_rmse":
                    ratio = (self.params.data.max_output_dimension / self.params.data[type].dim) ** 0.5
                    res_mean_type[k] = v / results["size"] * ratio
                    stats[k] += v * ratio
                else:
                    res_mean_type[k] = v / results["size"]
                    stats[k] += v
            results_per_type[type] = res_mean_type
        stats = {k: v / total_size for k, v in stats.items()}

        # report metrics per equation type as a table
        headers = ["type", "dim", "size", "data_loss"] + [k for k in self.validation_metrics]
        table = []
        for type, results in results_per_type.items():
            row = [type, self.params.data[type].dim]
            for k in headers[2:]:
                row.append(results[k])
            table.append(row)

        headers = list(map(lambda s: metric_to_header[s] if s in metric_to_header else s, headers))
        logger.info(
            "Evaluation Stats (total size = {})\n{}".format(
                total_size, tabulate(table, headers=headers, tablefmt="grid")
            )
        )

        return stats, results_per_type


