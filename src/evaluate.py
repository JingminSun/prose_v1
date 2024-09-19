import os
import numpy as np
from logging import getLogger
from collections import defaultdict
import copy
import sympy as sy
from generator.data_gen_NLE import burgers_f
import scipy
from jax import numpy as jnp

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
    "_r2": "r2"
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
        self.space_dim = self.params.data.max_input_dimension
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

        t_grid = np.linspace( self.params.data.t_range[0], self.params.data.t_range[1], self.params.data.t_num)
        x_grid = np.linspace(self.params.data.x_range[0], self.params.data.x_range[1], self.params.data.x_num)
        coeff = np.random.uniform(-5, 5, size=(8, self.params.data.max_input_dimension))
        # Create mesh grids
        T, X = np.meshgrid(t_grid, x_grid, indexing="ij")
        self.T, self.X = T,X
        x,t = sy.symbols('x t')
        u = sy.Function('u_0')(x, t)
        tens_poly = (coeff[0, 0] + coeff[1, 0] * t + coeff[2, 0] * t ** 2) * (
                coeff[3, 0] + coeff[4, 0] * x + coeff[5, 0] * x ** 2 + coeff[6, 0] * x ** 3 + coeff[
            7, 0] * x ** 4)
        self.sympy_sybol = {
            "u": u,
            "x":x,
            "t":t,
            "tens_poly":tens_poly
        }
        input_points = np.zeros((self.params.data.max_input_dimension, self.params.data.t_num, self.params.data.x_num, 8))
        for i in range(self.params.data.max_input_dimension):
            input_points[i, :, :, 0] = (coeff[0, i] + coeff[1, i] * T + coeff[2, i] * T ** 2) * (
                    coeff[3, i] + coeff[4, i] * X + coeff[5, i] * X ** 2 + coeff[6, i] * X ** 3 + coeff[7, i] * X ** 4
            )
            input_points[i, :, :, 1] = (coeff[1, i] + 2 * coeff[2, i] * T) * (
                    coeff[3, i] + coeff[4, i] * X + coeff[5, i] * X ** 2 + coeff[6, i] * X ** 3 + coeff[7, i] * X ** 4
            )
            # ut
            input_points[i, :, :, 2] = (2 * coeff[2, i]) * (
                    coeff[3, i] + coeff[4, i] * X + coeff[5, i] * X ** 2 + coeff[6, i] * X ** 3 + coeff[7, i] * X ** 4
            )
            # utt
            input_points[i, :, :, 3] = (coeff[0, i] + coeff[1, i] * T + coeff[2, i] * T ** 2) * (
                    coeff[4, i] + 2 * coeff[5, i] * X + 3 * coeff[6, i] * X ** 2 + 4 * coeff[7, i] * X ** 3
            )  # ux
            input_points[i, :, :, 4] = (coeff[0, i] + coeff[1, i] * T + coeff[2, i] * T ** 2) * (
                    2 * coeff[5, i] + 6 * coeff[6, i] * X + 12 * coeff[7, i] * X ** 2
            )  # uxx
            input_points[i, :, :, 5] = (coeff[0, i] + coeff[1, i] * T + coeff[2, i] * T ** 2) * (
                    6 * coeff[6, i] + 24 * coeff[7, i] * X
            )  # uxxx
            input_points[i, :, :, 6] = (coeff[0, i] + coeff[1, i] * T + coeff[2, i] * T ** 2) * (
                    24 * coeff[7, i]
            )  # uxxxx
            input_points[i, :, :, 7] = X
        self.input_points = input_points
    @torch.no_grad()
    def evaluate(self):

        params = self.params

        model = self.modules["model"]
        model.eval()

        if params.print_outputs:
            save_folder = os.path.join(params.eval_dump_path, "figures/")
            if not os.path.isdir(save_folder):
                os.makedirs(save_folder)

        if params.log_eval_plots > 0 or params.plot_worst:
            plot_folder = os.path.join(params.eval_dump_path, f"epoch_{self.trainer.epoch}_{self.params.local_rank}")
            if not os.path.isdir(plot_folder):
                os.makedirs(plot_folder)

        all_results = {}

        for type, loader in self.dataloaders.items():
            eval_size = 0
            num_plotted = 0
            text_valid = 0
            results = defaultdict(list)

            worst_case_pred = 0

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
                        symbol_input_padding_mask=dict["symbol_input_mask"]
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

                    raw_data = dict["data_input"] * (dict["std"] + eps) + dict["mean"]


                cur_result = compute_metrics(
                    data_output, data_label, metrics=params.validation_metrics_print, batched=True
                )

                if self.params.plot_worst and np.max(cur_result["_l2_error"]) > worst_case_pred:
                    worst_case_pred = np.max(cur_result["_l2_error"])
                    worst_index = np.argmax(cur_result["_l2_error"])

                    # plot_title = "Type {} |  $L^2$ error {:.4f} % ".format(type,cur_result[ "_l2_error"][worst_index] * 100)
                    path = plot_1d_pde(
                        data_output[worst_index].float().numpy(force=True),
                        None,
                        samples["t"][worst_index],
                        samples["x"][worst_index],
                        samples["data"][worst_index].numpy(force=True),
                        params.input_len,
                        None,
                        filename=f"{type}_plot_worst",
                        folder=plot_folder,
                        dim=params.data[type].dim,
                        input_step=params.input_step,
                        output_step=params.output_step,
                        output_start=params.input_len if params.output_step == 1 else params.input_len + 1,
                        diff_plot=False,
                        input_plot=False
                    )


                if not params.model.no_text_decoder:
                    symbol_output = output_dict["symbol_generated"]
                    min_loss_save = []
                    min_loss_save_ref = []
                    valid_ref_gen_l2 = []
                    valid_orig_gen_l2 = []

                    symbol_output = symbol_output.unsqueeze(-1)
                    symbol_output = (
                        symbol_output.transpose(1, 2).cpu().tolist()
                    )  # (bs, 1, text_len)
                    symbol_output = [
                        list(
                            filter(
                                lambda x: x is not None,
                                [
                                    self.symbol_env.idx_to_infix(hyp[1:-1], is_float=False, str_array=False)
                                    for hyp in symbol_output[i]
                                ],
                            )
                        )
                        for i in range(bs)
                    ]  # nested list of shape (bs, 1), some inner lists are possibly empty

                    for i in range(bs):
                        tree_list = symbol_output[i]
                        label_outputs = None
                        valid_loss = []
                        valid_refined_loss=[]

                        for tree in tree_list:
                            if self.params.symbol.use_sympy:
                                try:
                                    expr = tree[0]
                                    expr_copy = expr
                                    equation = sy.sympify(expr)
                                    expr = equation.subs(self.sympy_sybol["u"],
                                                         self.sympy_sybol["tens_poly"])
                                    eval_expr = sy.lambdify(
                                        [self.sympy_sybol["x"], self.sympy_sybol["t"]], expr.doit(),
                                        "numpy")
                                    generated_outputs = eval_expr(self.X, self.T)
                                    if self.params.symbol.refine:
                                        init_variable, fluxx, viscous = self.init_param(expr_copy)
                                        refined_expr_org, generated_data_ref = self.refinement(expr_copy,raw_data[i],init_variable, fluxx, viscous )

                                        equation = sy.sympify(refined_expr_org)
                                        refined_expr = equation.subs(self.sympy_sybol["u"], self.sympy_sybol["tens_poly"])
                                        eval_expr = sy.lambdify([self.sympy_sybol["x"], self.sympy_sybol["t"]], refined_expr.doit(), "numpy")
                                        generate_refined_output = eval_expr(self.X, self.T)
                                except:
                                    continue
                            else:
                                try:
                                    generated_outputs = tree.val(self.input_points, self.space_dim)

                                except:
                                    continue

                            if label_outputs is None:
                                if self.params.symbol.use_sympy:
                                    tree_expr =  str(dict["tree_structure"][i])

                                    original_expr = sy.sympify(tree_expr)[0]
                                    original_expr = original_expr.subs(self.sympy_sybol["u"], self.sympy_sybol["tens_poly"])
                                    eval_expr = sy.lambdify([self.sympy_sybol["x"], self.sympy_sybol["t"]], original_expr.doit(), "numpy")
                                    label_outputs = eval_expr(self.X, self.T)
                                else:
                                    label_outputs = dict["tree_structure"][i].val(self.input_points, self.space_dim)
                                assert np.isfinite(label_outputs).all()
                            try:
                                if np.isfinite(generated_outputs).all():
                                    error = np.sqrt(np.sum((generated_outputs - label_outputs) ** 2)/ (np.sum(label_outputs ** 2) + eps))
                                    assert error< 100
                                    valid_loss.append(error)
                                    if self.params.symbol.refine:
                                        error_refined = np.sqrt(
                                            np.sum((generate_refined_output - label_outputs) ** 2) / (
                                                        np.sum(label_outputs ** 2) + eps))
                                        valid_refined_loss.append(error_refined)
                            except:
                                continue
                        if len(valid_loss) > 0:
                            # generated tree is valid, compute other metrics
                            min_loss = min(valid_loss)
                            text_valid += 1
                            min_loss_save.append(min_loss)
                            if self.params.symbol.refine:
                                min_loss_ref = min(valid_refined_loss)
                                min_loss_save_ref.append(min_loss_ref)
                            if self.params.symbol.use_sympy:

                                if self.params.symbol.refine:
                                    original_exp = self.generated_symbol(self.params,
                                                                         raw_data[i][0, :,
                                                                         0].cpu().numpy(),
                                                                         init_variable, fluxx=fluxx,
                                                                         viscous=viscous)
                                    result_org = compute_metrics(
                                        to_cuda(torch.from_numpy(
                                            original_exp.reshape(data_label[i].size()))),
                                        data_label[i],
                                        metrics="_l2_error", batched=False
                                    )

                                    valid_orig_gen_l2.append(result_org["_l2_error"])
                                    result_ref = compute_metrics(
                                        to_cuda(torch.from_numpy(
                                            generated_data_ref.reshape(data_label[i].size()))),
                                        data_label[i],
                                        metrics="_l2_error", batched=False
                                    )
                                    valid_ref_gen_l2.append(result_ref["_l2_error"])
                            if params.print_outputs:
                                logger.info(
                                    "[{}] Text loss: {:.4f}".format(
                                        i, min_loss
                                    )
                                )
                                if self.params.symbol.refine:
                                    logger.info(
                                        "Refined loss: {:.4f}".format(
                                             min_loss_ref
                                        )
                                    )
                                logger.info("Input:     {}".format(dict["input_structure"][i]))
                                logger.info("Target:    {}".format(dict["tree_structure"][i]))
                                try:
                                    if self.params.symbol.refine:
                                        logger.info("Generated: {}".format(tree_list[0]))
                                        logger.info("Refined: {}\n".format(refined_expr_org))
                                    else:
                                        logger.info("Generated: {}\n".format(tree_list[0]))

                                except:
                                    # logger.info("Generated: {}\n".format(tree_list[1]))
                                    pass
                    results["text_loss"].extend(min_loss_save)
                    if self.params.symbol.refine:
                        results["text_loss_ref"].extend(min_loss_save_ref)
                        if self.params.symbol.use_sympy:
                            results["orig_gen_l2"].extend(valid_orig_gen_l2)
                            results["ref_gen_l2"].extend(valid_ref_gen_l2)
                for k in cur_result.keys():
                    keys = k
                    if k == "_r2":
                        results[keys].append(cur_result[k])
                    else:
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
                            data_output[i] if isinstance(data_output, np.ndarray) else data_output[i] .float().numpy(force=True),
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

                    if isinstance(data_output, np.ndarray):
                        # already converted to numpy
                        output_zero_shot = data_output[0]
                        cur_data = data_all[0]
                    else:
                        output_zero_shot = data_output[0].float().numpy(force=True)
                        cur_data = samples["data"][0].numpy(force=True)

                    index = idx * params.batch_size_eval
                    plot_title = "Type {} |  $L^2$ error {:.4f} % ".format(type,cur_result[ "_l2_error"][ 0] * 100)
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
                if k == "_r2":
                    results[k] = np.mean(np.array(v))
                else:
                    results[k] = np.sum(np.array(v))
            results["size"] = eval_size
            if not params.model.no_text_decoder:
                results["text_valid"] = text_valid
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
        total_valid = 0
        results_per_type = {}
        stats = defaultdict(float)
        for type, results in all_results.items():
            res_mean_type = {}
            for k, v in results.items():
                if k == "size":
                    res_mean_type[k] = v
                    total_size += v
                elif k=="_r2":
                    res_mean_type[k] = v
                    stats[k] += v / len(self.dataloaders)
                elif k == "_mse":
                    # rescale mse due to padding dimensions
                    ratio = self.params.data.max_output_dimension / self.params.data[type].dim
                    res_mean_type[k] = v / results["size"] * ratio
                    stats[k] += v * ratio
                elif k == "_rmse":
                    ratio = (self.params.data.max_output_dimension / self.params.data[type].dim) ** 0.5
                    res_mean_type[k] = v / results["size"] * ratio
                    stats[k] += v * ratio
                elif k.startswith("text_loss") or k== "orig_gen_l2" or k=="ref_gen_l2":
                    res_mean_type[k] = v / max(results["text_valid"],1)
                    stats[k] += v
                elif k == "text_valid":
                    res_mean_type["valid_fraction"] = v / results["size"]
                    total_valid += v
                    stats["valid_fraction"] += v
                else:
                    res_mean_type[k] = v / results["size"]
                    stats[k] += v
            results_per_type[type] = res_mean_type
        stats = {k: v if k=="_r2" else v/max(total_valid,1) if k.startswith("text_loss") else v / total_size for k, v in stats.items()}

        # report metrics per equation type as a table
        if self.params.model.no_text_decoder:
            headers = ["type", "dim", "size", "data_loss"] + [k for k in self.validation_metrics]
        else:
            headers = ["type", "dim", "size", "data_loss"] + [k for k in self.validation_metrics] + [ "text_loss","valid_fraction"]
            if self.params.symbol.use_sympy:
                if self.params.symbol.refine:
                    headers = headers + ["orig_gen_l2","text_loss_ref" ,"ref_gen_l2"]
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

    def init_param(self,expr):
        p = self.params
        x, t = sy.symbols('x t')
        u = sy.Function('u_0')(x, t)


        try:
            init_flux = float(expr.coeff(u * sy.diff(u, x)))
            assert init_flux != 0
            fluxx = "quadratic"
        except:
            try:
                init_flux = float(expr.coeff(u ** 2 * sy.diff(u, x)))
                assert init_flux != 0
                fluxx = "cubic"
            except:
                try:
                    init_flux = float(expr.coeff(sy.cos(u) * sy.diff(u, x)))
                    assert init_flux != 0
                    fluxx = "sin"
                except:
                    raise NotImplementedError
        # if type == "inviscid_burgers":
        #     try:
        #         expr = sy.sympify(expr)
        #         expr = expr.doit()
        #         init_variable = float(expr.coeff(u * sy.diff(u, x)))
        #         fluxx = "quadratic"
        #         viscous = False
        #     except:
        #         raise "Initial Value not found"
        # elif type == "inviscid_conservation_cubicflux":
        #     try:
        #         expr = sy.sympify(expr)
        #         expr = expr.doit()
        #         init_variable = float(expr.coeff(u ** 2 * sy.diff(u, x)))
        #         fluxx = "cubic"
        #         viscous = False
        #     except:
        #         raise "Initial Value not found"
        # elif type == "inviscid_conservation_sinflux":
        #     try:
        #         expr = sy.sympify(expr)
        #         expr = expr.doit()
        #         init_variable = float(expr.coeff(sy.cos(u) * sy.diff(u, x)))
        #         fluxx = "sin"
        #         viscous = False
        #     except:
        #         raise "Initial Value not found"
        # elif type == "burgers":
        #     try:
        #         expr = sy.sympify(expr)
        #         expr = expr.doit()
        #         init_flux = expr.coeff(u * sy.diff(u, x))
        #         init_diff = -expr.coeff(sy.diff(u, (x, 2)))
        #         init_variable = np.array([float(init_flux), float(init_diff)])
        #         fluxx = "quadratic"
        #         viscous = True
        #     except:
        #         raise "Initial Value not found"
        # elif type == "conservation_cubicflux":
        #     try:
        #         expr = sy.sympify(expr)
        #         expr = expr.doit()
        #         init_flux = expr.coeff(u ** 2 * sy.diff(u, x))
        #         init_diff = -expr.coeff(sy.diff(u, (x, 2)))
        #         init_variable = np.array([float(init_flux), float(init_diff)])
        #         fluxx = "cubic"
        #         viscous = True
        #     except:
        #         raise "Initial Value not found"
        # elif type == "conservation_sinflux":
        #     try:
        #         expr = sy.sympify(expr)
        #         expr = expr.doit()
        #         init_flux = expr.coeff(sy.cos(u) * sy.diff(u, x))
        #         init_diff = -expr.coeff(sy.diff(u, (x, 2)))
        #         init_variable = np.array([float(init_flux), float(init_diff)])
        #         fluxx = "sin"
        #         viscous = True
        #     except:
        #         raise "Initial Value not found"
        # else:
        #     raise NotImplementedError
        try:
            init_diff = float(-expr.coeff(sy.diff(u, (x, 2))))
            assert init_diff != 0
            init_variable = np.array([init_flux,init_diff])
            viscous = True
        except:
            init_variable = init_flux
            viscous = False
        return init_variable,fluxx, viscous

    def refinement(self, expr, data_input,init_variable, fluxx, viscous ):
        p = self.params
        # expr, init_variable, fluxx, viscous = self.init_param(expr)
        # Number of particles
        M = 500
        T = 10

        obs_noise =  np.linalg.norm(data_input[0].cpu().numpy())/20

        # STD of Process noise
        process_noise = 0.00001

        # Define the initial distribution of particles for alpha (uniform distribution)
        def initial_distribution(init_alpha):
            if not isinstance(init_alpha,np.ndarray):
                init_alpha = np.array(init_alpha, dtype=float)

            return np.random.uniform(0.9 * init_alpha, 1.1 * init_alpha, (M,) + init_alpha.shape)

        # Propagate particles with noise
        def propagate_particles(particles, process_noise):
            noise = np.random.normal(0, process_noise, particles.shape)
            return particles + noise

        # Resample particles based on their weights using systematic resampling
        def resample(particles, weights):
            positions = (np.arange(M) + np.random.uniform(0, 1)) / M
            indexes = np.zeros(M, 'i')
            cumulative_sum = np.cumsum(weights)
            i, j = 0, 0
            while i < M:
                if positions[i] < cumulative_sum[j]:
                    indexes[i] = j
                    i += 1
                else:
                    j += 1
            return particles[indexes]

        def compute_weights(particles, observation, u_prev):
            weights = np.zeros(M)
            for i in range(M):
                u_pred = self.update(p, u_prev, particles[i], fluxx=fluxx,viscous=viscous)
                weights[i] = np.exp(-0.5 * np.sum((observation - u_pred) ** 2) / obs_noise ** 2)
            return weights / np.sum(weights)

        particles = initial_distribution(init_variable)

        observations = data_input[:, :,0].cpu().numpy()

        for t in range(T):
            particles = propagate_particles(particles, process_noise)

            weights = compute_weights(particles, observations[t+1,:],observations[t, :])

            particles = resample(particles, weights)

        # Estimate the state
        refined_value = np.mean(particles,axis=0)
        refined_exp = self.generated_symbol(p,  observations[0,:], refined_value, fluxx=fluxx, viscous=viscous)
        if viscous:
            expr_new = expr.subs(init_variable[0], refined_value[0])
            expr_new = expr_new.subs(init_variable[1],refined_value[1])
            return expr_new,refined_exp
        else:
            return expr.subs(init_variable, refined_value),refined_exp

    def update(self, p, u_prev, particles, fluxx="quadratic", viscous=True):
        GivenIC = u_prev.reshape(1,p.data.x_num)
        if viscous:
            eps =particles[1] * np.pi
            k = particles[0]
        else:
            eps = 0
            k = particles
        uu = burgers_f(
            p.data.x_range[1],
            p.data.x_range[0],
            p.data.x_num,
            0.0,
            (p.data.t_range[1] / p.data.t_num) * p.input_step,
            p.data.t_range[1] / p.data.t_num,
            p.input_step + 1,
            # p.data.t_num,
            0.4,
            1,
            1,
            np.random.randint(100000),
            eps,
            k,
            viscous=viscous,
            fluxx=fluxx,
            GivenIC=GivenIC,
            mode="periodic"
        )
        return  np.array(uu[0,0,-1, :])

    def generated_symbol(self, p, IC, param, fluxx="quadratic", viscous=True):
        GivenIC = IC.reshape(1,p.data.x_num)
        if viscous:
            eps =param[1] * np.pi
            k = param[0]
        else:
            eps = 0
            k = param
        uu = burgers_f(
            p.data.x_range[1],
            p.data.x_range[0],
            p.data.x_num,
           0.0,
            p.data.t_range[1],
            p.data.t_range[1] / p.data.t_num,
            p.data.t_num,
            0.4,
            1,
            1,
            np.random.randint(100000),
            eps,
            k,
            viscous=viscous,
            fluxx=fluxx,
            GivenIC=GivenIC,
            mode="periodic"
        )
        output_start = self.params.input_len if  self.params.output_step == 1 else  self.params.input_len + 1
        return  np.array(uu[0,0,output_start::self.params.output_step, :])
