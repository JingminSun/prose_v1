from collections import defaultdict
import numpy as np
import scipy
import torch
from sklearn.metrics import r2_score

def compute_metrics(output, label, metrics="_mse", batched=False):
    """
    output, label: torch tensor (seq_len, ... output_dim) if batched=False
                                (bs, seq_len, ..., output_dim) if batched=True
    """
    assert label.shape == output.shape, f"shape mismatch: output: {output.shape}, label: {label.shape}"

    results = dict()
    seq_len = output.size(1) if batched else output.size(0)
    eps = 1e-7

    if metrics == "":
        return results

    for metric in metrics.split(","):
        if metric == "_mse":
            if batched:
                mse = ((output - label) ** 2).flatten(1).mean(1).tolist()  # (bs, )
            else:
                mse = ((output - label) ** 2).mean().item()
            results[metric] = mse

        elif metric == "_rmse":
            if batched:
                mse = torch.sqrt(((output - label) ** 2).flatten(1).mean(1)).tolist()  # (bs, )
            else:
                mse = torch.sqrt(((output - label) ** 2).mean()).item()
            results[metric] = mse

        elif metric == "_r2":
            if batched:
                bs = output.shape[0]
                output_reshape = output.reshape(bs,-1)
                label_reshape = label.reshape(bs,-1)
                r2 =  r2_score(output_reshape.cpu().numpy(),label_reshape.cpu().numpy())
            else:
                raise "r2 error has to be computed in a batch"
            results[metric] = r2


        elif metric.startswith("_l2_error"):
            if batched:
                if metric == "_l2_error":
                    predicted, true = output, label

                elif metric == "_l2_error_first_half":
                    predicted = output[:, : (seq_len // 2)]
                    true = label[:, : (seq_len // 2)]

                elif metric == "_l2_error_second_half":
                    predicted = output[:, (seq_len // 2) :]
                    true = label[:, (seq_len // 2) :]

                elif metric == "_l2_error_mean_prediction":
                    predicted = torch.mean(label, dim=0,keepdim=True)
                    true = label

                else:
                    assert False, f"Unknown metric: {metric}"

                error = torch.sqrt(((true - predicted) ** 2).flatten(1).sum(1))
                scale = eps + torch.sqrt((true**2).flatten(1).sum(1))
                error = torch.div(error, scale).tolist()

            else:
                if metric == "_l2_error":
                    predicted, true = output, label

                elif metric == "_l2_error_first_half":
                    predicted = output[: (seq_len // 2)]
                    true = label[: (seq_len // 2)]

                elif metric == "_l2_error_second_half":
                    predicted = output[(seq_len // 2) :]
                    true = label[(seq_len // 2) :]
                elif metric == "_l2_error_mean_prediction":
                    assert False, "Cannot get mean prediction given one data"


                else:
                    assert False, f"Unknown metric: {metric}"

                error = torch.sqrt(((true - predicted) ** 2).sum()).item()
                scale = eps + torch.sqrt((true**2).sum()).item()
                error = error / scale

            results[metric] = error

    return results
