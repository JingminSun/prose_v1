from logging import getLogger
import torch
from tabulate import tabulate

from .transformer_wrappers import PROSE_1DPDE
logger = getLogger()


def build_model(params, model_config, data_config, symbol_env):

    modules = {}

    # get model
    name = model_config.name

    if name == "prose":
        # 2to1 prose model
        base_model = PROSE_1DPDE(
            model_config,
            symbol_env,
            data_config
        )


        modules["model"] = base_model
    else:
        assert False, f"Model {name} hasn't been implemented"

    # reload pretrained modules
    if params.reload_model:
        logger.info(f"Reloading modules from {params.reload_model} ...")
        reloaded = torch.load(params.reload_model)
        # new_state_dict = {}
        for k, v in modules.items():
            assert k in reloaded, f"{k} not in save"

            original_keys = reloaded[k].keys()
            transformed_keys = {}

            for key in original_keys:
                new_key = key
                if key.startswith("module."):
                    new_key = key[len("module."):]
                if key.startswith("_orig_mod."):
                    new_key = key[len("_orig_mod."):]
                transformed_keys[new_key] = reloaded[k][key]
            # new_state_dict[k] = transformed_keys
            v.load_state_dict(transformed_keys)


    # log
    for k, v in modules.items():
        logger.info(f"{k}: {v}")
    for k, v in modules.items():
        s = f"Number of parameters ({k}): {sum([p.numel() for p in v.parameters() if p.requires_grad]):,}"
        if hasattr(v, "summary"):
            # for individual components of a wrapper model
            s += v.summary()
        logger.info(s)

    for k, v in modules.items():
        table_data = [(name, str(param.shape), param.requires_grad) for name, param in v.named_parameters()]
        logger.info("\n" + tabulate(table_data, headers=["Parameter Name", "Shape", "Requires Grad"], tablefmt="grid"))
        table_data = [(name, str(param.shape)) for name, param in v.named_parameters() if param.requires_grad]
        logger.info("\n" + tabulate(table_data, headers=["Trainable Parameters", "Shape"], tablefmt="grid"))

    # cuda
    if not params.cpu:
        for v in modules.values():
            v.cuda()

    if params.compile:
        for k, v in modules.items():
            # modules[k] = torch.compile(v, mode="reduce-overhead")
            modules[k] = torch.compile(v)

    return modules
