PROSE(1DPDE) version 1.0

## Data Generation: See detailed instruction in script/gen_data.sh
    bash scripts/gen_data.sh
The data is by default saving to "your_directory/type_name/(type_name)_(IC_per_params).prefix" for symbol part 
                        and "your_directory/type_name/(type_name)_(IC_per_params)_data.h5" for data part

If some specific data need to be generated, change "file_name=specific_name", samples are given in script/gen_data.sh 
and data will be stored in "your_directory/type_name/(type_name)_(IC_per_params)_(file_name).prefix" 
                        and "your_directory/type_name/(type_name)_(IC_per_params)_(file_name)_data.h5" 


## Run the code

For test runs:

    bash scripts/run.sh

For the experiments in the paper  [Time-Series Forecasting, Knowledge Distillation, and Refinement within a Multimodal PDE Foundation Model](https://arxiv.org/abs/2409.11609):
    
    bash scripts/sympy.sh

For the experiments in the paper  [Towards a Foundation Model for Partial Differential Equations: Multi-Operator Learning and Extrapolation](https://arxiv.org/abs/2404.12355):

    bash scripts/pde_experiments.sh

### Data

Just specify your ``data.train_types`` and ``data.eval_types`` and if some specific_name needed, 
add ``data.eval_data=specific_name`` and ``data.train_data=specific_name``

Num of training and Num of evaluation:

``data.train_size`` is the num of training sample in total, and we subsample ``data.train_size_get`` for real training.
Same for ``data.eval_size``  and  ``data.eval_size_get`` 

We use ``skip = data.train_size`` (~line 54 of ``evaluator.py``) to avoid sampling the same data for training and evaluation.
However, if you want to save space/ your evaluation dataset and training dataset are different, you can comment that out.


Note you may set ``model.data_decoder.full_tx=false`` to run with a larger batch_size

### Modes

In data configuration, you can enable/disable the skeleton tree input by ``symbol.use_skeleton=True/False``

In model configuration, you can include/exclude the text (symbol) encoder/decoder by ``model.no_text_encoder=True/False`` 
and ``model.no_text_decoder=True/False`` , the default setting is text encoder but no text decoder.

## Citation

If you find this code useful, please consider citing:

```
@article{sun2025towards,
  title = {Towards a foundation model for partial differential equations: Multioperator learning and extrapolation},
  author = {Sun, Jingmin and Liu, Yuxuan and Zhang, Zecheng and Schaeffer, Hayden},
  journal = {Phys. Rev. E},
  volume = {111},
  issue = {3},
  pages = {035304},
  numpages = {18},
  year = {2025},
  month = {Mar},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevE.111.035304},
  url = {https://link.aps.org/doi/10.1103/PhysRevE.111.035304}
}


@article{jollie2025time,
	author  = {Derek  Jollie and Jingmin  Sun and Zecheng  Zhang and Hayden Schaeffer},
	title   = {TIME-SERIES FORECASTING AND REFINEMENT WITHIN A MULTIMODAL PDE FOUNDATION MODEL},
	journal = {Journal of Machine Learning for Modeling and Computing},
	issn    = {2689-3967},
	year    = {2025},
	volume  = {6},
	number  = {2},
	pages   = {77--89}
}

```
