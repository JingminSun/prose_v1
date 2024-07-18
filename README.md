PROSE(1DPDE) version 1

## Data Generation: See detailed instruction in script/gen_data.sh
    bash scripts/gen_data.sh
The data is by default saving to "your_directory/type_name/(type_name)_(IC_per_params).prefix" for symbol part 
                        and "your_directory/type_name/(type_name)_(IC_per_params)_data.h5" for data part

If some specific data need to be generated, change "file_name=specific_name", samples are given in script/gen_data.sh 
and data will be stored in "your_directory/type_name/(type_name)_(IC_per_params)_(file_name).prefix" 
                        and "your_directory/type_name/(type_name)_(IC_per_params)_(file_name)_data.h5" 


## Run the code: Samples in scripts/run.sh

    bash scripts/run.sh
### Data

Just specify your ``data.train_types`` and ``data.eval_types`` and if some specific_name needed, 
add ``data.eval_data=specific_name`` and ``data.train_data=specific_name``

Num of training and Num of evaluation:

``data.train_size`` is the num of training sample in total, and we subsample ``data.train_size_get`` for real training.
Same for ``data.eval_size``  and  ``data.eval_size_get`` 

We use ``skip = data.train_size`` (~line 54 of ``evaluator.py``) to avoid sampling the same data for training and evaluation.
However, if you want to save space/ your evaluation dataset and training dataset are different, you can comment that out.


Note you may set ``model.data_decoder.full_tx=false`` to run with a larger batch_size



## Something to be done (In progress)

We add symbol_decoder in the model, but it is not complete for the evaluation metrics or so for the symbol part, so we suggest you only 
use 2-to=1 (data) model for now.