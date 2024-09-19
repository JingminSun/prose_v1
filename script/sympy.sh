GPU=1
directory="dataset"


b_type=[conservation_sinflux,inviscid_conservation_cubicflux,inviscid_burgers,burgers,conservation_cubicflux,inviscid_conservation_sinflux]
invb_type=[inviscid_conservation_cubicflux,inviscid_burgers,inviscid_conservation_sinflux]

## Original Tree

CUDA_VISIBLE_DEVICES=$GPU python src/main.py  exp_name=sympy_v1 exp_id=base_skeleton use_wandb=1 eval_size=5120 train_size=25600 eval_size_get=-1  train_size_get=-1  batch_size=512 batch_size_eval=512 n_steps_per_epoch=2000 max_epoch=30  data.train_types=$b_type data.eval_types=$b_type  save_periodic=-1 model.no_text_decoder=false symbol.use_skeleton=true data.directory=/home/shared/prosepde
CUDA_VISIBLE_DEVICES=$GPU python src/main.py  exp_name=sympy_v1 exp_id=eval_base_skeleton eval_from_exp=checkpoint/jingmins/dumped/sympy_v1/base_skeleton eval_only=1 use_wandb=1 eval_size=5120 train_size=30720 eval_size_get=-1  train_size_get=-1  batch_size=512 batch_size_eval=512  data.train_types=$b_type data.eval_types=$b_type noise=0 model.no_text_decoder=false symbol.use_skeleton=true data.directory=/home/shared/prosepde
CUDA_VISIBLE_DEVICES=$GPU python src/main.py  exp_name=sympy_v1 exp_id=eval_base_skeleton_swapping eval_from_exp=checkpoint/jingmins/dumped/sympy_v1/base_skeleton eval_only=1 use_wandb=1 eval_size=5120 train_size=30720 eval_size_get=-1  train_size_get=-1  batch_size=512 batch_size_eval=512  data.train_types=$b_type data.eval_types=$b_type noise=0 model.no_text_decoder=false symbol.use_skeleton=true data.directory=/home/shared/prosepde symbol.swapping=true
CUDA_VISIBLE_DEVICES=$GPU python src/main.py  exp_name=sympy_v1 exp_id=eval_base_skeleton_noisytext eval_from_exp=checkpoint/jingmins/dumped/sympy_v1/base_skeleton eval_only=1 use_wandb=1 eval_size=5120 train_size=30720 eval_size_get=-1  train_size_get=-1  batch_size=512 batch_size_eval=512  data.train_types=$b_type data.eval_types=$b_type noise=0 model.no_text_decoder=false symbol.use_skeleton=true data.directory=/home/shared/prosepde symbol.noisy_text_input=true

## Sympy Tree

CUDA_VISIBLE_DEVICES=$GPU python src/main.py  exp_name=sympy_v1 exp_id=sbase_skeleton use_wandb=1 eval_size=5120 train_size=25600 eval_size_get=-1  train_size_get=-1  batch_size=512 batch_size_eval=512 n_steps_per_epoch=2000 max_epoch=30  data.train_types=$b_type data.eval_types=$b_type  save_periodic=-1 model.no_text_decoder=false symbol.use_skeleton=true symbol.use_sympy=true
CUDA_VISIBLE_DEVICES=$GPU python src/main.py  exp_name=sympy_v1 exp_id=eval_sbase_skeleton eval_from_exp=checkpoint/jingmins/dumped/sympy_v1/sbase_skeleton eval_only=1 use_wandb=1 eval_size=5120 train_size=30720 eval_size_get=-1  train_size_get=-1  batch_size=512 batch_size_eval=512  data.train_types=$b_type data.eval_types=$b_type noise=0 model.no_text_decoder=false symbol.use_skeleton=true symbol.use_sympy=true
CUDA_VISIBLE_DEVICES=$GPU python src/main.py  exp_name=sympy_v1 exp_id=eval_sbase_skeleton_noisytext eval_from_exp=checkpoint/jingmins/dumped/sympy_v1/sbase_skeleton eval_only=1 use_wandb=1 eval_size=5120 train_size=30720 eval_size_get=-1  train_size_get=-1  batch_size=512 batch_size_eval=512  data.train_types=$b_type data.eval_types=$b_type noise=0 model.no_text_decoder=false symbol.use_skeleton=true symbol.noisy_text_input=true symbol.use_sympy=true
CUDA_VISIBLE_DEVICES=$GPU python src/main.py  exp_name=sympy_v1 exp_id=sfullbase_skeleton use_wandb=1 eval_size=5120 train_size=25600 eval_size_get=512  train_size_get=-1  batch_size=512 batch_size_eval=512 n_steps_per_epoch=2000 max_epoch=30  data.train_types=-1 data.eval_types=-1  save_periodic=-1 model.no_text_decoder=false symbol.use_skeleton=true symbol.use_sympy=true
CUDA_VISIBLE_DEVICES=$GPU python src/main.py  exp_name=sympy_v1 exp_id=eval_sfullbase_skeleton eval_from_exp=checkpoint/jingmins/dumped/sympy_v1/sfullbase_skeleton eval_only=1 use_wandb=1 eval_size=5120 train_size=30720 eval_size_get=-1  train_size_get=-1  batch_size=512 batch_size_eval=512  data.train_types=-1 data.eval_types=-1 noise=0 model.no_text_decoder=false symbol.use_skeleton=true symbol.use_sympy=true
CUDA_VISIBLE_DEVICES=$GPU python src/main.py  exp_name=sympy_v1 exp_id=eval_sfullbase_skeleton_noisytext eval_from_exp=checkpoint/jingmins/dumped/sympy_v1/sfullbase_skeleton eval_only=1 use_wandb=1 eval_size=5120 train_size=30720 eval_size_get=-1  train_size_get=-1  batch_size=512 batch_size_eval=512  data.train_types=-1 data.eval_types=-1 noise=0 model.no_text_decoder=false symbol.use_skeleton=true symbol.noisy_text_input=true symbol.use_sympy=true



## Run for filters
CUDA_VISIBLE_DEVICES=$GPU python src/main.py  exp_name=sympy_v1 exp_id=eval_sbase_skeleton_filter eval_from_exp=checkpoint/jingmins/dumped/sympy_v1/sbase_skeleton eval_only=1 use_wandb=1 eval_size=10 train_size=30720 eval_size_get=-1  train_size_get=-1  batch_size=10 batch_size_eval=10  data.train_types=$invb_type data.eval_types=$invb_type noise=0 model.no_text_decoder=false symbol.use_skeleton=true symbol.use_sympy=true symbol.refine=true
CUDA_VISIBLE_DEVICES=$GPU python src/main.py  exp_name=sympy_v1 exp_id=eval_sbase_skeleton_filter2 eval_from_exp=checkpoint/jingmins/dumped/sympy_v1/sbase_skeleton eval_only=1 use_wandb=1 eval_size=100 train_size=30720 eval_size_get=-1  train_size_get=-1  batch_size=10 batch_size_eval=100  data.train_types=$b_type data.eval_types=$b_type noise=0 model.no_text_decoder=false symbol.use_skeleton=true symbol.use_sympy=true symbol.refine=true

CUDA_VISIBLE_DEVICES=$GPU python src/main.py  exp_name=sympy_v1 exp_id=eval_sbase_skeleton_noisytext_filter eval_from_exp=checkpoint/jingmins/dumped/sympy_v1/sbase_skeleton eval_only=1 use_wandb=1 eval_size=100 train_size=30720 eval_size_get=-1  train_size_get=-1  batch_size=512 batch_size_eval=100  data.train_types=$b_type data.eval_types=$b_type noise=0 model.no_text_decoder=false symbol.use_skeleton=true symbol.noisy_text_input=true symbol.use_sympy=true symbol.refine=true
