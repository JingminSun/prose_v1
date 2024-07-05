GPU=0


train_type=[advection,conservation_sinflux,diff_bistablereact_1D,fplanck,heat,inviscid_conservation_cubicflux,Klein_Gordon]
eval_type=[diff_linearreact_1D,inviscid_burgers,burgers,diff_squarelogisticreact_1D,cahnhilliard_1D,conservation_cubicflux,Sine_Gordon,inviscid_conservation_sinflux,kdv,diff_logisreact_1D,porous_medium,wave]
b_type=[conservation_sinflux,inviscid_conservation_cubicflux,inviscid_burgers,burgers,conservation_cubicflux,inviscid_conservation_sinflux]

#CUDA_VISIBLE_DEVICES=$GPU python src/main.py  exp_name=debug exp_id=burgerstype wandb.id=burgerstype eval_size_get=100  train_size_get=20 n_steps_per_epoch=10 max_epoch=1  data.train_types=$b_type data.eval_types=$b_type model.name=prose &&

CUDA_VISIBLE_DEVICES=$GPU python src/main.py  exp_name=debug exp_id=burgerstypev1 use_wandb=0 eval_size_get=4000  train_size_get=2000  batch_size=128 batch_size_eval=512 n_steps_per_epoch=2000 max_epoch=15  data.train_types=$b_type data.eval_types=$b_type model.name=prose &&

echo "Done."