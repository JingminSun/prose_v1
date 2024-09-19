GPU=0
ICs_equation=50
directory="/home/shared/prosepde"

datasets=(burgers conservation_sinflux conservation_cubicflux inviscid_burgers inviscid_conservation_sinflux inviscid_conservation_cubicflux conservation_linearflux advection diff_bistablereact_1D fplanck heat  Klein_Gordon diff_linearreact_1D diff_squarelogisticreact_1D cahnhilliard_1D Sine_Gordon kdv diff_logisreact_1D wave)
for dataset in "${datasets[@]}"; do
    # Cleanup old data to make sure no mismatching of datasize
    ### Data will stored in the following files as well:
    rm $directory/$dataset/${dataset}_$ICs_equation.prefix
    rm $directory/$dataset/${dataset}_${ICs_equation}_data.h5
     ### Generate Basic Data, param range [(1 - data.param_range_gamma)* q_c, (1 + data.param_range_gamma)* q_c]
    CUDA_VISIBLE_DEVICES=$GPU python3 src/data_gen_pde.py num_workers=12  IC_per_param=$ICs_equation data.param_range_gamma=0.1 data.types=${dataset}  size=512000 directory=$directory

    file_name=onlyqc
    rm $directory/$dataset/${dataset}_${ICs_equation}_${file_name}.prefix
    rm $directory/$dataset/${dataset}_${ICs_equation}_${file_name}_data.h5
    CUDA_VISIBLE_DEVICES=$GPU python3 src/data_gen_pde.py num_workers=12  IC_per_param=$ICs_equation data.param_range_gamma=0 data.types=${dataset}  size=512000 directory=$directory file_name=${file_name}

    ## If you just want to generate data with q_c parameter, just set data.param_range_gamma=0
    ## And if you want to generate data with q_a \neq q_c, and q_a \in [(1 - data.param_range_gamma)* q_c, (1 + data.param_range_gamma)* q_c]
    ## It is better to generate it one by one, sry we did not support generating multiple ones for now

    ## Here is an example you want to generate with three different q_as: (each 1024 data)
    # Cleanup old data to make sure no mismatching of datasize
    ### Data will stored in the following files as well:
    ICs_equation=1024
    file_name=ood0.3_1
    rm $directory/$dataset/${dataset}_${ICs_equation}_${file_name}.prefix
    rm $directory/$dataset/${dataset}_${ICs_equation}_${file_name}_data.h5
    CUDA_VISIBLE_DEVICES=$GPU python3 src/data_gen_pde.py num_workers=0  IC_per_param=$ICs_equation data.param_range_gamma=0.3 data.types=${dataset}  size=1024 directory=$directory file_name=${file_name}


    file_name=ood0.3_2
    rm $directory/$dataset/${dataset}_${ICs_equation}_${file_name}.prefix
    rm $directory/$dataset/${dataset}_${ICs_equation}_${file_name}_data.h5
    CUDA_VISIBLE_DEVICES=$GPU python3 src/data_gen_pde.py num_workers=0  IC_per_param=$ICs_equation data.param_range_gamma=0.3 data.types=${dataset}  size=1024 directory=$directory file_name=${file_name}

    file_name=ood0.3_3
    rm $directory/$dataset/${dataset}_${ICs_equation}_${file_name}.prefix
    rm $directory/$dataset/${dataset}_${ICs_equation}_${file_name}_data.h5
    CUDA_VISIBLE_DEVICES=$GPU python3 src/data_gen_pde.py num_workers=0  IC_per_param=$ICs_equation data.param_range_gamma=0.3 data.types=${dataset}  size=1024 directory=$directory file_name=${file_name}


done





#Set symbol.all_type=true to generate both sympy tree and PROSE tree
directory="dataset_sample"
datasets=(burgers conservation_sinflux conservation_cubicflux inviscid_burgers inviscid_conservation_sinflux inviscid_conservation_cubicflux conservation_linearflux)
for dataset in "${datasets[@]}"; do
    # Cleanup old data to make sure no mismatching of datasize
    ### Data will stored in the following files as well:
    rm $directory/$dataset/${dataset}_$ICs_equation.prefix
    rm $directory/$dataset/${dataset}_${ICs_equation}_data.h5
     ### Generate Basic Data, param range [(1 - data.param_range_gamma)* q_c, (1 + data.param_range_gamma)* q_c]
    CUDA_VISIBLE_DEVICES=$GPU python3 src/data_gen_pde.py num_workers=12  IC_per_param=$ICs_equation data.param_range_gamma=0.1 data.types=${dataset}  size=50 directory=$directory symbol.all_type=true

done

