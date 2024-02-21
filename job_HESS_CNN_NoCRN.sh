#!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --export=NONE


unset SLURM_EXPORT_ENV



module load python/3.9-anaconda
module load cuda/11.8.0
module load cudnn/8.6.0.163-11.8
module load tensorrt/8.5.3.1-cuda11.8-cudnn8.6


micromamba activate dl1dh

srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Run.py -e 250 -c 4 -ft 'earlymax2' -base 'moda' -plt 'no' -single 'no'
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Run.py -e 250 -c 4  -ft 'earlyconv2' -base 'moda' -plt 'no' -single 'no'
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Run.py -e 250 -c 4  -ft 'earlyconcat2' -base 'moda' -plt 'no' -single 'no'

srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Run.py -e 250  -ft 'earlymax' -base 'moda' -plt 'no' 
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Run.py -e 250  -ft 'earlyconcat' -base 'moda' -plt 'no' 
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Run.py -e 250  -ft 'earlyconcat' -base 'moda' -plt 'no' 

srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Run.py -e 250 -c 4  -ft 'scoresum' -base 'moda' -plt 'no' -single 'no'
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Run.py -e 250 -c 4  -ft 'scoreproduct' -base 'moda' -plt 'no' -single 'no'
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Run.py -e 250 -c 4  -ft 'scoremax' -base 'moda' -plt 'no' -single 'no'

srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Run.py -e 250 -c 4  -ft 'latefc' -base 'moda' -plt 'no' -single 'no'
srun python /home/hpc/b129dc/b129dc26/ECAP_HiWi_Project/HESS_CNN_Run.py -e 250 -c 4  -ft 'latemax' -base 'moda' -plt 'no' -single 'no'
