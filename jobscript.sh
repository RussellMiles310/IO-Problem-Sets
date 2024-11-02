#!/bin/bash
#SBATCH -A e32328 
#SBATCH -p normal
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=oxb9618@kellogg.northwestern.edu
#SBATCH --job-name=MM_estimation

# unload any modules that carried over from your command line session
module purge

# load modules you need to use
module load matlab/r2023b

# A command you actually want to execute:
matlab -nodisplay -nosplash -nodesktop -r "main; exit"
