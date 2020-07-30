#!/bin/bash
#PBS -lwalltime=24:00:00
#PBS -lselect=1:ncpus=1:mem=32gb

module load anaconda3/personal

cp $HOME/minkowski_tools.py $TMPDIR
cp $HOME/perc_r.py $TMPDIR

mkdir $WORK/$PBS_JOBID

anaconda-setup

conda search numpy
conda install numpy

conda install matplotlib

python3 $HOME/perc_r.py

cp * $WORK/$PBS_JOBID