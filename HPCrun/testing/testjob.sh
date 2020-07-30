#!/bin/bash
#PBS -lwalltime=1:00:00
#PBS -lselect=1:ncpus=1:mem=1gb

module load anaconda3/personal

cp $HOME/minkowski_tools.py $TMPDIR
cp $HOME/test.py $TMPDIR

anaconda-setup

conda search numpy
conda install numpy

conda install matplotlib

python3 $HOME/test.py

mkdir $WORK/$PBS_JOBID
cp * $WORK/$PBS_JOBID