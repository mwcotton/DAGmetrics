#!/bin/bash
#PBS -lwalltime=24:00:00
#PBS -lselect=1:ncpus=1:mem=32gb

module load anaconda3/personal

cp $HOME/minkowskitools.py $TMPDIR
cp $HOME/perc_r/perc_r2.py $TMPDIR

anaconda-setup

conda search numpy
conda install numpy

conda install matplotlib

python3 $HOME/perc_r/perc_r2.py

cp * $HOME/perc_r
