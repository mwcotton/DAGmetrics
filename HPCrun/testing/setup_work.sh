#!/bin/bash

module load anaconda3/personal

anaconda-setup

conda search numpy
conda install numpy

conda install matplotlib

