#!/usr/bin/env bash
set -euo pipefail
V=${1:-5000}; MINW=${2:--30}; MAXW=${3:-30}; DENS=${4:-0.001}; THREADS=${5:-8}; SPLITR=${6:-0.5}
make -s all
./bin/BF_serial $V $MINW $MAXW $DENS
OMP_NUM_THREADS=$THREADS ./bin/BF_openmp $V $MINW $MAXW $DENS $THREADS
./bin/BF_cuda $V $MINW $MAXW $DENS
OMP_NUM_THREADS=$THREADS ./bin/BF_hybrid $V $MINW $MAXW $SPLITR $DENS $THREADS
python3 scripts/compare_rmse.py serial_output__${V}_${MAXW}_${MINW}.txt openmp_output__${V}_${MAXW}_${MINW}.txt
python3 scripts/compare_rmse.py serial_output__${V}_${MAXW}_${MINW}.txt cuda_output__${V}_${MAXW}_${MINW}.txt
python3 scripts/compare_rmse.py serial_output__${V}_${MAXW}_${MINW}.txt hybrid_output__${V}_${MAXW}_${MINW}.txt
