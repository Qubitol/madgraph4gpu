#!/bin/bash

### TO KILL A TEST:
###kill -9 $(ps -ef | egrep '(mg5amc-madgraph4gpu-2022-bmk.sh|throughputX.sh|check.exe|gpudriver.sh)' | egrep -v '(grep|emacs)' | awk '{print $2}')

startdate=$(date)

# Defaults for all nodes (may be overridden)
image=oras://registry.cern.ch/hep-workloads/mg5amc-madgraph4gpu-2022-bmk:v0.6
extraargs="-ggttgg -dbl -flt -inl0 --gpu" # default for ggttgg is "-p 2048 256 1"
###events=100 # this was ~2h up to 2048 blocks (for 256 threads)
events=800 # this was ~11h up to 1280 blocks (for 256 threads)

# Node-specific configuration
if [ "$(hostname)" == "itscrd70.cern.ch" ]; then
  export SINGULARITY_TMPDIR=/scratch/TMP_AVALASSI/
  export SINGULARITY_CACHEDIR=/scratch/SINGULARITY_CACHEDIR
  resDir=/scratch/TMP_RESULTS
  tstDir=BMK-itscrd70-cuda
  jts=""; for j in 1 2 4 8; do for t in 1; do if [ $((j*t)) -le 8 ]; then jts="$jts [$j,$t]"; fi; done; done 
  ###gbgts=""; for gb in 1 2 4 8 16 32 64 128 256 512 1024 2048; do for gt in 256; do if [ $((gb*gt)) -le $((2048*256)) ]; then gbgts="$gbgts [$gb,$gt]"; fi; done; done 
  ###gbgts=""; for gb in 1 2 4 8 16 20 32 40 64 80 128 160 256 320 640 1280; do for gt in 256; do if [ $((gb*gt)) -le $((2560*256)) ]; then gbgts="$gbgts [$gb,$gt]"; fi; done; done # Tesla V100 uses 80 SM, try to use a multiple of that
  gbgts=""; for gb in 1 2 4 8 16 20 32 40 64 80 128 160 256 320 512 640 1024 1280 2048 2560 5120 10240; do for gt in 32; do if [ $((gb*gt)) -le $((2560*256)) ]; then gbgts="$gbgts [$gb,$gt]"; fi; done; done # Tesla V100 uses 80 SM, try to use a multiple of that
else
  echo "ERROR! Unknown host $(hostname)"; exit 1
fi

# Loop over tests
for gbgt in $gbgts; do
  gbgt=${gbgt:1:-1}; gt=${gbgt#*,}; gb=${gbgt%,*}
  ###echo "gpublks=$gb, gputhrs=$gt"
  for jt in $jts; do
    jt=${jt:1:-1}; t=${jt#*,}; j=${jt%,*}
    ###echo "jobs=$j, threads=$t"
    wDir="${tstDir}/sa-cuda$(printf '%s%05i%s%05i%s%03i%s%03i%s%04i' '-gb' ${gb} '-gt' ${gt} '-j' ${j} '-t' ${t} '-e' ${events})"
    ###echo ${resDir}/${wDir}
    \rm -rf ${resDir}/${wDir}
    singularity run -B ${resDir}:/results ${image} --extra-args "${extraargs} -p${gb},${gt},1" -c${j} -t${t} -e${events} -w /results/${wDir} -W
    ###ls -l ${resDir}/${wDir}
  done
done

echo "Started at ${startdate}"
echo "Completed at $(date)"
