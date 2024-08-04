#!/bin/bash

/usr/sbin/nginx


PY=D:\App\miniconda3\envs\ragflow\python.exe
export PYTHONPATH=D:\Code\ragflowv8.0\ragflow
export HF_ENDPOINT=https://hf-mirror.com

# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/

# PY=python3
if [[ -z "$WS" || $WS -lt 1 ]]; then
  WS=1
fi

function task_exe(){
    while [ 1 -eq 1 ];do
      $PY rag/svr/task_executor.py ;
    done
}

for ((i=0;i<WS;i++))
do
  task_exe  &
done

while [ 1 -eq 1 ];do
    $PY api/ragflow_server.py
done

wait;
