#!/bin/bash
# Wait for B2 to finish, then launch A2 on GPU 1
cd ~/projects1/IPT/final_experiments_v2

echo "[$(date)] Waiting for B2 (PID $1) to finish..."
while kill -0 $1 2>/dev/null; do
    sleep 60
done
echo "[$(date)] B2 finished. Launching A2 on GPU 1..."
sleep 5

CUDA_VISIBLE_DEVICES=1 nohup python -u ipt_final_v2.py --exp A2 > results_A2/log.txt 2>&1 &
A2_PID=$!
echo "[$(date)] A2 launched with PID $A2_PID"
echo $A2_PID > /tmp/a2_pid.txt
