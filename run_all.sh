#!/bin/bash
set -e
cd ~/projects1/IPT/final_experiments_v2
mkdir -p results_A2 results_B2 results_C2

echo "=== READING EXISTING CODEBASE ==="
ls -la ~/projects1/IPT/Calude_IPT/3_IPT_Memory/ipt_experiment/
echo ""

echo "=== EXP C2: Scaling (original model) === $(date)"
python -u ipt_final_v2.py --exp C2 2>&1 | tee results_C2/log.txt

echo "=== EXP B2: Hopf angle === $(date)"
python -u ipt_final_v2.py --exp B2 2>&1 | tee results_B2/log.txt

echo "=== EXP A2: betaG collapse === $(date)"
python -u ipt_final_v2.py --exp A2 2>&1 | tee results_A2/log.txt

echo "=== ALL COMPLETE === $(date)"
