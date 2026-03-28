#!/bin/bash
# IPT Comprehensive Experiments — Fractal Tree
# Phase 1: D1 (GPU 0) + F2 (GPU 1) in parallel
# Phase 2: D2 sequentially (GPU 0, no GRU needed)
# Phase 3: E1 sequentially (dynamics GPU 0, GRU GPU 1)
# Phase 4: F1 (assembly, no GPU needed)

set -e
cd "$(dirname "$0")"
mkdir -p results_D1 results_D2 results_E1 results_F1 results_F2

echo "=========================================="
echo "  IPT Comprehensive Experiments Launcher"
echo "  $(date)"
echo "=========================================="

# Phase 1: D1 (GPU 0) + F2 (GPU 1) in parallel
echo ""
echo ">>> Phase 1: D1 (GPU 0) + F2 (GPU 1)"
echo ">>> Starting $(date)"

CUDA_VISIBLE_DEVICES=0 python -u ipt_comprehensive.py --exp D1 2>&1 | tee results_D1/log.txt &
PID_D1=$!

CUDA_VISIBLE_DEVICES=1 python -u ipt_comprehensive.py --exp F2 2>&1 | tee results_F2/log.txt &
PID_F2=$!

echo "  D1 PID=$PID_D1, F2 PID=$PID_F2"
wait $PID_D1
echo ">>> D1 finished $(date)"
wait $PID_F2
echo ">>> F2 finished $(date)"

# Phase 2: D2 (dynamics + flat comparison on GPU 0)
echo ""
echo ">>> Phase 2: D2"
echo ">>> Starting $(date)"

python -u ipt_comprehensive.py --exp D2 2>&1 | tee results_D2/log.txt
echo ">>> D2 finished $(date)"

# Phase 3: E1 (dynamics GPU 0, GRU training GPU 1)
echo ""
echo ">>> Phase 3: E1"
echo ">>> Starting $(date)"

python -u ipt_comprehensive.py --exp E1 2>&1 | tee results_E1/log.txt
echo ">>> E1 finished $(date)"

# Phase 4: F1 (assembles all results)
echo ""
echo ">>> Phase 4: F1 (assembly)"
echo ">>> Starting $(date)"

python -u ipt_comprehensive.py --exp F1 2>&1 | tee results_F1/log.txt
echo ">>> F1 finished $(date)"

echo ""
echo "=========================================="
echo "  ALL COMPREHENSIVE EXPERIMENTS COMPLETE"
echo "  $(date)"
echo "=========================================="
