#!/bin/bash
# 安全停止:kill 命令在脚本文件里,进程 cmdline 是 "bash dsv4_stop.sh" 不含 run_deepseek → 不会 pkill -f 自杀
pkill -9 -f run_deepseek_v4 2>/dev/null || true
pkill -9 -f sglang 2>/dev/null || true
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
sleep 2
echo "[$(hostname)] stopped. residual python: $(pgrep -fc 'run_deepseek_v4|sglang::' 2>/dev/null || echo 0)"
