#!/bin/bash
# Run all AMD-specific tests
# Usage: HIP_VISIBLE_DEVICES=0 bash scripts/run_amd_tests.sh
set -e

echo "=== Miles AMD Test Suite ==="
echo "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null)"
echo ""

PASS=0
FAIL=0

run_test() {
    local name=$1
    local cmd=$2
    echo -n "  $name... "
    if eval "$cmd" > /dev/null 2>&1; then
        echo "PASS"
        ((PASS++))
    else
        echo "FAIL"
        ((FAIL++))
    fi
}

echo "--- Unit Tests ---"
run_test "FP8 AMD tests (13)" "python -m pytest tests/fast/test_fp8_amd.py -q"
run_test "FSDP import test" "python tests/test_fsdp_import.py"
run_test "MoE backward test" "python tests/test_fused_experts_backward.py"

echo ""
echo "--- FSDP Training Tests ---"
run_test "FSDP training suite (6)" "python -m pytest tests/test_fsdp_training_amd.py -q"

echo ""
echo "--- Results ---"
echo "  Passed: $PASS"
echo "  Failed: $FAIL"
echo "  Total: $((PASS + FAIL))"

if [ $FAIL -gt 0 ]; then
    echo "  STATUS: SOME TESTS FAILED"
    exit 1
else
    echo "  STATUS: ALL TESTS PASSED"
fi
