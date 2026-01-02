#!/bin/bash

# Run all SGLangRolloutWithLogit tests

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "================================================================================"
echo "SGLangRolloutWithLogit Test Suite"
echo "================================================================================"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

failed_tests=()
passed_tests=()

# Test 1: Quick Test
echo "${YELLOW}Running Quick Test...${NC}"
if python tests/quick_test.py; then
    passed_tests+=("Quick Test")
    echo -e "${GREEN}✓ Quick Test PASSED${NC}\n"
else
    failed_tests+=("Quick Test")
    echo -e "${RED}✗ Quick Test FAILED${NC}\n"
fi

# Test 2: Unit Tests
echo "${YELLOW}Running Unit Tests...${NC}"
if python tests/test_sglang_rollout_with_logit.py; then
    passed_tests+=("Unit Tests")
    echo -e "${GREEN}✓ Unit Tests PASSED${NC}\n"
else
    failed_tests+=("Unit Tests")
    echo -e "${RED}✗ Unit Tests FAILED${NC}\n"
fi

# Test 3: Integration Tests
echo "${YELLOW}Running Integration Tests...${NC}"
if python tests/integration_test_logit_flow.py; then
    passed_tests+=("Integration Tests")
    echo -e "${GREEN}✓ Integration Tests PASSED${NC}\n"
else
    failed_tests+=("Integration Tests")
    echo -e "${RED}✗ Integration Tests FAILED${NC}\n"
fi

# Summary
echo "================================================================================"
echo "Test Summary"
echo "================================================================================"

total_passed=${#passed_tests[@]}
total_failed=${#failed_tests[@]}
total_tests=$((total_passed + total_failed))

if [ $total_passed -gt 0 ]; then
    echo -e "${GREEN}Passed Tests:${NC}"
    for test in "${passed_tests[@]}"; do
        echo "  ✓ $test"
    done
    echo ""
fi

if [ $total_failed -gt 0 ]; then
    echo -e "${RED}Failed Tests:${NC}"
    for test in "${failed_tests[@]}"; do
        echo "  ✗ $test"
    done
    echo ""
fi

echo "================================================================================"
echo "Total: $total_passed/$total_tests tests passed"
echo "================================================================================"

if [ $total_failed -eq 0 ]; then
    echo -e "${GREEN}All tests passed! ✓${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed! ✗${NC}"
    exit 1
fi

