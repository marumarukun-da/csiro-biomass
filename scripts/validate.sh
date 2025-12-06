#!/bin/bash
set -e

# =============================================================================
# Kaggle Submission Validation Script
# =============================================================================
# Usage: sh scripts/validate.sh <experiment_name> [--skip-inference]
#
# This script validates your submission setup before pushing to Kaggle.
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
EXP_NAME=""
SKIP_INFERENCE=false

for arg in "$@"; do
    case $arg in
        --skip-inference)
            SKIP_INFERENCE=true
            shift
            ;;
        *)
            if [ -z "$EXP_NAME" ]; then
                EXP_NAME="$arg"
            fi
            ;;
    esac
done

if [ -z "$EXP_NAME" ]; then
    echo -e "${RED}Error: Experiment name is required.${NC}"
    echo "Usage: sh scripts/validate.sh <experiment_name> [--skip-inference]"
    exit 1
fi

echo "========================================"
echo "Validating submission for experiment: $EXP_NAME"
echo "========================================"

ERRORS=0
WARNINGS=0

# Helper functions
check_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

check_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ERRORS=$((ERRORS + 1))
}

check_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    WARNINGS=$((WARNINGS + 1))
}

# =============================================================================
# 1. Check experiment directory exists
# =============================================================================
echo ""
echo "--- Checking experiment directory ---"

EXP_DIR="experiments/$EXP_NAME"
if [ -d "$EXP_DIR" ]; then
    check_pass "Experiment directory exists: $EXP_DIR"
else
    check_fail "Experiment directory not found: $EXP_DIR"
    exit 1
fi

# =============================================================================
# 2. Check required files exist
# =============================================================================
echo ""
echo "--- Checking required files ---"

if [ -f "$EXP_DIR/config.py" ]; then
    check_pass "config.py exists"
else
    check_fail "config.py not found in $EXP_DIR"
fi

if [ -f "$EXP_DIR/inference.py" ]; then
    check_pass "inference.py exists"
else
    check_fail "inference.py not found in $EXP_DIR"
fi

# =============================================================================
# 3. Check .env file
# =============================================================================
echo ""
echo "--- Checking environment variables ---"

if [ -f ".env" ]; then
    check_pass ".env file exists"

    # Check KAGGLE_USERNAME
    if grep -q "^KAGGLE_USERNAME=" .env && ! grep -q "^KAGGLE_USERNAME=your_username" .env; then
        check_pass "KAGGLE_USERNAME is set"
    else
        check_fail "KAGGLE_USERNAME is not set or still default value"
    fi

    # Check KAGGLE_KEY
    if grep -q "^KAGGLE_KEY=" .env && ! grep -q "^KAGGLE_KEY=your_key" .env; then
        check_pass "KAGGLE_KEY is set"
    else
        check_fail "KAGGLE_KEY is not set or still default value"
    fi
else
    check_fail ".env file not found (copy from .env.sample)"
fi

# =============================================================================
# 4. Check output directory and artifacts
# =============================================================================
echo ""
echo "--- Checking output artifacts ---"

OUTPUT_DIR="data/output/$EXP_NAME/1"
if [ -d "$OUTPUT_DIR" ]; then
    check_pass "Output directory exists: $OUTPUT_DIR"

    # Check if there are any model files
    MODEL_COUNT=$(find "$OUTPUT_DIR" -name "*.pkl" -o -name "*.joblib" -o -name "*.pt" -o -name "*.pth" -o -name "*.bin" 2>/dev/null | wc -l)
    if [ "$MODEL_COUNT" -gt 0 ]; then
        check_pass "Found $MODEL_COUNT model file(s)"
    else
        check_warn "No model files found (*.pkl, *.joblib, *.pt, *.pth, *.bin)"
    fi

    # Check total size
    TOTAL_SIZE=$(du -sm "$OUTPUT_DIR" 2>/dev/null | cut -f1)
    if [ "$TOTAL_SIZE" -lt 500 ]; then
        check_pass "Output size is ${TOTAL_SIZE}MB (< 500MB limit)"
    else
        check_warn "Output size is ${TOTAL_SIZE}MB (approaching 500MB limit)"
    fi
else
    check_fail "Output directory not found: $OUTPUT_DIR"
    echo "    -> Run your experiment first to generate artifacts"
fi

# =============================================================================
# 5. Check deps/kernel-metadata.json
# =============================================================================
echo ""
echo "--- Checking deps configuration ---"

if [ -f "deps/kernel-metadata.json" ]; then
    check_pass "deps/kernel-metadata.json exists"
else
    check_fail "deps/kernel-metadata.json not found"
fi

if [ -f "deps/code.ipynb" ]; then
    check_pass "deps/code.ipynb exists"
else
    check_fail "deps/code.ipynb not found"
fi

# =============================================================================
# 6. Check sub/kernel-metadata.json
# =============================================================================
echo ""
echo "--- Checking submission configuration ---"

if [ -f "sub/kernel-metadata.json" ]; then
    check_pass "sub/kernel-metadata.json exists"

    # Check if model_sources contains the experiment
    if grep -q "$EXP_NAME" "sub/kernel-metadata.json"; then
        check_pass "sub/kernel-metadata.json references experiment $EXP_NAME"
    else
        check_warn "sub/kernel-metadata.json may not reference experiment $EXP_NAME"
        echo "    -> Update model_sources in sub/kernel-metadata.json"
    fi
else
    check_fail "sub/kernel-metadata.json not found"
fi

if [ -f "sub/code.ipynb" ]; then
    check_pass "sub/code.ipynb exists"

    # Check if inference.py path is correct
    if grep -q "experiments/$EXP_NAME/inference.py" "sub/code.ipynb"; then
        check_pass "sub/code.ipynb references correct inference.py"
    else
        check_warn "sub/code.ipynb may not reference experiments/$EXP_NAME/inference.py"
        echo "    -> Update the inference path in sub/code.ipynb"
    fi
else
    check_fail "sub/code.ipynb not found"
fi

# =============================================================================
# 7. Test inference.py execution (optional)
# =============================================================================
if [ "$SKIP_INFERENCE" = false ]; then
    echo ""
    echo "--- Testing inference.py execution ---"

    if [ -f "$EXP_DIR/inference.py" ]; then
        echo "Running inference.py (this may take a moment)..."

        # Create a temporary test
        cd "$EXP_DIR"
        if python inference.py 2>&1; then
            check_pass "inference.py executed successfully"

            # Check if submission.csv was created
            cd - > /dev/null
            if [ -f "$OUTPUT_DIR/submission.csv" ]; then
                check_pass "submission.csv was generated"
            else
                check_warn "submission.csv was not found after inference"
            fi
        else
            cd - > /dev/null
            check_fail "inference.py execution failed"
        fi
    fi
else
    echo ""
    echo "--- Skipping inference.py test (--skip-inference) ---"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================"
echo "Validation Summary"
echo "========================================"

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}All checks passed! Ready for submission.${NC}"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}Passed with $WARNINGS warning(s). Review warnings before submission.${NC}"
    exit 0
else
    echo -e "${RED}Failed with $ERRORS error(s) and $WARNINGS warning(s).${NC}"
    echo "Please fix the errors before submission."
    exit 1
fi
