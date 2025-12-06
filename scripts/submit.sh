#!/bin/bash
set -e

# =============================================================================
# Kaggle Unified Submission Script
# =============================================================================
# Usage: sh scripts/submit.sh <experiment_name> [options]
#
# Options:
#   --dry-run       Show what would be done without actually pushing
#   --skip-validate Skip validation step
#   --skip-codes    Skip codes dataset upload
#   --skip-artifacts Skip artifacts model upload
#
# This script handles the entire submission flow:
# 1. Validate submission setup
# 2. Upload codes as Dataset
# 3. Upload artifacts as Model
# 4. Update sub/kernel-metadata.json
# 5. Push submission kernel
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
EXP_NAME=""
DRY_RUN=false
SKIP_VALIDATE=false
SKIP_CODES=false
SKIP_ARTIFACTS=false

for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-validate)
            SKIP_VALIDATE=true
            shift
            ;;
        --skip-codes)
            SKIP_CODES=true
            shift
            ;;
        --skip-artifacts)
            SKIP_ARTIFACTS=true
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
    echo ""
    echo "Usage: sh scripts/submit.sh <experiment_name> [options]"
    echo ""
    echo "Options:"
    echo "  --dry-run       Show what would be done without actually pushing"
    echo "  --skip-validate Skip validation step"
    echo "  --skip-codes    Skip codes dataset upload"
    echo "  --skip-artifacts Skip artifacts model upload"
    exit 1
fi

# Header
echo ""
echo "========================================"
echo -e "${BLUE}Kaggle Submission${NC}"
echo "Experiment: $EXP_NAME"
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}[DRY RUN MODE]${NC}"
fi
echo "========================================"

# =============================================================================
# Step 1: Validation
# =============================================================================
if [ "$SKIP_VALIDATE" = false ]; then
    echo ""
    echo -e "${BLUE}Step 1/5: Validation${NC}"
    echo "----------------------------------------"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would run: sh scripts/validate.sh $EXP_NAME --skip-inference"
    else
        sh scripts/validate.sh "$EXP_NAME" --skip-inference
        if [ $? -ne 0 ]; then
            echo -e "${RED}Validation failed. Aborting submission.${NC}"
            exit 1
        fi
    fi
else
    echo ""
    echo -e "${YELLOW}Step 1/5: Validation [SKIPPED]${NC}"
fi

# =============================================================================
# Step 2: Upload codes as Dataset
# =============================================================================
if [ "$SKIP_CODES" = false ]; then
    echo ""
    echo -e "${BLUE}Step 2/5: Uploading codes dataset${NC}"
    echo "----------------------------------------"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would run: python src/upload.py codes"
    else
        echo "Uploading codes..."
        python src/upload.py codes
        if [ $? -ne 0 ]; then
            echo -e "${RED}Codes upload failed. Aborting submission.${NC}"
            exit 1
        fi
        echo -e "${GREEN}Codes uploaded successfully.${NC}"
    fi
else
    echo ""
    echo -e "${YELLOW}Step 2/5: Codes upload [SKIPPED]${NC}"
fi

# =============================================================================
# Step 3: Upload artifacts as Model
# =============================================================================
if [ "$SKIP_ARTIFACTS" = false ]; then
    echo ""
    echo -e "${BLUE}Step 3/5: Uploading artifacts model${NC}"
    echo "----------------------------------------"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would run: python src/upload.py artifacts --exp_name $EXP_NAME"
    else
        echo "Uploading artifacts for experiment: $EXP_NAME"
        python src/upload.py artifacts --exp_name "$EXP_NAME"
        if [ $? -ne 0 ]; then
            echo -e "${RED}Artifacts upload failed. Aborting submission.${NC}"
            exit 1
        fi
        echo -e "${GREEN}Artifacts uploaded successfully.${NC}"
    fi
else
    echo ""
    echo -e "${YELLOW}Step 3/5: Artifacts upload [SKIPPED]${NC}"
fi

# =============================================================================
# Step 4: Update sub/kernel-metadata.json
# =============================================================================
echo ""
echo -e "${BLUE}Step 4/5: Checking sub/kernel-metadata.json${NC}"
echo "----------------------------------------"

# Check if model_sources needs updating
if grep -q "/$EXP_NAME/1" "sub/kernel-metadata.json"; then
    echo -e "${GREEN}sub/kernel-metadata.json already references experiment $EXP_NAME${NC}"
else
    echo -e "${YELLOW}Note: sub/kernel-metadata.json may need to be updated to reference $EXP_NAME${NC}"
    echo "Current model_sources:"
    grep -A5 '"model_sources"' sub/kernel-metadata.json || echo "  (not found)"
    echo ""
    echo "Expected pattern: .../$EXP_NAME/1"
fi

# Check if inference path in sub/code.ipynb is correct
if grep -q "experiments/$EXP_NAME/inference.py" "sub/code.ipynb"; then
    echo -e "${GREEN}sub/code.ipynb references correct inference.py${NC}"
else
    echo -e "${YELLOW}Warning: sub/code.ipynb may need to be updated${NC}"
    echo "Expected: experiments/$EXP_NAME/inference.py"
fi

# =============================================================================
# Step 5: Push submission kernel
# =============================================================================
echo ""
echo -e "${BLUE}Step 5/5: Pushing submission kernel${NC}"
echo "----------------------------------------"

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would run: cd ./sub && kaggle k push"
    echo ""
    echo -e "${GREEN}[DRY RUN] Submission would complete successfully.${NC}"
else
    echo "Pushing submission kernel..."
    cd ./sub || { echo -e "${RED}Failed to enter ./sub directory${NC}"; exit 1; }
    kaggle k push || { echo -e "${RED}Kaggle push failed${NC}"; cd - > /dev/null; exit 1; }
    cd - > /dev/null

    echo ""
    echo -e "${GREEN}========================================"
    echo "Submission pushed successfully!"
    echo "========================================${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Check submission status: sh scripts/status.sh"
    echo "  2. View on Kaggle: https://www.kaggle.com/code"
fi

echo ""
echo "Done!"
