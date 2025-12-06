#!/bin/bash

# =============================================================================
# Kaggle Submission Status Check Script
# =============================================================================
# Usage: sh scripts/status.sh [kernel_slug]
#
# If kernel_slug is not provided, it will be read from sub/kernel-metadata.json
#
# Shows:
# - Latest kernel execution status
# - Recent submissions for the competition
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "========================================"
echo -e "${BLUE}Kaggle Submission Status${NC}"
echo "========================================"

# Get kernel slug from argument or kernel-metadata.json
KERNEL_SLUG="$1"

if [ -z "$KERNEL_SLUG" ]; then
    if [ -f "sub/kernel-metadata.json" ]; then
        KERNEL_SLUG=$(python -c "import json; print(json.load(open('sub/kernel-metadata.json'))['id'])" 2>/dev/null)
        if [ -z "$KERNEL_SLUG" ]; then
            echo -e "${RED}Error: Could not read kernel slug from sub/kernel-metadata.json${NC}"
            exit 1
        fi
    else
        echo -e "${RED}Error: sub/kernel-metadata.json not found${NC}"
        echo "Usage: sh scripts/status.sh [kernel_slug]"
        exit 1
    fi
fi

echo "Kernel: $KERNEL_SLUG"
echo ""

# =============================================================================
# Check kernel status
# =============================================================================
echo -e "${BLUE}--- Kernel Status ---${NC}"

kaggle kernels status "$KERNEL_SLUG" 2>&1 || {
    echo -e "${YELLOW}Could not fetch kernel status.${NC}"
    echo "The kernel may not exist yet or you may not have access."
}

# =============================================================================
# List recent kernel versions
# =============================================================================
echo ""
echo -e "${BLUE}--- Recent Kernel Outputs ---${NC}"

kaggle kernels output "$KERNEL_SLUG" -p /tmp/kaggle_output 2>&1 && {
    echo "Output files downloaded to /tmp/kaggle_output"
    ls -la /tmp/kaggle_output 2>/dev/null || echo "(no files)"
} || {
    echo -e "${YELLOW}Could not fetch kernel output.${NC}"
    echo "The kernel may still be running or no output is available."
}

# =============================================================================
# Get competition name and list submissions
# =============================================================================
echo ""
echo -e "${BLUE}--- Recent Submissions ---${NC}"

# Try to get competition name from kernel-metadata.json
COMPETITION=""
if [ -f "sub/kernel-metadata.json" ]; then
    COMPETITION=$(python -c "
import json
data = json.load(open('sub/kernel-metadata.json'))
sources = data.get('competition_sources', [])
if sources:
    print(sources[0])
" 2>/dev/null)
fi

if [ -n "$COMPETITION" ]; then
    echo "Competition: $COMPETITION"
    echo ""
    kaggle competitions submissions "$COMPETITION" 2>&1 | head -20 || {
        echo -e "${YELLOW}Could not fetch submissions.${NC}"
    }
else
    echo -e "${YELLOW}Competition name not found in kernel-metadata.json${NC}"
    echo "Cannot list submissions without competition name."
fi

# =============================================================================
# Tips
# =============================================================================
echo ""
echo "========================================"
echo -e "${BLUE}Tips${NC}"
echo "========================================"
echo "- View kernel on web: https://www.kaggle.com/code/$KERNEL_SLUG"
echo "- View full logs: kaggle kernels output $KERNEL_SLUG"
echo "- Pull kernel: kaggle kernels pull $KERNEL_SLUG"
echo ""
