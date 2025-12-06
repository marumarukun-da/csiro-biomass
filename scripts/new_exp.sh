#!/bin/bash

# =============================================================================
# Create New Experiment Script
# =============================================================================
# Usage: sh scripts/new_exp.sh [options]
#
# Options:
#   --template TYPE   Template type: tabular (default), image
#   --name NAME       Experiment name (default: auto-increment number)
#
# Examples:
#   sh scripts/new_exp.sh                          # Create 002 with tabular template
#   sh scripts/new_exp.sh --template image         # Create 002 with image template
#   sh scripts/new_exp.sh --name my_exp            # Create my_exp folder
# =============================================================================

set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
TEMPLATE="tabular"
EXP_NAME=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --template)
            TEMPLATE="$2"
            shift 2
            ;;
        --name)
            EXP_NAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: sh scripts/new_exp.sh [options]"
            echo ""
            echo "Options:"
            echo "  --template TYPE   Template type: tabular (default), image"
            echo "  --name NAME       Experiment name (default: auto-increment number)"
            echo ""
            echo "Examples:"
            echo "  sh scripts/new_exp.sh                          # Create next numbered experiment"
            echo "  sh scripts/new_exp.sh --template image         # Create with image template"
            echo "  sh scripts/new_exp.sh --name my_exp            # Create named experiment"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Validate template
TEMPLATE_DIR="experiments/templates/$TEMPLATE"
if [ ! -d "$TEMPLATE_DIR" ]; then
    echo -e "${RED}Error: Template '$TEMPLATE' not found.${NC}"
    echo "Available templates:"
    ls -1 experiments/templates/ 2>/dev/null || echo "  (none)"
    exit 1
fi

# Auto-generate experiment name if not provided
if [ -z "$EXP_NAME" ]; then
    # Find the highest numbered experiment
    LAST_NUM=0
    for dir in experiments/*/; do
        dir_name=$(basename "$dir")
        # Check if directory name is a number
        if [[ "$dir_name" =~ ^[0-9]+$ ]]; then
            num=$((10#$dir_name))  # Convert to decimal
            if [ "$num" -gt "$LAST_NUM" ]; then
                LAST_NUM=$num
            fi
        fi
    done

    # Increment and format with leading zeros
    NEXT_NUM=$((LAST_NUM + 1))
    EXP_NAME=$(printf "%03d" $NEXT_NUM)
fi

# Check if experiment already exists
EXP_DIR="experiments/$EXP_NAME"
if [ -d "$EXP_DIR" ]; then
    echo -e "${RED}Error: Experiment '$EXP_NAME' already exists.${NC}"
    exit 1
fi

# Create experiment directory
echo ""
echo -e "${BLUE}Creating new experiment: $EXP_NAME${NC}"
echo "Template: $TEMPLATE"
echo ""

mkdir -p "$EXP_DIR"

# Copy template files
echo "Copying template files..."
cp -r "$TEMPLATE_DIR"/* "$EXP_DIR/"

# Update experiment name in config.py if it exists
if [ -f "$EXP_DIR/config.py" ]; then
    # The EXP_NAME is automatically derived from the directory name in config.py
    echo -e "${GREEN}config.py will use EXP_NAME='$EXP_NAME' automatically${NC}"
fi

echo ""
echo -e "${GREEN}========================================"
echo "Experiment created successfully!"
echo "========================================${NC}"
echo ""
echo "Location: $EXP_DIR"
echo ""
echo "Files created:"
ls -la "$EXP_DIR"
echo ""
echo "Next steps:"
echo "  1. Edit $EXP_DIR/code.ipynb for training"
echo "  2. Edit $EXP_DIR/inference.py for inference"
echo "  3. Run your experiment"
echo "  4. Submit: sh scripts/submit.sh $EXP_NAME"
echo ""
