#!/bin/bash

CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}❯ Starting LocalBro...${NC}"

# Clean up any leftover bridge files from old versions
rm -f to_summarize.json summary_result.json worker.log

if [ ! -d "venv" ]; then
    echo -e "${RED}Error: venv not found. Please create it first.${NC}"
    exit 1
fi

source venv/bin/activate

# No background worker needed — compression is now inline and synchronous
echo -e "${CYAN}❯ Launching Terminal...${NC}"
python3 main.py
