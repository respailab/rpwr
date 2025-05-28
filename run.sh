#!/bin/bash

# Script to run two Python commands sequentially

echo "Running agent_without_kb script..."
# python scripts/agent_without_kb.py --provider GOOGLE --model gemini-2.0-flash --results_dir results/agent_wkb
# python scripts/agent_without_kb.py --provider GOOGLE --model gemini-2.5-pro-preview-03-25 --results_dir results/agent_wkb
# python scripts/agent_without_kb.py --provider OPENAI --model o1 --results_dir results/agent_wkb
python scripts/agent_without_kb.py --provider OLLAMA --model deepseek-r1:70b --results_dir results/agent_wkb
# python scripts/agent_without_kb.py --provider OLLAMA --model qwq --results_dir results/agent_wkb

echo "Running one_agent_with_kb script..."
python scripts/one_agent_with_kb.py --provider OLLAMA --model deepseek-r1:70b --results_dir results/agent_kb
python scripts/one_agent_with_kb.py --provider OLLAMA --model qwq --results_dir results/agent_kb
# python scripts/one_agent_with_kb.py --provider GOOGLE --model gemini-2.5-pro-preview-03-25 --results_dir results/agent_kb
# python scripts/one_agent_with_kb.py --provider GOOGLE --model gemini-2.0-flash --results_dir results/agent_kb

echo "All tasks completed!"
