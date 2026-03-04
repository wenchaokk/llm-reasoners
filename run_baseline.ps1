# Run CoT Baseline on Blocksworld
# Usage: .\run_baseline.ps1

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

# Set environment
$env:PLANBENCH_PATH = Join-Path $root "LLMs-Planning"
$env:VAL = Join-Path $root "LLMs-Planning\planner_tools\VAL"

# Load API key from file
if (-not $env:OPENAI_API_KEY) {
    $keyFile = Join-Path $root "api_key.txt"
    if (Test-Path $keyFile) {
        $env:OPENAI_API_KEY = (Get-Content $keyFile -Raw).Trim()
    }
}

Write-Host "Running CoT baseline (step_4 subset)..." -ForegroundColor Green
python examples/CoT/blocksworld/cot_inference.py `
  --model_dir openai `
  --data_path examples/CoT/blocksworld/data/split_v1/split_v1_step_4_data.json `
  --prompt_path examples/CoT/blocksworld/prompts/pool_prompt_v1.json `
  --log_dir logs/blocksworld_cot_openai_step4/ `
  --temperature 0.0
