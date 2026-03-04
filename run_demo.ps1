# Run demo with OpenAI API
# Usage: .\run_demo.ps1
#        .\run_demo.ps1 "sk-your-key"
# Or:   $env:OPENAI_API_KEY = "sk-xxx"; .\run_demo.ps1

param([string]$ApiKey = $env:OPENAI_API_KEY)
if ($ApiKey) { $env:OPENAI_API_KEY = $ApiKey }

if (-not $env:OPENAI_API_KEY) {
    Write-Host "Enter your OPENAI_API_KEY:" -ForegroundColor Yellow
    $env:OPENAI_API_KEY = Read-Host
    if (-not $env:OPENAI_API_KEY) {
        Write-Host "No key entered. Exit." -ForegroundColor Red
        exit 1
    }
}

Write-Host "Running CoT quick test..." -ForegroundColor Green
python run_cot_quick.py
if ($LASTEXITCODE -eq 0) {
    Write-Host "Success! Run full demo: jupyter notebook demo.ipynb" -ForegroundColor Green
}
