$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$manuscript = Join-Path $repo "manuscript"
$builtPdf = Join-Path $manuscript "main.pdf"
$publishedPdf = Join-Path $repo "paper.pdf"

Push-Location $manuscript
try {
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
    if (-not (Test-Path -LiteralPath $builtPdf)) {
        throw "LaTeX completed without producing $builtPdf"
    }
    Copy-Item -LiteralPath $builtPdf -Destination $publishedPdf -Force
}
finally {
    Pop-Location
}

Write-Output "Built $publishedPdf"
