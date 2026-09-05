python experiment.py > blind_review_2_run_out.txt 2>&1
Write-Output ('EXIT:' + $LASTEXITCODE)
Get-Content blind_review_2_run_out.txt -Tail 20
