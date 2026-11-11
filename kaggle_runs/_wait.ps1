# Chờ 1 kernel Kaggle xong: kiểm tra trạng thái mỗi $Every giây, thoát khi không còn queued/running.
# Dùng: powershell -File kaggle_runs\_wait.ps1 <mã> [giây_giữa_2_lần=600]
param([string]$Exp, [int]$Every = 600)
$env:PYTHONIOENCODING = 'utf-8'
$ref = "phngnam296/edgeai-exp-$Exp"
while ($true) {
    $s = (kaggle kernels status $ref 2>&1 | Out-String).Trim()
    if ($s -notmatch '(?i)queued|running|cancel_requested') {
        Write-Output "[$Exp] $(Get-Date -Format 'HH:mm') $s"
        break
    }
    Start-Sleep -Seconds $Every
}
