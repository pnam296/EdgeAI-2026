# Hàng đợi run Kaggle: Kaggle chỉ cho 2 run GPU cùng lúc mỗi tài khoản -> khi có chỗ trống thì đẩy run tiếp theo.
# Dùng: powershell -File kaggle_runs\_queue.ps1   (sửa danh sách $queues bên dưới)
$env:PYTHONIOENCODING = 'utf-8'
$kaggle = "$PSScriptRoot\..\Scripts\kaggle.exe"
$queues = @{
    'phngnam296'   = @{ token = '';                                     running = @('ema', 'giou');      todo = [System.Collections.ArrayList]@('nb1', 'boxw2', 'wh3') }
    'dohuuminhtri' = @{ token = "$HOME\.kaggle\access_token_ban";       running = @('ep240', 'mosaic');  todo = [System.Collections.ArrayList]@('lr3', 'seed7') }
}

function Use-Account($user, $q) {
    if ($q.token) { $env:KAGGLE_API_TOKEN = $q.token; $env:KAGGLE_USER = $user }
    else { Remove-Item Env:KAGGLE_API_TOKEN -ErrorAction SilentlyContinue; Remove-Item Env:KAGGLE_USER -ErrorAction SilentlyContinue }
}

while ($true) {
    $left = 0
    foreach ($user in $queues.Keys) {
        $q = $queues[$user]
        Use-Account $user $q
        $still = @()
        foreach ($k in $q.running) {
            $s = (& $kaggle kernels status "$user/edgeai-exp-$k" 2>&1 | Out-String).Trim()
            if ($s -match '(?i)queued|running|cancel_requested') { $still += $k }
            else { Write-Output "[$(Get-Date -Format 'dd/MM HH:mm')] XONG $user/$k : $s" }
        }
        $q.running = $still
        while ($q.running.Count -lt 2 -and $q.todo.Count -gt 0) {
            $k = $q.todo[0]
            $out = (& $kaggle kernels push -p "$PSScriptRoot\$k" 2>&1 | Out-String).Trim()
            Write-Output "[$(Get-Date -Format 'dd/MM HH:mm')] ĐẨY $user/$k : $out"
            if ($out -match 'successfully') { $q.running += $k; $q.todo.RemoveAt(0) } else { break }
        }
        $left += $q.running.Count + $q.todo.Count
    }
    if ($left -eq 0) { Write-Output "HẾT HÀNG ĐỢI"; break }
    Start-Sleep -Seconds 600
}
