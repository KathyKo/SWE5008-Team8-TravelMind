$body = @{
    origin = "Taipei"
    destination = "Singapore"
    dates = "2026-10-01 to 2026-10-07"
    duration = "7 days"
    budget = "5000 USD"
    preferences = "family friendly"
    outbound_time_pref = "morning"
    return_time_pref = "afternoon"
    user_profile = @{}
} | ConvertTo-Json

Write-Host "正在發送請求至後端 (這可能需要 1-2 分鐘，因為包含 Agent 辯論)..."

# 發送請求並取得結果
$response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/planner/debate" -Method Post -Body $body -ContentType "application/json"

# 產生檔名並儲存檔案
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$filename = "test_result_$($timestamp).json"
$response | ConvertTo-Json -Depth 10 | Out-File -FilePath $filename -Encoding utf8

Write-Host "測試完成！結果已儲存至: $filename"
