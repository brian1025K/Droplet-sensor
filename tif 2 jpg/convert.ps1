# 1. 將組件載入移至最外層，確保腳本載入時即刻生效
Add-Type -AssemblyName System.Drawing

function ConvertImage {
    [CmdletBinding()]
    Param(
        [Parameter(Mandatory = $true)]
        [string]$path,

        [Parameter(Mandatory = $true)]
        [string]$jpgOrpng
    )

    if (Test-Path $path) {
        $files = Get-ChildItem -Path $path -Filter *.tif -Recurse -File
        
        # 2. 清除字串可能的多餘空白並轉小寫，改用 if 判斷式確保變數賦值成功
        $ext = $jpgOrpng.Trim().ToLower()

        if ($ext -eq 'jpg' -or $ext -eq 'jpeg') {
            $format = [System.Drawing.Imaging.ImageFormat]::Jpeg
        } else {
            $format = [System.Drawing.Imaging.ImageFormat]::Png
        }

        # 3. 執行轉換迴圈
        foreach ($file in $files) {
            $convertfile = New-Object System.Drawing.Bitmap($file.FullName)
            $newfilename = [System.IO.Path]::ChangeExtension($file.FullName, $ext)

            $convertfile.Save($newfilename, $format)
            $convertfile.Dispose()

            Write-Output "已轉換: $newfilename"
        }
    }
    else {
        Write-Host "找不到路徑: $path"
    }
}