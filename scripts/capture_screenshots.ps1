param(
    [string]$Pythonw = ".\venv\Scripts\pythonw.exe",
    [string]$OutputDirectory = "docs\assets"
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class TranslatorWindowCapture {
    [StructLayout(LayoutKind.Sequential)]
    public struct RECT { public int Left; public int Top; public int Right; public int Bottom; }
    [DllImport("user32.dll")]
    public static extern bool GetWindowRect(IntPtr handle, out RECT rectangle);
    [DllImport("user32.dll")]
    public static extern bool MoveWindow(IntPtr handle, int x, int y, int width, int height, bool repaint);
    [DllImport("user32.dll")]
    public static extern bool PrintWindow(IntPtr handle, IntPtr deviceContext, uint flags);
    [DllImport("user32.dll")]
    public static extern int GetSystemMetrics(int index);
}
"@

function Save-TranslatorScreenshot {
    param(
        [ValidateSet("dark", "light")]
        [string]$Theme,
        [string]$Destination
    )

    $env:NINJA_TRANSLATOR_THEME = $Theme
    $env:NINJA_TRANSLATOR_CONFIG = "config.example.yaml"
    $startedAt = Get-Date
    $launcher = Start-Process $Pythonw -ArgumentList "interface.py" -WorkingDirectory $PWD -PassThru
    $windowProcess = $null

    try {
        for ($attempt = 0; $attempt -lt 40; $attempt++) {
            Start-Sleep -Milliseconds 200
            $windowProcess = Get-Process pythonw -ErrorAction SilentlyContinue | Where-Object {
                $_.StartTime -ge $startedAt.AddSeconds(-1) -and
                $_.MainWindowTitle -like "Ninja Translator*"
            } | Select-Object -First 1
            if ($windowProcess) { break }
        }
        if (-not $windowProcess) {
            throw "A janela do tema $Theme não foi encontrada."
        }

        $targetWidth = 1720
        $targetHeight = 968
        $screenWidth = [TranslatorWindowCapture]::GetSystemMetrics(0)
        $screenHeight = [TranslatorWindowCapture]::GetSystemMetrics(1)
        $left = [Math]::Max(0, [Math]::Floor(($screenWidth - $targetWidth) / 2))
        $top = [Math]::Max(0, [Math]::Floor(($screenHeight - $targetHeight) / 2))
        [TranslatorWindowCapture]::MoveWindow(
            $windowProcess.MainWindowHandle,
            $left,
            $top,
            $targetWidth,
            $targetHeight,
            $true
        ) | Out-Null
        Start-Sleep -Milliseconds 700

        $rectangle = New-Object TranslatorWindowCapture+RECT
        [TranslatorWindowCapture]::GetWindowRect(
            $windowProcess.MainWindowHandle,
            [ref]$rectangle
        ) | Out-Null
        $width = $rectangle.Right - $rectangle.Left
        $height = $rectangle.Bottom - $rectangle.Top
        if ($width -ne $targetWidth -or $height -ne $targetHeight) {
            throw "Dimensão inesperada no tema ${Theme}: ${width}x${height}."
        }

        $bitmap = New-Object System.Drawing.Bitmap($width, $height)
        $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
        $deviceContext = $graphics.GetHdc()

        try {
            $captured = [TranslatorWindowCapture]::PrintWindow(
                $windowProcess.MainWindowHandle,
                $deviceContext,
                2
            )
            if (-not $captured) {
                throw "O Windows não conseguiu capturar o tema $Theme."
            }
        }
        finally {
            $graphics.ReleaseHdc($deviceContext)
        }

        $outputPath = Join-Path $OutputDirectory $Destination
        $bitmap.Save($outputPath, [System.Drawing.Imaging.ImageFormat]::Png)
        $graphics.Dispose()
        $bitmap.Dispose()
        Get-Item $outputPath | Select-Object Name, Length
    }
    finally {
        if ($windowProcess) {
            Stop-Process -Id $windowProcess.Id -Force -ErrorAction SilentlyContinue
        }
        Stop-Process -Id $launcher.Id -Force -ErrorAction SilentlyContinue
    }
}

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
try {
    Save-TranslatorScreenshot "dark" "ninja-translator-escuro.png"
    Save-TranslatorScreenshot "light" "ninja-translator-claro.png"
}
finally {
    Remove-Item Env:NINJA_TRANSLATOR_THEME -ErrorAction SilentlyContinue
    Remove-Item Env:NINJA_TRANSLATOR_CONFIG -ErrorAction SilentlyContinue
}
