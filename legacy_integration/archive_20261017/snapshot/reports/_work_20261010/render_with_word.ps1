param(
    [Parameter(Mandatory=$true)][string]$InputPath,
    [Parameter(Mandatory=$true)][string]$OutputDirectory
)
$ErrorActionPreference = 'Stop'
$documentPath = (Resolve-Path -LiteralPath $InputPath).Path
$renderDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
[System.IO.Directory]::CreateDirectory($renderDirectory) | Out-Null
$pdfPath = Join-Path $renderDirectory 'render.pdf'
$wordApplication = $null
$wordDocument = $null
$ownsApplication = $false
try {
    $wordApplication = New-Object -ComObject Word.Application
    if ($wordApplication.Documents.Count -ne 0) {
        throw 'Word returned an application with existing documents; no user document will be touched.'
    }
    $ownsApplication = $true
    $wordApplication.Visible = $false
    $wordApplication.DisplayAlerts = 0
    $wordDocument = $wordApplication.Documents.Open($documentPath, $false, $true, $false)
    $wordDocument.Repaginate()
    $pageCount = $wordDocument.ComputeStatistics(2)
    $wordDocument.ExportAsFixedFormat($pdfPath, 17)
    Write-Output "Exported $pageCount pages to $pdfPath"
}
finally {
    if ($null -ne $wordDocument) {
        $wordDocument.Close(0)
        [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($wordDocument) | Out-Null
    }
    if ($null -ne $wordApplication) {
        if ($ownsApplication) { $wordApplication.Quit(0) }
        [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($wordApplication) | Out-Null
    }
}
