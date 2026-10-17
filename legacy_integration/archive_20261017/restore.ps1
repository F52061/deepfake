param([Parameter(Mandatory=$true)][string]$Destination)
$ErrorActionPreference='Stop'
$archiveRoot=$PSScriptRoot
$snapshotRoot=Join-Path $archiveRoot 'snapshot'
$targetRoot=[IO.Path]::GetFullPath($Destination)
if(-not [IO.Path]::IsPathRooted($Destination)){throw 'Use an explicit absolute destination path.'}
if($targetRoot.Equals($archiveRoot,[StringComparison]::OrdinalIgnoreCase) -or $targetRoot.StartsWith($archiveRoot+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Destination must be outside the archive to avoid recursive copying.'}
if(Test-Path -LiteralPath $targetRoot){throw 'Destination already exists. Choose a new directory; nothing will be overwritten.'}
$inventory=Get-Content -LiteralPath (Join-Path $archiveRoot 'inventory.json') -Raw -Encoding utf8 | ConvertFrom-Json
foreach($entry in $inventory){
    $source=[IO.Path]::GetFullPath((Join-Path $archiveRoot $entry.snapshot_path))
    if(-not $source.StartsWith($snapshotRoot+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Inventory path escaped snapshot.'}
    if((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256){throw ('Archive hash mismatch: '+$entry.original_path)}
}
New-Item -ItemType Directory -Path $targetRoot | Out-Null
foreach($item in Get-ChildItem -LiteralPath $snapshotRoot -Force){
    Copy-Item -LiteralPath $item.FullName -Destination (Join-Path $targetRoot $item.Name) -Recurse -Force
}
foreach($entry in $inventory){
    $target=[IO.Path]::GetFullPath((Join-Path $targetRoot $entry.original_path))
    if(-not $target.StartsWith($targetRoot+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Inventory path escaped destination.'}
    if((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256){throw ('Restored hash mismatch: '+$entry.original_path)}
}
Write-Output ('Restored and verified '+$inventory.Count+' original files at '+$targetRoot)
