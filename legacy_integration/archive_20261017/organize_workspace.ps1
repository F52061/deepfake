$ErrorActionPreference='Stop'
$taskRoot=(Get-Location).ProviderPath.TrimEnd('\')
$taskArchive=Join-Path $taskRoot 'legacy_integration\archive_20261017'
$taskSnapshot=Join-Path $taskArchive 'snapshot'
$taskDocuments=Join-Path $taskRoot 'documents'
function Assert-WorkspacePath([string]$path) {
    $resolved=[IO.Path]::GetFullPath($path)
    if(-not $resolved.StartsWith($taskRoot+'\',[StringComparison]::OrdinalIgnoreCase)){throw ('Outside workspace: '+$resolved)}
    return $resolved
}
Assert-WorkspacePath $taskArchive | Out-Null
Assert-WorkspacePath $taskSnapshot | Out-Null
Assert-WorkspacePath $taskDocuments | Out-Null
if((Test-Path -LiteralPath $taskArchive) -or (Test-Path -LiteralPath $taskDocuments) -or (Test-Path -LiteralPath (Join-Path $taskRoot 'integration'))){throw 'Destination already exists; refusing to overwrite'}
$taskOriginals=@(Get-ChildItem -LiteralPath $taskRoot -Force | Where-Object {$_.Name -notin @('.git','.claude','legacy_integration')})
$taskMoveNames=@('vit_module','reports','paper','scripts','WORKLOG.md','FINDINGS_问题验证数据.md','describe.md','acc.txt','README.md','run.bat','run_eval_bridge.bat','run_train_bridge.bat','stage_2_train.sh','stage_3_train.sh','stage_3_inference_det.sh','stage_3_inference_exp.sh')
$taskInventory=@()
foreach($item in $taskOriginals){
    $itemPath=Assert-WorkspacePath $item.FullName
    if($item.Attributes -band [IO.FileAttributes]::ReparsePoint){throw ('Reparse point: '+$itemPath)}
    if($item.PSIsContainer){
        $descendants=@(Get-ChildItem -LiteralPath $itemPath -Force -Recurse)
        if($descendants | Where-Object {$_.Attributes -band [IO.FileAttributes]::ReparsePoint}){throw ('Nested reparse point: '+$itemPath)}
        $files=@($descendants | Where-Object {-not $_.PSIsContainer})
    }else{$files=@($item)}
    foreach($file in $files){
        $relative=$file.FullName.Substring($taskRoot.Length+1).Replace('\','/')
        $taskInventory += [PSCustomObject]@{original_path=$relative;bytes=$file.Length;sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant();snapshot_path=('snapshot/'+$relative)}
    }
}
New-Item -ItemType Directory -Path $taskSnapshot,$taskDocuments | Out-Null
$taskInventory | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskArchive 'inventory.json') -Encoding utf8
& git status --porcelain=v1 -uall | Set-Content -LiteralPath (Join-Path $taskArchive 'git_status_before.txt') -Encoding utf8
& git rev-parse HEAD | Set-Content -LiteralPath (Join-Path $taskArchive 'git_head_before.txt') -Encoding utf8
$taskJournal=@()
foreach($item in $taskOriginals){
    $source=Assert-WorkspacePath $item.FullName
    $target=Assert-WorkspacePath (Join-Path $taskSnapshot $item.Name)
    if(Test-Path -LiteralPath $target){throw ('Collision: '+$target)}
    if($item.Name -in $taskMoveNames){
        Move-Item -LiteralPath $source -Destination $target
        $action='move'
    }else{
        Copy-Item -LiteralPath $source -Destination $target -Recurse -Force
        $action='copy_reference'
    }
    $taskJournal += [PSCustomObject]@{action=$action;source=$item.Name;destination=('snapshot/'+$item.Name)}
    $taskJournal | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskArchive 'operations.json') -Encoding utf8
}
# All old documents have already been copied into the snapshot; move their
# original copies into a neutral documents folder, preserving user edits.
foreach($doc in @(Get-ChildItem -LiteralPath $taskRoot -File -Filter '*.docx')){
    $source=Assert-WorkspacePath $doc.FullName
    $target=Assert-WorkspacePath (Join-Path $taskDocuments $doc.Name)
    if(Test-Path -LiteralPath $target){throw ('Document collision: '+$target)}
    Move-Item -LiteralPath $source -Destination $target
    $taskJournal += [PSCustomObject]@{action='move_document';source=$doc.Name;destination=('../../documents/'+$doc.Name)}
}
$taskJournal | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskArchive 'operations.json') -Encoding utf8
# Keep reusable primitives available at their original import paths.
$taskVit=Assert-WorkspacePath (Join-Path $taskRoot 'vit_module')
$taskScripts=Assert-WorkspacePath (Join-Path $taskRoot 'scripts')
New-Item -ItemType Directory -Path $taskVit,$taskScripts | Out-Null
foreach($name in @('__init__.py','vit_adaptive_mattn_aps.py','flash_attn_shim')){
    Copy-Item -LiteralPath (Join-Path $taskSnapshot ('vit_module\'+$name)) -Destination (Join-Path $taskVit $name) -Recurse -Force
}
foreach($name in @('zero2.json','zero3.json','zero3_offload.json')){
    Copy-Item -LiteralPath (Join-Path $taskSnapshot ('scripts\'+$name)) -Destination (Join-Path $taskScripts $name) -Force
}
$taskVerified=0
foreach($entry in $taskInventory){
    $path=Assert-WorkspacePath (Join-Path $taskArchive $entry.snapshot_path)
    if(-not(Test-Path -LiteralPath $path -PathType Leaf)){throw ('Missing archived file: '+$entry.original_path)}
    $actual=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if($actual -ne $entry.sha256){throw ('Hash mismatch: '+$entry.original_path)}
    $taskVerified++
}
[PSCustomObject]@{snapshot_verified_files=$taskVerified;original_total_bytes=($taskInventory | Measure-Object bytes -Sum).Sum;all_original_file_hashes_match=$true;archive='legacy_integration/archive_20261017';server_modified=$false;originals_deleted=$false} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskArchive 'verification.json') -Encoding utf8
Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $taskArchive 'organize_workspace.ps1')
Get-Content -LiteralPath (Join-Path $taskArchive 'verification.json')
