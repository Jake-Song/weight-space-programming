import pathlib, tarfile
root=pathlib.Path('/content/weight-space')
with tarfile.open('/content/weight-space-results.tar.gz','w:gz') as t:
    for p in root.iterdir():
        if p.name in {'models','base_merged.safetensors','edited_tensors'}: continue
        t.add(p,arcname=p.name)
print('ARCHIVE_BYTES',pathlib.Path('/content/weight-space-results.tar.gz').stat().st_size)
