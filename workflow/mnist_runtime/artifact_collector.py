"""Container-only preparation collector; /output is a bounded tmpfs.
Emits an untrusted artifact bundle. The host validates bytes after container cleanup.
"""
import base64,json,subprocess,sys
from pathlib import Path
out=Path('/output');stdout=out/'.build.stdout';stderr=out/'.build.stderr'
with stdout.open('wb') as a,stderr.open('wb') as b:
 r=subprocess.run([sys.executable,'-I','-B',sys.argv[1]],stdin=subprocess.DEVNULL,stdout=a,stderr=b)
if r.returncode:
 sys.stderr.write(stderr.read_bytes()[:4096].decode(errors='replace'));raise SystemExit(r.returncode)
artifacts={};total=0
for p in sorted(out.iterdir()):
 if p.name in ('.build.stdout','.build.stderr'):continue
 if p.is_symlink() or not p.is_file():raise ValueError('direct regular output files required')
 raw=p.read_bytes();total+=len(raw)
 if total>16*1024*1024 or len(artifacts)>=32:raise ValueError('artifact bundle over cap')
 artifacts[p.name]=base64.b64encode(raw).decode('ascii')
if not artifacts:raise ValueError('no artifact emitted')
print(json.dumps({'protocol':'synthetic-artifact-bundle/1','files':artifacts},separators=(',',':')),flush=True)
