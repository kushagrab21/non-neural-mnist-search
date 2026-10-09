"""Separately budgeted Docker build/stage, with capped tmpfs output and transport.
Only explicit synthetic work is admitted. No host writable directory is mounted.
"""
from pathlib import Path
import base64,json,os,secrets,selectors,signal,subprocess,time
from .common import Blocked,canonical,digest,atomic_write,strict_json
from .resources import Budget,units
from .docker_worker import DOCKER,IMAGE_RECORD

MAX_BUNDLE=16*1024*1024
BUILD_MEMORY_BYTES=512*1024*1024
BUILD_CPUS=1
BUILD_PIDS=32
BUILD_CLEANUP_SECONDS=10


def build_limits(wall_seconds):
    return {'memory_bytes':BUILD_MEMORY_BYTES,'cpus':BUILD_CPUS,'pids':BUILD_PIDS,
            'wall_allocation_seconds':wall_seconds,'execution_seconds':wall_seconds-BUILD_CLEANUP_SECONDS,
            'output_bytes':MAX_BUNDLE,'network':'none','output_noexec':True,
            'output_files_exactly':['classifier.py','provenance.json']}

def run(store,*,reservation_id,kind,input_dir,output_dir,script_name,wall_seconds,phase,batch,host,scope,fit_dir=None,fit_manifest=None):
    if scope not in ('synthetic_preparation_only','development_pilot') or kind not in ('artifact_build','artifact_stage'):raise Blocked('known artifact job kind/scope required')
    if scope=='development_pilot':
        from .pilot_release import require_store_release
        require_store_release(store)
    if fit_dir is not None:
        from .development_data import validate_fit_export
        validate_fit_export(fit_dir,fit_manifest)
        if scope=='synthetic_preparation_only' and fit_manifest.get('cohort',{}).get('manifest_sha256'):
            raise Blocked('identified real development data requires a pilot release')
    if type(wall_seconds) not in (int,float) or not 15<=wall_seconds<=600:raise Blocked('15..600s inclusive allocation required')
    src=Path(input_dir).resolve();out=Path(output_dir)
    if out.exists():raise Blocked('output already exists; no automatic retry')
    if Path(script_name).name!=script_name or not script_name.endswith('.py'):raise Blocked('one direct script name required')
    input_bytes={};total=0
    paths=list(src.iterdir())
    if not 1<=len(paths)<=32:raise Blocked('one to32 direct inputs required')
    for path in sorted(paths):
        if path.is_symlink() or not path.is_file():raise Blocked('regular direct input required')
        total+=path.stat().st_size
        if total>MAX_BUNDLE:raise Blocked('input bundle over16 MiB')
        input_bytes[path.name]=path.read_bytes()
    inputs={k:digest(v) for k,v in input_bytes.items()}
    if script_name not in inputs:raise Blocked('build/stage script missing')
    image=json.loads(IMAGE_RECORD.read_text())['candidate']['image_id'];collector=Path(__file__).with_name('artifact_collector.py').read_bytes();b=Budget(store)
    b.reserve(reservation_id,kind,units(wall_seconds=wall_seconds),phase=phase,batch=batch,binding={'scope':scope,'input_hashes':inputs,'image_id':image,'script':script_name,'output':str(out.resolve()),'runner_sha256':digest(Path(__file__).read_bytes()),'collector_sha256':digest(collector),'fit_manifest_sha256':digest(fit_manifest) if fit_manifest is not None else None},host=host);b.begin(reservation_id,host)
    out.mkdir(parents=True);frozen=out/'frozen_inputs';frozen.mkdir();runtime=out/'collector.py';runtime.write_bytes(collector);runtime.chmod(0o444)
    for filename,raw in input_bytes.items():store.archive(raw);p=frozen/filename;p.write_bytes(raw);p.chmod(0o444)
    emitted=out/'emitted';name='mnist-artifact-'+secrets.token_hex(12)
    command=DOCKER+['run','--pull','never','--name',name,'--label','org.mnist.preparation=artifact-v2','--user','65534:65534','--read-only','--network','none','--ipc','none','--cap-drop','ALL','--security-opt','no-new-privileges:true','--pids-limit',str(BUILD_PIDS),'--memory',str(BUILD_MEMORY_BYTES),'--memory-swap',str(BUILD_MEMORY_BYTES),'--cpus',str(BUILD_CPUS),'--tmpfs',f'/output:rw,noexec,nosuid,nodev,size={MAX_BUNDLE},mode=1777','--mount',f'type=bind,src={frozen.resolve()},dst=/input,readonly','--mount',f'type=bind,src={runtime.resolve()},dst=/executor/collector.py,readonly','--entrypoint','python',image,'-I','-B','/executor/collector.py','/input/'+script_name]
    if fit_dir is not None:
        position=command.index('--entrypoint')
        command[position:position]=['--mount',f'type=bind,src={Path(fit_dir).resolve()},dst=/fit,readonly']
    guard=b.execution_guard(reservation_id);proc=None;error=None;clean=False;artifacts={};streams={'stdout':bytearray(),'stderr':bytearray()};limits={'stdout':MAX_BUNDLE*4//3+65536,'stderr':65536};selector=selectors.DefaultSelector()
    try:
        guard();proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
        for stream,label in [(proc.stdout,'stdout'),(proc.stderr,'stderr')]:os.set_blocking(stream.fileno(),False);selector.register(stream,selectors.EVENT_READ,label)
        while selector.get_map() or proc.poll() is None:
            elapsed,_=b.elapsed_evidence(reservation_id)
            if elapsed is None or elapsed>=wall_seconds-BUILD_CLEANUP_SECONDS:raise Blocked('artifact execution allowance exhausted; cleanup headroom retained')
            guard()
            for key,_ in selector.select(.05):
                raw=os.read(key.fileobj.fileno(),65536)
                if not raw:selector.unregister(key.fileobj);continue
                target=streams[key.data];room=limits[key.data]-len(target);target.extend(raw[:room])
                if len(raw)>room:raise Blocked('artifact transport output cap exceeded')
        if proc.returncode!=0:raise Blocked('artifact container returned failure')
    except BaseException as exc:error={'type':type(exc).__name__,'detail':str(exc)}
    finally:
        selector.close();cleanup={}
        try:
            removed=subprocess.run(DOCKER+['rm','--force',name],capture_output=True,text=True,timeout=5);cleanup['remove_returncode']=removed.returncode
            absent=subprocess.run(DOCKER+['ps','--all','--filter','name=^/'+name+'$','--format','{{.Names}}'],capture_output=True,text=True,timeout=3);clean=absent.returncode==0 and not absent.stdout.strip();cleanup['absence_confirmed']=clean
        except (OSError,subprocess.TimeoutExpired) as exc:cleanup['error']=str(exc)
        if proc is not None:
            if proc.poll() is None:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            try:proc.wait(timeout=2)
            except subprocess.TimeoutExpired:clean=False
            proc.stdout.close();proc.stderr.close()
        clean=clean and (proc is None or proc.poll() is not None)
        # The received bytes are the artifact. A live output-directory snapshot
        # is never claimed; no candidate-writable host path exists.
        if error is None and clean:
            try:
                bundle=strict_json(bytes(streams['stdout']))
                if set(bundle)!={'protocol','files'} or bundle['protocol']!='synthetic-artifact-bundle/1' or not isinstance(bundle['files'],dict) or not 1<=len(bundle['files'])<=32:raise Blocked('invalid artifact bundle')
                decoded={};total=0
                for filename,encoded in bundle['files'].items():
                    if not isinstance(filename,str) or filename in ('.','..') or Path(filename).name!=filename or not filename or '\\' in filename:raise Blocked('artifact name escape')
                    raw=base64.b64decode(encoded,validate=True);total+=len(raw)
                    if total>MAX_BUNDLE:raise Blocked('decoded bundle over cap')
                    decoded[filename]=raw
                emitted.mkdir()
                for filename,raw in decoded.items():
                    sha=store.archive(raw);(emitted/filename).write_bytes(raw);artifacts[filename]={'sha256':sha,'bytes':len(raw)}
            except BaseException as exc:error={'type':type(exc).__name__,'detail':str(exc)};artifacts={}
        elapsed,clock=b.elapsed_evidence(reservation_id);actual=units(wall_seconds=elapsed) if clean and elapsed is not None else None
        evidence={'kind':kind,'scope':scope,'status':'completed' if error is None and clean else 'failed','error':error,'cleanup_confirmed':clean,'cleanup_evidence':cleanup,'resource_clock':clock,'artifacts':artifacts,'argv':command,'returncode':None if proc is None else proc.returncode,'stdout_bytes':len(streams['stdout']),'stdout_sha256':digest(bytes(streams['stdout'])),'stderr':streams['stderr'].decode(errors='replace'),'input_hashes':inputs,'output_storage':'16MiB container tmpfs included in512MiB cgroup; no host writable mount; host transport buffers independently bounded','artifact_semantics':'validated received bundle after container removal, not a claim about concurrent live filesystem state'}
        evidence['fit_export']=fit_manifest
        atomic_write(out/'RECEIPT.json',canonical(evidence)+b'\n');sha=store.archive((out/'RECEIPT.json').read_bytes());b.settle(reservation_id,actual,{'receipt_sha256':sha,'cleanup_confirmed':clean,'resource_clock':clock},host);atomic_write(out/'RESOURCE_RECEIPT.json',canonical({'actual':actual,'allocation_held':actual is None,'receipt_sha256':sha})+b'\n')
    return evidence
