"""Docker child boundary with existing canonical one-image protocol.
Host is trusted custodian of Docker; this adapter does not isolate host agents.
Only explicitly staged source/weights are exposed; datasets stay in the parent.
"""
from pathlib import Path
from dataclasses import asdict
import json,os,secrets,selectors,signal,subprocess,time,tempfile
from .canonical import Worker,Limits,check_deadlines
from .common import Blocked,atomic_write,canonical,digest,now

DOCKER=['/opt/homebrew/bin/docker','--context','colima-tome']
IMAGE_RECORD=Path(__file__).resolve().parents[2]/'docker/scientific_image/IMAGES.json'

class DockerWorker(Worker):
    def __init__(self,source,evidence_dir,*,limits=Limits(startup_seconds=20,observed_rss_bytes=512*1024*1024,max_calls=10240),mode='candidate',evaluation_deadline=None,baseline_stage=None):
        limits.validate()
        if mode not in ('candidate','cnn_fixture','cnn_staged'):raise Blocked('unknown Docker inference mode')
        staged=None
        if mode=='cnn_staged':
            if baseline_stage is None:raise Blocked('explicit baseline stage required')
            from .baseline_stage import validate_stage
            staged=validate_stage(baseline_stage)
            if digest(Path(source).read_bytes())!=staged['source_sha256']:raise Blocked('source differs from frozen CNN stage')
        elif baseline_stage is not None:raise Blocked('baseline stage only valid in CNN mode')
        self.limits=limits;self.directory=Path(evidence_dir);self.directory.mkdir(parents=True,exist_ok=False)
        self.clock=time.monotonic();self.evaluation_deadline=evaluation_deadline;self.calls=0
        self.stdout=bytearray();self.stderr=bytearray();self.receipts=[];self.raw_frames=[];self.rss_samples=[];self.max_rss=0
        self.termination=None;self.ready=None;self.closed=False;self.proc=None;self.selector=selectors.DefaultSelector();self.cleanup_confirmed=False
        self.container='mnist-inference-'+secrets.token_hex(12)
        # Opaque paths and names do not reveal order/history/timing session semantics.
        scratch_root=IMAGE_RECORD.parent/'scratch';scratch_root.mkdir(exist_ok=True)
        self.scratch=tempfile.TemporaryDirectory(prefix='mnist-docker-',dir=scratch_root);self.source=Path(self.scratch.name)/'classifier.py'
        artifact=Path(source).read_bytes();self.source.write_bytes(artifact);self.source.chmod(0o444)
        (self.directory/'classifier.py').write_bytes(artifact)
        image_key='candidate' if mode=='candidate' else 'baseline';image=json.loads(IMAGE_RECORD.read_text())[image_key]['image_id']
        if staged and staged['image_id']!=image:raise Blocked('staged CNN runtime identity changed')
        runtime=json.loads((IMAGE_RECORD.parent/'RUNTIME_SMOKE.json').read_text())[image_key]
        if runtime['image_id']!=image or runtime['architecture']!='arm64' or runtime['os']!='linux':raise Blocked('scientific image identity/platform mismatch')
        if mode=='cnn_fixture' and digest(artifact)!=runtime['runtime']['files']['/baseline/fixture_cnn.py']['sha256']:raise Blocked('CNN fixture source differs from measured image')
        if not image.startswith('sha256:') or len(image)!=71:raise Blocked('exact locally pinned Docker image required')
        args=DOCKER+['run','--pull','never','--name',self.container,'--label','org.mnist.preparation=scientific-v1','--interactive','--user','65534:65534','--read-only','--network','none','--ipc','none','--cap-drop','ALL','--security-opt','no-new-privileges:true','--pids-limit','32','--memory',str(limits.observed_rss_bytes),'--memory-swap',str(limits.observed_rss_bytes),'--cpus','1','--ulimit','nofile=64:64']
        if mode=='candidate':args+=['--mount','type=bind,src='+str(self.source)+',dst=/candidate/classifier.py,readonly']
        if mode=='cnn_staged':
            weights=(Path(baseline_stage)/'weights.pt').read_bytes()
            if digest(weights)!=staged['weights_sha256']:raise Blocked('CNN checkpoint changed during staging')
            weight_path=Path(self.scratch.name)/'weights.pt';weight_path.write_bytes(weights);weight_path.chmod(0o444)
            args+=['--mount',f'type=bind,src={self.source},dst=/staged/architecture.py,readonly','--mount',f'type=bind,src={weight_path},dst=/staged/weights.pt,readonly']
            child_args=['/staged/architecture.py','cnn_baseline','-','/staged/weights.pt']
        else:child_args=['/candidate/classifier.py','candidate','-','-'] if mode=='candidate' else ['/baseline/fixture_cnn.py','cnn_baseline','-','/baseline/fixture.pt']
        args += [image]+child_args
        self.metadata={'started_at':now(),'source_sha256':digest(artifact),'worker_sha256':digest(Path(__file__).with_name('canonical_child.py').read_bytes()),'docker_adapter_sha256':digest(Path(__file__).read_bytes()),'limits':asdict(limits),'mode':mode,'image_id':image,'container_name':self.container,'argv':args,'resource_memory_semantics':'Docker cgroup memory cap for whole container; no strict Python peak RSS claim','docker_host_trusted':True,'scope':'synthetic_preparation_only','fixture_source_argument_sha256':digest(artifact),'baseline_fixture_in_image':mode=='cnn_fixture','runtime_binding':runtime}
        if staged:self.metadata.update(scope='staged_trained_baseline',baseline_stage=staged)
        try:
            self.launch_clock=time.monotonic()
            self.proc=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0,start_new_session=True)
            os.set_blocking(self.proc.stdin.fileno(),False)
            for stream,label in [(self.proc.stdout,'stdout'),(self.proc.stderr,'stderr')]:
                os.set_blocking(stream.fileno(),False);self.selector.register(stream,selectors.EVENT_READ,label)
            self.ready=self._response(self.launch_clock+limits.startup_seconds,phase='startup',timeout_code='STARTUP_TIMEOUT')
            if self.ready.get('status')!='ready' or self.ready.get('numpy_version')!='2.3.5':raise Blocked('Docker child runtime/version not ready')
            # Inspect actual daemon config before candidate input. Metadata is not passed to child.
            inspect=subprocess.run(DOCKER+['inspect',self.container],capture_output=True,text=True,timeout=5,check=True)
            actual=json.loads(inspect.stdout)[0];self.metadata['container_inspect']=actual
            c=actual['HostConfig'];mounts=actual['Mounts']
            if not(c['ReadonlyRootfs'] and not c['Privileged'] and not c['CapAdd'] and not c['Devices'] and c['PidsLimit']==32 and c['NanoCpus']==1000000000 and c['PidMode']=='' and c['NetworkMode']=='none' and c['IpcMode']=='none' and c['Memory']==limits.observed_rss_bytes and c['MemorySwap']==limits.observed_rss_bytes and actual['Config']['User']=='65534:65534' and set(c['CapDrop'] or [])=={'ALL'} and any(x.startswith('no-new-privileges') for x in c['SecurityOpt'])):raise Blocked('actual container boundary differs')
            expected_destinations={'/candidate/classifier.py'} if mode=='candidate' else ({'/staged/architecture.py','/staged/weights.pt'} if mode=='cnn_staged' else set())
            if len(mounts)!=len(expected_destinations) or {x['Destination'] for x in mounts}!=expected_destinations or any(x['RW'] for x in mounts):raise Blocked('unexpected container mount')
            self.metadata['parent_startup_ms']=(time.monotonic()-self.launch_clock)*1000
            check_deadlines(self._deadlines(), 'after_container_inspect')
        except BaseException as exc:
            self.termination=str(exc);self.close();raise

    def _rss(self,deadline):
        # Docker's cgroup enforces memory for child+descendants. Host docker CLI
        # RSS would be the wrong process and is deliberately not used.
        pass

    def close(self):
        if self.closed:return
        self.closed=True;cleanup={'container':self.container};errors=[]
        try:
            removed=subprocess.run(DOCKER+['rm','--force',self.container],capture_output=True,text=True,timeout=8)
            cleanup['remove_returncode']=removed.returncode;cleanup['remove_stdout']=removed.stdout;cleanup['remove_stderr']=removed.stderr
            # Success of rm alone is insufficient; require daemon reachable and exact name absent.
            listed=subprocess.run(DOCKER+['ps','--all','--filter','name=^/'+self.container+'$','--format','{{.Names}}'],capture_output=True,text=True,timeout=5)
            cleanup['list_returncode']=listed.returncode;cleanup['remaining_names']=listed.stdout.splitlines()
            gone=listed.returncode==0 and not listed.stdout.strip()
        except (OSError,subprocess.TimeoutExpired) as exc:errors.append(str(exc));gone=False
        if self.proc is not None:
            if self.proc.poll() is None:
                try:os.killpg(self.proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            try:self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:errors.append('Docker client not reaped')
            for stream in (self.proc.stdin,self.proc.stdout,self.proc.stderr):stream.close()
        self.selector.close();self.cleanup_confirmed=gone and (self.proc is None or self.proc.poll() is not None) and not errors
        result=dict(self.metadata,ended_at=now(),calls=self.calls,termination=self.termination,ready=self.ready,receipts=self.receipts,raw_frames=self.raw_frames,stderr=self.stderr.decode(errors='replace'),leftover_stdout=self.stdout.decode(errors='replace'),cleanup='container_removed_and_client_reaped' if self.cleanup_confirmed else 'unconfirmed',cleanup_evidence=cleanup,cleanup_errors=errors,elapsed_seconds=time.monotonic()-self.clock)
        atomic_write(self.directory/'RECEIPT.json',canonical(result)+b'\n');self.scratch.cleanup()
        if not self.cleanup_confirmed:raise Blocked('Docker cleanup not confirmed; retain full reservation')
