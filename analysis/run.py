"""Run the frozen diagnostic within one wall-clock budget, with isolated stages."""
import hashlib
import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
IMAGE = 'sha256:3d1d1dc544e932d083845039524f51dd52e20d47e0548c097ed80739bf52355e'
DOCKER = ['/opt/homebrew/bin/docker', '--context', 'colima-tome']
OUT = HERE / 'results'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, data):
    (HERE/name).write_text(json.dumps(data, indent=2) + '\n')


def main():
    expected = {
        'code/classifier.py':'4a718c35621e3c76a170a6447a765e87f39b4c53438e76ddb1a3134b23418a8d',
        'code/original_build.py':'fc6daaa920fe6e4dd35182e06cd8d363a3a6d5ce5416927c9f2802e98216fb39',
    }
    assert not (HERE/'EXECUTION.json').exists(), 'Do not overwrite a completed attempt'
    assert not list(OUT.glob('*.npz')), 'Use an explicit verified resume for existing checkpoints'
    for path, value in expected.items():
        assert digest(HERE/path) == value, path
    receipt = json.loads((ROOT/'mnist_search/runs/experiment_001/candidates/C2/build/RECEIPT.json').read_text())
    fitdir = ROOT/'mnist_search/runs/experiment_001/fit_only'
    inputs = ROOT/'experiment_inputs'
    expected_inputs = {fitdir/n:rec['sha256'] for n,rec in receipt['fit_export']['files'].items()}
    expected_inputs.update({inputs/n:rec['sha256'] for n,rec in receipt['fit_export']['cohort']['source_files'].items()})
    expected_inputs[inputs/'MANIFEST_repair_v4.json'] = receipt['fit_export']['cohort']['manifest_sha256']
    assert all(digest(p)==h for p,h in expected_inputs.items())
    files = {str(p.relative_to(ROOT)): h for p,h in expected_inputs.items()}
    code_hashes={str(p.relative_to(HERE)):digest(p) for p in sorted((HERE/'code').glob('*.py'))}
    write('PREFLIGHT.json', {'verified_inputs':files,'code':code_hashes,
                            'workflow_sha256':digest(HERE/'WORKFLOW_FROZEN.md'),
                            'host_launcher_sha256':digest(Path(__file__)),
                            'image':IMAGE,'originals_modified':False,'extra_charge_calls':0})
    started = time.monotonic(); deadline=started+1800
    run = {'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
           'cap_seconds':1800,'stages':[],'status':'RUNNING','paid_calls':0,
           'final_test_access':False,'candidate_replaced':False}
    write('EXECUTION.json',run)
    for phase in ['fit','evaluate']:
        assert all(digest(HERE/n)==h for n,h in code_hashes.items()), 'Frozen code changed'
        name='mnist-compact-representation-001-'+phase
        args=DOCKER+['run','--rm','--pull','never','--name',name,
            '--read-only','--network','none','--ipc','none','--cap-drop','ALL',
            '--security-opt','no-new-privileges:true','--pids-limit','32',
            '--memory','1073741824','--memory-swap','1073741824','--cpus','1',
            '--user','65534:65534','--mount',f'type=bind,src={HERE / "code"},dst=/src,readonly',
            '--mount',f'type=bind,src={OUT},dst=/out']
        if phase=='fit':
            args += ['--mount',f'type=bind,src={fitdir},dst=/fit,readonly']
        else:
            assert (OUT/'FIT_FROZEN.json').exists()
            args += ['--mount',f'type=bind,src={inputs / "development"},dst=/development,readonly']
        args += ['--entrypoint','python',IMAGE,'-I','-B','/src/experiment.py',phase]
        stage={'phase':phase,'argv':args}
        run['stages'].append(stage);write('EXECUTION.json',run)
        logpath=HERE/(phase+'.log'); begin=time.monotonic();position=0
        with logpath.open('w') as stream:
            proc=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT)
            while True:
                time.sleep(.2)
                with logpath.open() as logs:
                    logs.seek(position);text=logs.read();position=logs.tell()
                if text:print(text,end='',flush=True)
                if proc.poll() is not None:break
                if time.monotonic()>=deadline:
                    subprocess.run(DOCKER+['rm','-f',name],capture_output=True,timeout=15)
                    proc.kill();proc.wait()
                    stage['timeout']=True
                    break
        stage.update(returncode=proc.returncode,seconds=time.monotonic()-begin)
        check=subprocess.run(DOCKER+['ps','-aq','--filter',f'name=^/{name}$'],capture_output=True,text=True,timeout=15)
        stage['container_absent']=check.returncode==0 and not check.stdout.strip()
        if proc.returncode or not stage['container_absent']:
            run.update(status='STOPPED',elapsed_seconds=time.monotonic()-started)
            write('EXECUTION.json',run)
            raise SystemExit(f'{phase} failed; records retained, no automatic scientific retry')
        write('EXECUTION.json',run)
    run.update(status='COMPLETE',elapsed_seconds=time.monotonic()-started,
               finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    write('EXECUTION.json',run)
    print(json.dumps({'status':run['status'],'elapsed_seconds':run['elapsed_seconds']}),flush=True)


if __name__=='__main__':
    main()
