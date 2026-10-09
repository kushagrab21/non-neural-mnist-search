"""All-Claude development roles with named-volume homes and no host mounts.

The trusted host passes exact prompts via stdin and collects JSON via stdout.
No Docker socket, final data, final key or host project is mounted in a role.
"""
import json
import secrets
import subprocess
from pathlib import Path
from .common import Blocked
from .docker_worker import DOCKER

ROLES = ('runner', 'researcher', 'controller')
IMAGE_FILE = Path(__file__).resolve().parents[2] / 'docker/agent_image/IMAGE.json'


def command(role, argv, *, name=None, network=False, interactive=False):
    if role not in ROLES: raise Blocked('unknown role')
    image = json.loads(IMAGE_FILE.read_text())['image_id']
    if not image.startswith('sha256:') or len(image) != 71: raise Blocked('pinned role image required')
    result = DOCKER + ['run', '--pull', 'never', '--name', name or 'mnist-role-'+secrets.token_hex(12),
        '--label', 'org.mnist.preparation=roles-v2', '--user', 'node', '--read-only',
        '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges:true', '--ipc', 'none',
        '--pids-limit', '128', '--memory', '1g', '--memory-swap', '1g', '--cpus', '1',
        '--tmpfs', '/tmp:rw,noexec,nosuid,nodev,size=64m,mode=1777',
        '--tmpfs', '/workspace:rw,noexec,nosuid,nodev,size=16m,mode=1777',
        '--mount', f'type=volume,src=mnist-role-{role}-home-v2,dst=/home/node',
        '--workdir', '/workspace', '--network', 'bridge' if network else 'none']
    if interactive: result += ['--interactive']
    return result + [image] + list(argv)


def cleanup(name):
    removed = subprocess.run(DOCKER+['rm', '--force', name], capture_output=True, timeout=8)
    absent = subprocess.run(DOCKER+['ps', '--all', '--filter', 'name=^/'+name+'$', '--format', '{{.Names}}'],
                            capture_output=True, timeout=5)
    return {'remove_returncode': removed.returncode,
            'confirmed': absent.returncode == 0 and not absent.stdout.strip()}


def verify_container(name, role):
    raw = subprocess.run(DOCKER+['inspect', name], capture_output=True, timeout=5, check=True)
    observed = json.loads(raw.stdout)[0]; c = observed['HostConfig']
    mounts = observed['Mounts']
    volumes = [m for m in mounts if m['Type'] != 'tmpfs']
    expected_volume = 'mnist-role-'+role+'-home-v2'
    valid = (len(volumes) == 1 and volumes[0]['Type'] == 'volume' and
             volumes[0]['Name'] == expected_volume and volumes[0]['Destination'] == '/home/node' and
             c['ReadonlyRootfs'] and not c['Privileged'] and c['NetworkMode'] in ('bridge', 'none') and
             set(c['CapDrop'] or []) == {'ALL'} and not c['CapAdd'] and not c['Devices'] and
             c['IpcMode'] == 'none' and c['PidMode'] == '' and c['PidsLimit'] == 128 and
             c['Memory'] == 1073741824 and c['MemorySwap'] == 1073741824 and
             observed['Config']['User'] == 'node' and any(s.startswith('no-new-privileges') for s in c['SecurityOpt']))
    if not valid: raise Blocked('actual role container boundary differs from declared configuration')
    # Deliberately omit environment / auth details from persisted inspection.
    return {'image_id': observed['Image'], 'user': observed['Config']['User'], 'host_config': c,
            'mounts': mounts, 'boundary_verified': True}
