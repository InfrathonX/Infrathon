#!/usr/bin/env python3
"""Root author diagnostic: 12 controls against one genuine ephemeral five-layer bundle.

No model/fixture generation, Docker scheduling or GPU reservation is performed.
Root authenticates five files and retains their FDs before candidate execution.
Every admitted control uses a new permanently dropped agent process, with L2
warm-up and all four target layers. This finite battery is not full qualification.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pwd
import re
import secrets
import signal
import stat
import subprocess
import sys
import tempfile
import time

PROTOCOL = 'minference-on-demand-integrity-gpu-controls-v3'
EXPECTED_RUNTIME = {
    'honest-seed': {'numeric-pass'}, 'manual-dense': {'numeric-pass'},
    'fixed-geometry-cache': {'numeric-pass'}, 'fixed-geometry-sparse': {'numeric-pass', 'numeric-fail'},
    'warm-output-replay': {'numeric-fail'}, 'input-pattern-cache': {'numeric-pass', 'numeric-fail'},
    'input-mutation': {'input_mutation'}, 'nonfinite-nan': {'incomplete_output'},
    'nonfinite-inf': {'incomplete_output'}, 'incomplete-write': {'incomplete_output'},
    'hidden-reference-access': {'source-admission-rejected'}, 'hidden-gpu-memory': {'source-admission-rejected'},
}
RUNTIME_FILES = ('single_prompt_archive_v3.py', 'prepare_single_prompt_v3.py',
    'measure_single_prompt_v3.py', 'run_single_prompt_worker_v3.py', 'stream_parent_v3.py')
SOURCE_FILES = tuple('runtime/' + name for name in RUNTIME_FILES) + (
    'runtime/on_demand_suite_v3.py', 'runtime/minference_prefill_runtime.py',
    'runtime/activation_archive_v3.py', 'runtime/kernel_fidelity_v3.py', 'candidate_worker.py', 'source_admission.py',
    'submission_loader.py', 'activations/recipe/capture_activations.py',
    'activations/recipe/RULER_V3_RECIPE.json')
SHA = re.compile(r'[a-f0-9]{64}')
GPU = re.compile(r'GPU-[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}')


def require(test, message):
    if not test:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def normalized_gpu(value):
    require(isinstance(value, str), 'actual GPU UUID must be text')
    value = value.lower()
    if not value.startswith('gpu-'): value = 'gpu-' + value
    require(GPU.fullmatch('GPU-' + value[4:]), 'actual GPU UUID is malformed')
    return value


def identity(info):
    return [info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_uid,
            info.st_gid, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def read(path, *, limit=1024**2, private=False):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts and
            not any(p.is_symlink() for p in (path, *path.parents)), 'canonical file path required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and 0 < before.st_size <= limit,
                'bounded single-link metadata/source required')
        if private:
            require(before.st_uid == os.geteuid() and stat.S_IMODE(before.st_mode) in (0o400, 0o600),
                    'private metadata owner/mode differs')
        raw = os.pread(fd, limit + 1, 0)
        require(len(raw) == before.st_size and identity(before) == identity(os.fstat(fd)) == identity(path.lstat()),
                'metadata/source changed during read')
        return raw
    finally:
        os.close(fd)


def parse(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, 'duplicate JSON field')
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def write(path, value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return digest(raw)


def source_paths(root):
    judge = root / 'images/judge' if (root / 'images/judge').is_dir() else root
    return {relative: (judge / relative if '/' not in relative else root / relative)
            for relative in SOURCE_FILES}


def trusted_modules(root):
    paths = source_paths(root)
    sources = {name: digest(read(path)) for name, path in paths.items()}
    sys.path.insert(0, str(root / 'runtime'))
    sys.path.insert(0, str(paths['candidate_worker.py'].parent))
    import activation_archive_v3 as archive
    import minference_prefill_runtime as runtime
    import candidate_worker
    import source_admission
    import submission_loader
    import single_prompt_archive_v3 as single
    import measure_single_prompt_v3 as measure
    import stream_parent_v3 as stream
    return dict(paths=paths, sources=sources, archive=archive, runtime=runtime,
        worker=candidate_worker, admission=source_admission, loader=submission_loader,
        single=single, measure=measure, stream=stream)


def builder_from_raw(path, raw):
    """Execute the already authenticated source, without reopening its path."""
    spec = importlib.util.spec_from_file_location('trusted_on_demand_control_builder', path)
    builder = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(path), 'exec'), builder.__dict__)
    return builder


def verify_stable_sources(modules, originals):
    require({name: digest(read(path)) for name, path in modules['paths'].items()} == modules['sources']
            and all(read(path, private=private) == raw for path, raw, private in originals),
            'source/input/policy/inspection changed during controls')


def validate_inspection(value, *, image_id, gpu_uuid):
    require(re.fullmatch(r'sha256:[a-f0-9]{64}', image_id) and GPU.fullmatch(gpu_uuid), 'actual image/GPU identity required')
    if isinstance(value, list):
        require(len(value) == 1, 'one actual container inspection required'); value = value[0]
    require(isinstance(value, dict) and SHA.fullmatch(str(value.get('Id')))
            and value.get('Image') == image_id and value.get('State', {}).get('Running') is True,
            'actual running container/image differs')
    host, config = value.get('HostConfig', {}), value.get('Config', {})
    require(config.get('User') in ('root', '0', '0:0') and host.get('NetworkMode') == 'none'
            and host.get('Privileged') is False and host.get('Devices') in (None, []),
            'root isolated container required')
    requests = host.get('DeviceRequests')
    require(isinstance(requests, list) and len(requests) == 1, 'one explicit allocation required')
    request = requests[0]; pool = request.get('DeviceIDs')
    require(request.get('Driver') == 'nvidia' and type(request.get('Count')) is int and request['Count'] == 0
            and isinstance(pool, list) and len(pool) == 1 and len(set(pool)) == len(pool)
            and all(isinstance(x, str) and GPU.fullmatch(x) for x in pool)
            and pool[0] == gpu_uuid and request.get('Capabilities') == [['gpu']],
            'kernel must use the first actual physical GPU allocation')
    return value['Id'], pool


def validate_battery(value, battery, admission, seed_sha, *, expected_sources=None):
    require(value.get('schema_version') == 3 and value.get('protocol') == 'minference-v3-integrity-battery-v1'
            and value.get('seed_sha256') == seed_sha and value.get('candidate_execution') is False,
            'fresh current-source battery required')
    require(value.get('source_admission_sha256') == digest(read(Path(admission.__file__).resolve())), 'battery admission source differs')
    cases = value.get('cases')
    require(isinstance(cases, list) and len(cases) == 12 and {c.get('label') for c in cases} == set(EXPECTED_RUNTIME),
            'exact 12 control matrix required')
    checked = []
    for case in cases:
        label = case['label']; relative = Path(case['source_file'])
        require(relative == Path('candidates') / label / 'submission.py', 'control source path differs')
        raw = read(battery / relative)
        require(digest(raw) == case.get('source_sha256') and len(raw) == case.get('source_bytes'),
                'control source bytes differ')
        if expected_sources is not None:
            require(raw == expected_sources[label], 'control differs from current builder and current seed')
        errors = admission.validate_source_text(raw.decode(), str(relative))
        expected = label not in ('hidden-reference-access', 'hidden-gpu-memory')
        require(case.get('expected_admission') is expected and (not errors) is expected,
                'current source admission differs from fixed control expectation')
        checked.append((case, raw, errors))
    return checked


def validate_capture(bundle, plan, single, sources, *, image_id, gpu_uuid, recipe):
    single.validate_bundle(bundle, plan)
    require(plan.get('protocol') == single.PROTOCOL and plan.get('partition') == 'final'
            and bundle.get('purpose') == 'private-judge' and bundle.get('gpu_uuid', '').lower().removeprefix('gpu-') == gpu_uuid.lower().removeprefix('gpu-'),
            'fresh private on-demand bundle required')
    bindings = plan['bindings']
    require(bindings.get('image_id') == image_id and bindings.get('model') == recipe['model']
            and bindings.get('attention_backend') == 'flash_attention_2'
            and bindings.get('flash_attention_version') == '2.8.3', 'actual preparation model/image/backend differs')
    require(bindings.get('capture_helper_sha256') == sources['activations/recipe/capture_activations.py']
            and bindings.get('on_demand_suite_sha256') == sources['runtime/on_demand_suite_v3.py']
            and bindings.get('on_demand_sources') == {name: sources['runtime/' + name] for name in RUNTIME_FILES},
            'actual on-demand source epoch differs')
    require(not any('sha256' in f or 'size' in f for f in plan['fixtures'].values()),
            'historical persistent QKV identities cannot qualify this runner')


def transport(bundle, plan, handles, identities, policy, policy_sha, candidate_sha, nonce):
    require(set(handles) == set(bundle['fixtures']) == set(identities), 'exact five FD membership required')
    require(type(nonce) is int and 0 < nonce < 2**128, 'private nonzero row nonce required')
    return {'bundle': bundle, 'plan': plan, 'policy': policy, 'policy_sha256': policy_sha,
        'candidate_sha256': candidate_sha, 'nonce': nonce,
        'fds': {fid: {'fd': h.fd, 'header': h.header, 'payload_offset': h.payload_offset,
                     'identity': identities[fid]} for fid, h in handles.items()}}


def verified_drop(dropper):
    require(os.geteuid() == 0, 'candidate child must start as root')
    account = pwd.getpwnam('agent'); require(account.pw_uid == 1000 and account.pw_gid == 1000, 'agent identity differs')
    dropper()
    nnp = ctypes.CDLL(None, use_errno=True).prctl(39, 0, 0, 0, 0)
    require(os.getresuid() == (1000, 1000, 1000) and os.getresgid() == (1000, 1000, 1000)
            and os.getgroups() == [] and nnp == 1, 'actual permanent agent drop differs')
    return {'execution_uid': 1000, 'execution_gid': 1000, 'no_new_privs': True,
            'candidate_privilege_drop': 'permanent-agent', 'supplementary_groups': []}


def child(args):
    require(os.geteuid() == 0, 'root setup is mandatory')
    raw = sys.stdin.buffer.read(1024**2 + 1); sys.stdin.close()
    require(0 < len(raw) <= 1024**2, 'bounded private stdin required')
    value = parse(raw); modules = trusted_modules(args.trusted_root)
    single, archive, runtime = modules['single'], modules['archive'], modules['runtime']
    bundle, plan = value['bundle'], value['plan']; single.validate_bundle(bundle, plan)
    require(digest(read(args.candidate)) == value['candidate_sha256'], 'sealed candidate differs')
    require(not modules['admission'].validate_source_text(read(args.candidate).decode(), str(args.candidate)),
            'child source admission rejected')
    require(set(value['fds']) == set(bundle['fixtures']), 'five retained descriptors required')
    handles = {}; drop = {'execution_uid': 0, 'candidate_privilege_drop': False}
    try:
        for fid, descriptor in value['fds'].items():
            fd = descriptor['fd']; entry = bundle['fixtures'][fid]
            require(type(fd) is int and fd >= 3 and identity(os.fstat(fd)) == descriptor['identity'], 'inherited FD identity differs')
            require(archive.digest_fd(fd, entry['size']) == entry['sha256'], 'inherited FD bytes differ')
            handles[fid] = archive.ActivationHandle(fid, entry, fd, descriptor['header'], descriptor['payload_offset'])
        import torch
        require(torch.__version__ == '2.8.0+cu129', 'Torch epoch differs')
        hardware = runtime.hardware_inventory(torch)
        require(normalized_gpu(hardware['uuid']) == normalized_gpu(bundle['gpu_uuid']),
                'actual visible GPU differs from capture')
        def before_import():
            drop.update(verified_drop(modules['worker']._drop_to_agent))
        try:
            report = modules['measure'].measure_prompt(args.candidate, bundle=bundle, plan=plan, handles=handles,
                identities={fid: item['identity'] for fid, item in value['fds'].items()}, policy=value['policy'],
                policy_sha256=value['policy_sha256'], expected_candidate_sha256=value['candidate_sha256'],
                nonce=value['nonce'], runtime=runtime, archive=archive, torch=torch,
                before_candidate_import=before_import, purpose='private-judge')
            result = {'completed': True, 'report': report, 'hardware': hardware, **drop}; rc = 0
        except runtime.RuntimeFailure as exc:
            result = {'completed': False, 'failure_category': exc.category.value, 'hardware': hardware, **drop}; rc = 1
        require(not modules['worker']._contains_private_randomness(result), 'private randomness in report')
        sys.stdout.buffer.write(single.canonical_bytes(result)); sys.stdout.flush()
        return rc
    finally:
        for handle in handles.values(): handle.close()


def kill_owned_process(process):
    try: os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError: pass
    process.wait(timeout=5)


def classify(value, rc, *, case, bundle, plan, policy, policy_sha, modules):
    require(type(rc) is int and rc in (0, 1), 'unexpected child exit')
    require(normalized_gpu(value.get('hardware', {}).get('uuid')) == normalized_gpu(bundle['gpu_uuid']), 'actual child GPU differs')
    require(value.get('execution_uid') == 1000 and value.get('execution_gid') == 1000
            and value.get('candidate_privilege_drop') == 'permanent-agent'
            and value.get('no_new_privs') is True and value.get('supplementary_groups') == [], 'permanent-drop proof missing')
    if value.get('failure_category'):
        require(rc == 1 and value.get('completed') is False, 'runtime failure/exit differs')
        return value['failure_category']
    require(rc == 0 and value.get('completed') is True, 'completed report/exit differs')
    report = value.get('report', {}); single = modules['single']; runtime = modules['runtime']
    require(report.get('candidate_sha256') == case['source_sha256'] and report.get('bundle_sha256') == single.canonical_sha(bundle)
            and report.get('plan_sha256') == single.canonical_sha(plan) and report.get('numerical_policy_sha256') == policy_sha
            and report.get('purpose') == 'private-judge' and report.get('execution_uid') == 1000
            and report.get('candidate_privilege_drop') is True, 'loaded candidate/bundle/plan/policy differs')
    trials = report.get('trials'); pid = bundle['prompt_id']
    require(isinstance(trials, list) and len(trials) == 4
            and [r.get('fixture_id') for r in trials] == [pid + '-L' + str(l) for l in (1, 4, 15, 27)], 'all four target layers required')
    passed = True
    for row in trials:
        expected = bundle['fixtures'][row['fixture_id']]
        require(row.get('prompt_id') == pid and row.get('input_token_sha256') == expected['input_token_sha256']
                and row.get('actual_n') == expected['actual_n'] and row.get('full_write_ok') is True
                and row.get('input_immutable_ok') is True, 'real input/write/immutable proof differs')
        for key in ('seconds', 'host_seconds', 'event_seconds'):
            require(type(row.get(key)) in (int, float) and math.isfinite(row[key]) and row[key] > 0, 'finite actual timing required')
        require(row['seconds'] == max(row['host_seconds'], row['event_seconds']), 'host/event maximum timing differs')
        for name, rows in [('uniform_stats', 32 * min(expected['actual_n'], 512)), ('tail_stats', 32 * min(expected['actual_n'], 64))]:
            stats = row.get(name)
            require(isinstance(stats, dict) and stats.get('finite') is True and type(stats.get('rows')) is int
                    and stats['rows'] == rows and all(type(stats.get(k)) in (int, float) and math.isfinite(stats[k])
                    and stats[k] >= 0 for k in ('mean', 'head_max_mean', 'tail', 'max', 'bad')), 'complete finite probe statistics required')
        group = next(g for g in plan['case_groups'] if row['fixture_id'] in g['trials'])
        limits = runtime.case_limits(policy, group)
        numeric = runtime.numeric_passes(row['uniform_stats'], row['tail_stats'], limits)
        require(row.get('limits') == limits and row.get('numeric_passed') is numeric, 'numeric policy must be independently recomputed')
        passed = passed and numeric
    return 'numeric-pass' if passed else 'numeric-fail'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--trusted-root', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, help=argparse.SUPPRESS)
    for name in ('battery', 'bundle-root', 'plan', 'numerical-policy', 'inspection', 'seed-source', 'builder-source', 'source-resources', 'output'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--image-id'); parser.add_argument('--gpu-uuid')
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args(argv)
    require(args.trusted_root.is_absolute() and not any(p.is_symlink() for p in (args.trusted_root, *args.trusted_root.parents)), 'canonical trusted root required')
    if args.child: return child(args)
    require(all(getattr(args, n) for n in ('battery', 'bundle_root', 'plan', 'numerical_policy', 'inspection', 'seed_source', 'builder_source', 'source_resources', 'output', 'image_id', 'gpu_uuid')),
            'explicit fresh battery, genuine bundle, plan, frozen policy, actual image/GPU inspection and output required')
    require(1 <= args.timeout <= 1800 and (args.plan_only or os.geteuid() == 0), 'root execution and bounded deadline required')
    runner_path = Path(__file__).resolve(); runner_raw = read(runner_path)
    modules = trusted_modules(args.trusted_root); single = modules['single']
    inspection_raw = read(args.inspection, private=True)
    cid, pool = validate_inspection(parse(inspection_raw), image_id=args.image_id, gpu_uuid=args.gpu_uuid)
    plan_raw = read(args.plan, private=True); plan = parse(plan_raw)
    bundle_raw = read(args.bundle_root / 'bundle.json', private=True); bundle = parse(bundle_raw)
    recipe = parse(read(modules['paths']['activations/recipe/RULER_V3_RECIPE.json']))
    validate_capture(bundle, plan, single, modules['sources'], image_id=args.image_id, gpu_uuid=args.gpu_uuid, recipe=recipe)
    policy_raw = read(args.numerical_policy, private=True)
    policy, policy_sha = modules['runtime'].load_numerical_policy(args.numerical_policy)
    require(digest(policy_raw) == policy_sha, 'actual policy raw SHA differs')
    battery_raw = read(args.battery / 'cases.json')
    seed_raw = read(args.seed_source); builder_raw = read(args.builder_source)
    source_resources_raw = read(args.source_resources)
    battery_value = parse(battery_raw)
    require(battery_value.get('builder_sha256') == digest(builder_raw), 'actual battery builder identity differs')
    builder = builder_from_raw(args.builder_source, builder_raw)
    expected_sources = {label: source.encode() for label, source, *_ in builder.sources(seed_raw.decode())}
    cases = validate_battery(battery_value, args.battery, modules['admission'],
        digest(seed_raw), expected_sources=expected_sources)
    require(not args.output.exists() and args.output.is_absolute() and '..' not in args.output.parts
            and not any(p.is_symlink() for p in args.output.parents), 'fresh canonical private output required')
    args.output.mkdir(mode=0o700)
    summary = {'schema_version': 4, 'protocol': PROTOCOL, 'status': 'plan-only' if args.plan_only else 'running',
        'task_id': 'minference_sparse_prefill_h100_seed_v3',
        'qualified': False, 'source_integrity_scope': 'finite controls, not population rates or downstream QA',
        'runner_sha256': digest(runner_raw), 'source_sha256': modules['sources'],
        'battery_sha256': digest(battery_raw), 'seed_sha256': digest(seed_raw), 'builder_sha256': digest(builder_raw),
        'source_resources_sha256': digest(source_resources_raw),
        'fixed_plan_raw_sha256': digest(plan_raw), 'bundle_raw_sha256': digest(bundle_raw),
        'fixed_plan_sha256': single.canonical_sha(plan), 'bundle_sha256': single.canonical_sha(bundle),
        'fixed_plan': plan, 'authenticated_bundle': bundle, 'numerical_policy_sha256': policy_sha,
        'actual_container_id': cid, 'image_id': args.image_id, 'gpu_uuid': args.gpu_uuid,
        'pool_gpu_uuids': pool, 'inspection_raw_sha256': digest(inspection_raw),
        'ephemeral_five_file_bundle': True, 'persistent_qkv_archive_used': False, 'plan_only': args.plan_only,
        'ownership_requirement': 'Caller independently attests image, GPU lease/watchdog and this container; runner never reserves or cleans containers.',
        'controls': []}
    for name, raw in [('inspection.json', inspection_raw), ('fixed-plan.json', plan_raw), ('bundle.json', bundle_raw), ('battery.json', battery_raw)]:
        fd = os.open(args.output / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
    if args.plan_only:
        summary['controls'] = [{'label': c['label'], 'candidate_sha256': c['source_sha256'], 'executed': False,
                                'source_admission_passed': not errors} for c, _, errors in cases]
        summary['bundle_fd_authentication'] = 'not-run-plan-only'
        write(args.output / 'summary.json', summary); return 0
    handles, identities = single.open_bundle(args.bundle_root, bundle, plan, modules['archive'])
    write(args.output / 'authenticated-fixture-identities.json', {
        'bundle_sha256': single.canonical_sha(bundle), 'plan_sha256': single.canonical_sha(plan),
        'five_fd_full_sha_and_native_header_authenticated_before_children': True,
        'identities': identities})
    summary['bundle_fd_authentication'] = 'all-five-complete-sha-and-native-header-before-children'
    nonce = secrets.randbits(128) or 1
    write(args.output / 'evaluation_nonce.json', {'evaluation_nonce': nonce, 'bundle_sha256': single.canonical_sha(bundle),
          'trial_randomness_protocol': modules['runtime'].TRIAL_RANDOMNESS_PROTOCOL})
    account = pwd.getpwnam('agent')
    try:
        with tempfile.TemporaryDirectory(prefix='minference-on-demand-controls-', dir='/tmp') as temporary:
            stage = Path(temporary); stage.chmod(0o711)
            for case, raw, errors in cases:
                label = case['label']
                if errors:
                    summary['controls'].append({'label': label, 'executed': False,
                        'candidate_sha256': case['source_sha256'], 'actual_outcome': 'source-admission-rejected',
                        'source_admission_passed': False, 'matches_expected_runtime': True})
                    continue
                approved = modules['admission'].admit_workspace(args.battery / 'candidates' / label)
                require(approved.submission_bytes == raw and approved.submission_sha256 == case['source_sha256'], 'approved literal source differs')
                sealed = modules['loader'].seal_approved_submission(approved, stage / label, directory_mode=0o511)
                sealed.path.chmod(0o444)
                cache = stage / (label + '-cache'); cache.mkdir(mode=0o700); os.chown(cache, account.pw_uid, account.pw_gid)
                value = transport(bundle, plan, handles, identities, policy, policy_sha, sealed.sha256, nonce)
                env = dict(os.environ); env.pop('PYTHONPATH', None)
                env.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, TMPDIR=str(cache), TRITON_CACHE_DIR=str(cache / 'triton'), PYTHONDONTWRITEBYTECODE='1')
                started = time.monotonic(); process = None
                try:
                    process = subprocess.Popen([sys.executable, '-I', str(Path(__file__).resolve()), '--child',
                        '--trusted-root', str(args.trusted_root), '--candidate', str(sealed.path)],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        pass_fds=tuple(h.fd for h in handles.values()), start_new_session=True, env=env)
                    stdout, stderr = modules['stream'].communicate_bounded(process, single.canonical_bytes(value), args.timeout)
                    for suffix, data in [('stdout', stdout), ('stderr', stderr)]:
                        fd = os.open(args.output / (label + '.' + suffix), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                        with os.fdopen(fd, 'wb') as f: f.write(data)
                    require(str(nonce).encode() not in stdout + stderr, 'child disclosed private nonce')
                    report = parse(stdout)
                    require(not modules['worker']._contains_private_randomness(report), 'private randomness in child report')
                    outcome = classify(report, process.returncode, case=case, bundle=bundle, plan=plan,
                                       policy=policy, policy_sha=policy_sha, modules=modules)
                    record = {'label': label, 'executed': True, 'candidate_sha256': case['source_sha256'],
                        'returncode': process.returncode, 'actual_outcome': outcome, 'source_admission_passed': True,
                        'matches_expected_runtime': outcome in EXPECTED_RUNTIME[label], 'report': report}
                except Exception as exc:
                    record = {'label': label, 'executed': True, 'candidate_sha256': case['source_sha256'],
                        'matches_expected_runtime': False, 'actual_outcome': 'runner-failure', 'error_type': type(exc).__name__}
                finally:
                    if process is not None: kill_owned_process(process)
                record['elapsed_seconds'] = time.monotonic() - started
                summary['controls'].append(record)
                write(args.output / (label + '.record.json'), record)
                if record['actual_outcome'] == 'runner-failure': break
    finally:
        for handle in handles.values(): handle.close()
    try:
        verify_stable_sources(modules, [(runner_path, runner_raw, False), (args.plan, plan_raw, True),
            (args.bundle_root / 'bundle.json', bundle_raw, True), (args.numerical_policy, policy_raw, True),
            (args.inspection, inspection_raw, True), (args.seed_source, seed_raw, False),
            (args.builder_source, builder_raw, False), (args.source_resources, source_resources_raw, False),
            (args.battery / 'cases.json', battery_raw, False)]
            + [(args.battery / case['source_file'], raw, False) for case, raw, _ in cases])
        summary['source_postflight_stable'] = True
    except Exception as exc:
        summary.update(source_postflight_stable=False, error_type=type(exc).__name__)
    summary['coverage'] = {'declared_controls': 12, 'executed_admitted': sum(c['executed'] for c in summary['controls']),
        'admission_only': sum(not c['executed'] for c in summary['controls']), 'selected_controls': 12, 'complete_control_matrix': len(summary['controls']) == 12}
    summary['status'] = 'passed' if summary['source_postflight_stable'] and len(summary['controls']) == 12 and all(c['matches_expected_runtime'] for c in summary['controls']) else 'failed'
    write(args.output / 'summary.json', summary)
    print(json.dumps({'status': summary['status'], 'controls': len(summary['controls']), 'qualified': False}))
    return 0 if summary['status'] == 'passed' else 1


if __name__ == '__main__':
    try: raise SystemExit(main())
    except Exception:
        print('on-demand control runner failed closed', file=sys.stderr)
        raise SystemExit(1)
