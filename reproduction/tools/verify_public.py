#!/usr/bin/env python3
"""Verify file identity and run the scientific checks from a ZIP download."""
from __future__ import annotations
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import unicodedata
import zipfile

MAX_FILE = 256 * 1024 * 1024
MAX_TOTAL = 2 * 1024 * 1024 * 1024
MANIFEST = 'FILE_MANIFEST.json'
TEXT_SUFFIXES = {'.py', '.md', '.txt', '.json', '.jsonl', '.csv', '.yaml', '.yml', '.tex', '.bib', '.bbl', '.cff', '.log', '.sha256', '.sh', '.toml', '.cfg'}
SENSITIVE = re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\bAKIA[0-9A-Z]{16}\b|(?:/Users/|/home/)[A-Za-z0-9_.-]+/')

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def members_checked(members):
    if not members or len(members) > 12000:
        raise ValueError('unsupported member count')
    seen, roots, total = set(), set(), 0
    for item in members:
        name = item.filename
        raw = getattr(item, 'orig_filename', name)
        clean = name[:-1] if name.endswith('/') else name
        parts = clean.split('/')
        if raw != name or '\x00' in name or '\\' in name or ':' in name or not clean or any(x in {'', '.', '..'} or unicodedata.normalize('NFKC', x) in {'', '.', '..'} or '/' in unicodedata.normalize('NFKC', x) for x in parts):
            raise ValueError('noncanonical member path')
        key = unicodedata.normalize('NFKC', clean).casefold()
        if key in seen or any(x.casefold() in {'.git', '.svn', '.hg', '__macosx', '.ds_store'} or x.startswith('._') for x in parts):
            raise ValueError('duplicate or unsupported member')
        seen.add(key)
        roots.add(parts[0])
        mode = item.external_attr >> 16
        kind = stat.S_IFMT(mode)
        expected_kind = stat.S_IFDIR if item.is_dir() else stat.S_IFREG
        if kind not in {0, expected_kind} or mode & 0o7000 or item.flag_bits & 1 or item.extra or item.comment:
            raise ValueError('unsupported member attributes')
        if item.file_size < 0 or item.file_size > MAX_FILE:
            raise ValueError('member size exceeds limit')
        total += item.file_size
    if len(roots) != 1 or total > MAX_TOTAL:
        raise ValueError('unsupported root or total size')
    return next(iter(roots))

def check_files(root):
    document = json.loads((root / MANIFEST).read_text())
    expected = {}
    for item in document['files']:
        rel = item['path']
        parts = PurePosixPath(rel).parts
        if not parts or rel.startswith('/') or '\\' in rel or ':' in rel or any(p in {'.', '..'} for p in parts) or str(PurePosixPath(rel)) != rel or rel in expected or rel == MANIFEST:
            raise ValueError('invalid manifest path')
        expected[rel] = item
    observed = {}
    for p in root.rglob('*'):
        if p.is_symlink():
            raise ValueError('symlink in extracted tree')
        if p.is_file() and p.relative_to(root).as_posix() != MANIFEST:
            observed[p.relative_to(root).as_posix()] = p
        elif not p.is_file() and not p.is_dir():
            raise ValueError('special file in extracted tree')
    if set(observed) != set(expected):
        raise ValueError('missing or unexpected files')
    for rel, p in observed.items():
        item = expected[rel]
        if p.stat().st_size != item['size'] or digest(p) != item['sha256']:
            raise ValueError('file identity mismatch: ' + rel)
        if p.suffix.lower() in TEXT_SUFFIXES or not p.suffix:
            if SENSITIVE.search(p.read_bytes()):
                raise ValueError('personal path or credential material: ' + rel)
    return len(observed) + 1

def run_reproduction(root, output, env):
    pins = (root / 'requirements.txt').read_text().splitlines()
    requirements = [pin.split('==') for pin in pins]
    if any(len(parts) != 2 for parts in requirements):
        raise ValueError('requirements.txt must contain exact package versions')
    try:
        installed = all(metadata.version(name) == version for name, version in requirements)
    except metadata.PackageNotFoundError:
        installed = False
    replay_args = [str(root / 'reproduce.py'), '--output', str(output)]
    if installed:
        subprocess.run([sys.executable, *replay_args], cwd=root, env=env, check=True, timeout=1200)
        return
    install_help = 'Activate a Python 3.12 environment with the versions in requirements.txt or populate the uv cache with those wheels.'
    uv = shutil.which('uv')
    if uv is None:
        raise SystemExit(install_help)
    resolver = [uv, 'run', '--quiet', '--offline', '--no-config', '--no-project',
                '--isolated', '--no-python-downloads', '--python', sys.executable,
                '--with-requirements', str(root / 'requirements.txt'), 'python']
    # An isolated uv environment may cease to exist when uv exits. Run the
    # scientific command inside its lifetime; all nonzero exits propagate.
    print('Using the local offline uv cache for the pinned numerical dependencies.', flush=True)
    result = subprocess.run([*resolver, *replay_args], cwd=root, env=env, timeout=1200)
    if result.returncode:
        print(install_help, file=sys.stderr)
        raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--files-only', action='store_true')
    args = parser.parse_args()
    archive = args.archive.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix='scientific-replay-') as tmp:
        target = Path(tmp)
        with zipfile.ZipFile(archive, mode='r') as source:
            if source.comment:
                raise ValueError('archive comment is unsupported')
            members = source.infolist()
            root_name = members_checked(members)
            for item in members:
                p = target / item.filename
                if item.is_dir():
                    p.mkdir(parents=True, exist_ok=True)
                else:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    with source.open(item, mode='r') as src, p.open('wb') as dst:
                        shutil.copyfileobj(src, dst)
        root = target / root_name
        count = check_files(root)
        if not args.files_only:
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TZ='UTC')
            out=target / 'tables.json'
            run_reproduction(root, out, env)
            check_files(root)
        print(json.dumps({'files_checked': count, 'file_identity': 'PASS', 'scientific_replay': 'NOT_RUN' if args.files_only else 'PASS'}))

if __name__ == '__main__':
    main()
