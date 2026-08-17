#!/usr/bin/env python3


'''
- Install the three MAGIC code packages on Ubuntu, each into its own virtual environment inside this repository
    - PyMAGIC/.venv: pyenv Python 3.13, .venv created with poetry
    - omf/.venv: pyenv Python 3.9, .venv created with venv
    - loadshape_upsampling/.venv: pyenv Python 3.13, .venv created with venv
- The script assumes the user has already installed python3 and has sudo
    - To ensure that the prerequisites are installed, run $ sudo apt-get update && DEBIAN_FRONTEND=noninteractive sudo apt-get install -y python3 git
      openssh-client
- PyMAGIC is not part of this repository, and it is private, so the user must download it and place it at the root of this repository themselves.
  utils/pymagic/patch.py copies omf_observer_federate.py into the PyMAGIC repository and applies the file patches to PyMAGIC's files via the patch_*.py
  scripts in utils/pymagic/src/
- The SMART-DS dataset is not part of this repository either. The script downloads its 2018 GSO rural tree into data/smartds/ with the aws CLI
- The RAE dataset is downloaded from the Harvard Dataverse into data/rae/ right after the PyMAGIC check. Dataverse sometimes blocks scripted downloads, so
  manual download may be necessary from https://doi.org/10.7910/DVN/ZJW4LC
'''


import os
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path


MAGIC_DIR = Path(__file__).resolve().parent
OMF_DIR = MAGIC_DIR / 'omf'
PYMAGIC_DIR = MAGIC_DIR / 'PyMAGIC'
LOADSHAPE_UPSAMPLING_DIR = MAGIC_DIR / 'loadshape_upsampling'
OMF_PYTHON_VERSION = '3.9'
PYMAGIC_PYTHON_VERSION = '3.13'
PYENV = Path.home() / '.pyenv' / 'bin' / 'pyenv'
POETRY = Path.home() / '.local' / 'bin' / 'poetry'
RAE_URL = 'https://dataverse.harvard.edu/api/access/dataset/:persistentId/?persistentId=doi:10.7910/DVN/ZJW4LC&format=original'


def run(command, check=True, **kwargs):
    return subprocess.run(command, shell=True, executable='/bin/bash', check=check, **kwargs)


def main():
    # - PyMAGIC is private, so the user must have downloaded it into the repository root already. 
    run(f'python3 {MAGIC_DIR}/utils/pymagic/patch.py')
    # - Download the RAE dataset with the standard library
    if not (MAGIC_DIR / 'data' / 'rae').exists():
        zip_path = MAGIC_DIR / 'data' / 'rae.zip'
        extract_dir = MAGIC_DIR / 'data' / 'rae-extract'
        try:
            (MAGIC_DIR / 'data').mkdir(exist_ok=True)
            shutil.rmtree(extract_dir, ignore_errors=True)
            request = urllib.request.Request(RAE_URL, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(request) as response, open(zip_path, 'wb') as f:
                shutil.copyfileobj(response, f)
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(extract_dir)
            (extract_dir / 'MANIFEST.TXT').unlink(missing_ok=True)
            if not all((extract_dir / c).is_file() and (extract_dir / c).stat().st_size > 0
                       for c in ('house1_power_blk1.csv', 'house1_power_blk2.csv', 'house2_power_blk1.csv')):
                raise RuntimeError('the downloaded data was empty/incomplete')
            extract_dir.rename(MAGIC_DIR / 'data' / 'rae')
            zip_path.unlink()
        except Exception as error:
            shutil.rmtree(extract_dir, ignore_errors=True)
            zip_path.unlink(missing_ok=True)
            print(f'RAE download failed: {error} (Harvard Dataverse sometimes blocks scripted downloads). Download it manually from '
                  f'https://doi.org/10.7910/DVN/ZJW4LC (original format) and place the files in {MAGIC_DIR}/data/rae/. The install continues '
                  'without the dataset', flush=True)
    # - Download the SMART-DS dataset with the aws CLI
    if not (MAGIC_DIR / 'data' / 'smartds' / '.complete').exists():
        if shutil.which('aws') is None:
            run('sudo env DEBIAN_FRONTEND=noninteractive apt-get update && '
                'sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends awscli')
        # - check=False: aws s3 sync exits nonzero when any transfer failed, and that must not abort the whole install so check=False
        sync = run(f'aws s3 sync "s3://oedi-data-lake/SMART-DS/v1.0/2018/GSO/rural/" {MAGIC_DIR}/data/smartds/2018/GSO/rural/ --no-sign-request',
                   check=False)
        # - Check that download was successful
        rural_dir = MAGIC_DIR / 'data' / 'smartds' / '2018' / 'GSO' / 'rural'
        if sync.returncode == 0 and (rural_dir / 'profiles').is_dir() and (rural_dir / 'scenarios').is_dir():
            (MAGIC_DIR / 'data' / 'smartds' / '.complete').touch()
        else:
            print(f'SMART-DS download failed or is incomplete (aws s3 sync exit code {sync.returncode}). Re-run this script to resume it '
                  '(already-downloaded files are skipped). The install continues without the dataset', flush=True)
    # - Make sure poetry creates its venv in the PyMAGIC directory
    os.environ['POETRY_VIRTUALENVS_IN_PROJECT'] = 'true'
    # - shell=True runs every command with /bin/sh by default
    # - DEBIAN_FRONTEND=noninteractive prevents interactive configuration prompt
    run('sudo env DEBIAN_FRONTEND=noninteractive apt-get update')
    # - The packages pyenv needs to compile Python, plus sudo and the ZeroMQ libraries opendsscmd needs
    run('sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends '
        'git curl sudo ca-certificates make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev '
        'libncursesw5-dev xz-utils tk-dev libxmlsec1-dev libffi-dev liblzma-dev libzstd-dev libxml2-dev libzmq5-dev libczmq-dev')
    # - pipefail makes a failed download abort the install instead of silently piping an empty script into bash
    if not PYENV.exists():
        run('set -o pipefail; curl -fsSL https://pyenv.run | bash')
    run(f'{PYENV} install --skip-existing {PYMAGIC_PYTHON_VERSION}')
    run(f'{PYENV} install --skip-existing {OMF_PYTHON_VERSION}')
    pymagic_python = run(f'{PYENV} prefix {PYMAGIC_PYTHON_VERSION}', capture_output=True, text=True).stdout.strip() + '/bin/python'
    omf_python = run(f'{PYENV} prefix {OMF_PYTHON_VERSION}', capture_output=True, text=True).stdout.strip() + '/bin/python'
    # - pyenv Python runs the Poetry installer
    if not POETRY.exists():
        run(f'set -o pipefail; curl -sSL https://install.python-poetry.org | {pymagic_python} -')
    # - The OMF includes opendsscmd-1.7.4-linux-x64-installer.run but does not install it. SourceForge no longer hosts 1.7.4
    if not shutil.which('opendsscmd'):
        run('curl -fSL -o /tmp/opendsscmd-installer.run '
            'https://downloads.sourceforge.net/project/electricdss/OpenDSSCmd/opendsscmd-1.7.7-linux-x64-installer.run')
        run('chmod +x /tmp/opendsscmd-installer.run && sudo /tmp/opendsscmd-installer.run --mode unattended && rm /tmp/opendsscmd-installer.run')
    # - Create the PyMAGIC virtual environment.
    shutil.rmtree(PYMAGIC_DIR / '.venv', ignore_errors=True)
    run(f'{POETRY} env use {pymagic_python}', cwd=PYMAGIC_DIR)
    run(f'{POETRY} install --no-interaction', cwd=PYMAGIC_DIR)
    # - The helics wheel statically links libstdc++ into libhelics.so and exports that runtime's symbols. Torch dynamically links the system libstdc++.
    #   With both in one process, helics's exported old symbols can win symbol resolution over the real runtime and the process segfaults. LD_PRELOAD puts
    #   the system libstdc++ first in the symbol search order so every library binds to the one newest runtime.
    run(f'{PYMAGIC_DIR}/.venv/bin/python -c "import pymagic, helics, torch"', cwd=PYMAGIC_DIR,
        env={**os.environ, 'LD_PRELOAD': '/usr/lib/x86_64-linux-gnu/libstdc++.so.6'})
    # - Create the omf virtual environment
    shutil.rmtree(OMF_DIR / '.venv', ignore_errors=True)
    run(f'{omf_python} -m venv {OMF_DIR}/.venv')
    run(f'{OMF_DIR}/.venv/bin/python -m pip install -e .', cwd=OMF_DIR)
    run(f'{OMF_DIR}/.venv/bin/python -c "import omf"', cwd=OMF_DIR)
    # - Create the loadshape_upsampling virtual environment
    shutil.rmtree(LOADSHAPE_UPSAMPLING_DIR / '.venv', ignore_errors=True)
    run(f'{pymagic_python} -m venv {LOADSHAPE_UPSAMPLING_DIR}/.venv')
    run(f'{LOADSHAPE_UPSAMPLING_DIR}/.venv/bin/python -m pip install -r requirements.txt', cwd=LOADSHAPE_UPSAMPLING_DIR)
    run(f'{LOADSHAPE_UPSAMPLING_DIR}/.venv/bin/python -c "import config, composer, shapelets, plot_utils"', cwd=LOADSHAPE_UPSAMPLING_DIR / 'src')


if __name__ == '__main__':
    main()
