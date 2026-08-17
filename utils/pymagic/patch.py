#!/usr/bin/env python3


'''
- Install the omf integration into the PyMAGIC repository at the root of the MAGIC repository
    - PyMAGIC is a private repository and is not part of the MAGIC repository, so the user must download it and place it at the root of the MAGIC
      repository
    - omf_observer_federate.py is a new file written for the omf integration, so it is stored in utils/pymagic/src/ and copied into place. Every other
      integration change is a small edit to a private PyMAGIC file, so those edits are applied in place by the patch_*.py scripts in utils/pymagic/src/
- linux_install.py runs this script at install time.
'''


import shutil
import subprocess
import sys
from pathlib import Path


MAGIC_DIR = Path(__file__).resolve().parent.parent.parent
PYMAGIC_DIR = MAGIC_DIR / 'PyMAGIC'
SRC_DIR = Path(__file__).resolve().parent / 'src'
PATCH_SCRIPTS = ['patch_federates_init.py', 'patch_opendss_federate.py', 'patch_run_simulation.py', 'patch_plot_voltages.py']
# - The PyMAGIC commit that is confirmed to work with this repository (also recorded in README.md); the patch scripts were written against this commit's
#   files
PYMAGIC_COMMIT = 'a904a6bb29d8c8fd9d1d2fef1e911ce716e3d050'


def main():
    if not PYMAGIC_DIR.is_dir():
        sys.exit(f'Error: {PYMAGIC_DIR} does not exist. Download the private PyMAGIC repository (github.com/lbnl-cybersecurity/PyMAGIC) and place '
                 'it at the root of the MAGIC repository, then re-run this script')
    # - Pin PyMAGIC to the confirmed-working commit before patching
    toplevel = subprocess.run(['git', '-C', str(PYMAGIC_DIR), 'rev-parse', '--show-toplevel'], capture_output=True, text=True)
    if toplevel.returncode != 0 or Path(toplevel.stdout.strip()) != PYMAGIC_DIR:
        sys.exit(f'Error: {PYMAGIC_DIR} is not a git repository, so the confirmed-working commit cannot be checked out. Replace it with a real '
                 'clone: git clone git@github.com:lbnl-cybersecurity/PyMAGIC.git')
    if subprocess.run(['git', '-C', str(PYMAGIC_DIR), 'cat-file', '-e', f'{PYMAGIC_COMMIT}^{{commit}}'], capture_output=True).returncode != 0:
        subprocess.run(['git', '-C', str(PYMAGIC_DIR), 'fetch', '--all'], check=True)
    subprocess.run(['git', '-C', str(PYMAGIC_DIR), 'checkout', PYMAGIC_COMMIT], check=True)
    destination = PYMAGIC_DIR / 'src' / 'pymagic' / 'federates' / 'omf_observer_federate.py'
    shutil.copy2(SRC_DIR / 'omf_observer_federate.py', destination)
    print(f'Copied {SRC_DIR / "omf_observer_federate.py"} -> {destination}', flush=True)
    for script in PATCH_SCRIPTS:
        subprocess.run([sys.executable, str(SRC_DIR / script)], check=True)


if __name__ == '__main__':
    main()
