#!/usr/bin/env python3


'''
- Edit PyMAGIC's src/pymagic/federates/__init__.py in place so the federates package exports run_omf_observer_federate
- Running this script twice is safe
'''


import sys
from pathlib import Path


MAGIC_DIR = Path(__file__).resolve().parent.parent.parent.parent
TARGET = MAGIC_DIR / 'PyMAGIC' / 'src' / 'pymagic' / 'federates' / '__init__.py'


def main():
    if not TARGET.is_file():
        sys.exit(f'Error: {TARGET} does not exist. Download the private PyMAGIC repository into the root of the MAGIC repository first')
    text = TARGET.read_text()
    if 'omf_observer_federate' in text:
        print(f'{TARGET} is already patched', flush=True)
        return
    if not text.strip():
        sys.exit(f'Error: {TARGET} is empty. PyMAGIC has likely moved past the pinned commit in README.md')
    # - The original file ends without a trailing newline, so the import is appended on its own new line
    TARGET.write_text(text.rstrip('\n') + '\nfrom .omf_observer_federate import run_omf_observer_federate')
    print(f'Patched {TARGET}', flush=True)


if __name__ == '__main__':
    main()
