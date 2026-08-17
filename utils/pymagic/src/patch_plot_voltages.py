#!/usr/bin/env python3


'''
- Edit PyMAGIC's scripts/plotting/plot_voltages.py in place to fix two small bugs the omf integration runs into
- Running this script twice is safe
'''


import sys
from pathlib import Path


MAGIC_DIR = Path(__file__).resolve().parent.parent.parent.parent
TARGET = MAGIC_DIR / 'PyMAGIC' / 'scripts' / 'plotting' / 'plot_voltages.py'


def replace_once(text, old, new):
    if text.count(old) != 1:
        sys.exit(f'Error: expected exactly 1 match in {TARGET} for:\n{old}\nFound {text.count(old)}. PyMAGIC has likely moved past the pinned '
                 'commit in README.md')
    return text.replace(old, new)


def main():
    if not TARGET.is_file():
        sys.exit(f'Error: {TARGET} does not exist. Download the private PyMAGIC repository into the root of the MAGIC repository first')
    text = TARGET.read_text()
    if "can't resolve '..' through" in text:
        print(f'{TARGET} is already patched', flush=True)
        return
    # - Create the output directory before the existence check instead of just before saving
    text = replace_once(text, '    if not os.path.exists(voltage_file):\n',
                        "    os.makedirs(output_dir, exist_ok=True) # must exist before the next check: os.path.exists() can't resolve '..' "
                        'through a nonexistent intermediate dir\n'
                        '    if not os.path.exists(voltage_file):\n')
    text = replace_once(text, '        os.makedirs(output_dir, exist_ok=True) # ensure save directory exists\n\n', '')
    # - Report the missing nodes, not the file name
    text = replace_once(text, ': {voltage_file}")\n', ': {missing}")\n')
    TARGET.write_text(text)
    print(f'Patched {TARGET}', flush=True)


if __name__ == '__main__':
    main()
