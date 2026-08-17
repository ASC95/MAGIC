#!/usr/bin/env python3


'''
- Edit PyMAGIC's src/pymagic/federates/opendss_federate.py in place so it publishes the substation power and regulator tap state that
  omf_observer_federate.py subscribes to
- Running this script twice is safe
'''


import sys
from pathlib import Path


MAGIC_DIR = Path(__file__).resolve().parent.parent.parent.parent
TARGET = MAGIC_DIR / 'PyMAGIC' / 'src' / 'pymagic' / 'federates' / 'opendss_federate.py'


def replace_once(text, old, new):
    if text.count(old) != 1:
        sys.exit(f'Error: expected exactly 1 match in {TARGET} for:\n{old}\nFound {text.count(old)}. PyMAGIC has likely moved past the pinned '
                 'commit in README.md')
    return text.replace(old, new)


def main():
    if not TARGET.is_file():
        sys.exit(f'Error: {TARGET} does not exist. Download the private PyMAGIC repository into the root of the MAGIC repository first')
    text = TARGET.read_text()
    if 'omf_observer_federate' in text:
        print(f'{TARGET} is already patched', flush=True)
        return
    # - Import the two snapshot helpers next to the other imports (json is already imported by the original file)
    text = replace_once(text, 'import opendssdirect as dss\n\n',
                        'import opendssdirect as dss\n'
                        '\n'
                        'from pymagic.federates.omf_observer_federate import (\n'
                        '    omf_substation_snapshot,\n'
                        '    omf_regulator_taps_snapshot,\n'
                        ')\n'
                        '\n')
    # - Register the two publications right after the existing timestamp publication
    text = replace_once(text, '    timestamp_pub = h.helicsFederateRegisterPublication(fed, "timestamp", h.HELICS_DATA_TYPE_DOUBLE, "")\n',
                        '    timestamp_pub = h.helicsFederateRegisterPublication(fed, "timestamp", h.HELICS_DATA_TYPE_DOUBLE, "")\n'
                        '    # OMF_Observer_Federate subscribes to these to write nreca/* CSVs.\n'
                        '    substation_power_pub = h.helicsFederateRegisterPublication(fed, "substation_power", h.HELICS_DATA_TYPE_STRING, "")\n'
                        '    regulator_taps_pub = h.helicsFederateRegisterPublication(fed, "regulator_taps", h.HELICS_DATA_TYPE_STRING, "")\n')
    # - Publish both snapshots every simulation step, just before the existing timestamp publication
    text = replace_once(text, '        h.helicsPublicationPublishDouble(timestamp_pub, current_time)\n',
                        '\n'
                        '        # --- Publish OMF-observer state (substation totals + regulator taps) ---\n'
                        '        h.helicsPublicationPublishString(\n'
                        '            substation_power_pub, json.dumps(omf_substation_snapshot(dss))\n'
                        '        )\n'
                        '        h.helicsPublicationPublishString(\n'
                        '            regulator_taps_pub, json.dumps(omf_regulator_taps_snapshot(dss))\n'
                        '        )\n'
                        '\n'
                        '        h.helicsPublicationPublishDouble(timestamp_pub, current_time)\n')
    TARGET.write_text(text)
    print(f'Patched {TARGET}', flush=True)


if __name__ == '__main__':
    main()
