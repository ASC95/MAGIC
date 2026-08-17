#!/usr/bin/env python3


'''
- Edit PyMAGIC's scripts/simulation/Run_simulation.py in place so the simulation launches omf_observer_federate alongside the regular federates
- Running this script twice is safe
'''


import sys
from pathlib import Path


MAGIC_DIR = Path(__file__).resolve().parent.parent.parent.parent
TARGET = MAGIC_DIR / 'PyMAGIC' / 'scripts' / 'simulation' / 'Run_simulation.py'


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
    # - Import the observer federate with the other federate imports, plus the static-metadata helper the launcher calls before starting threads
    text = replace_once(text, '    logger_federate,\n)\n',
                        '    logger_federate,\n'
                        '    omf_observer_federate,\n'
                        ')\n'
                        'from pymagic.federates.omf_observer_federate import collect_static_metadata as _omf_collect_static_metadata\n')
    # - The federate count handed to the HELICS broker must include the observer or the co-simulation hangs waiting for it
    text = replace_once(text, '    # Total federates = core feds + num of active device types\n',
                        '    # Plus omf_observer (subscribes to opendss + writes nreca/* CSVs for OMF)\n'
                        '    # Total federates = core feds + num of active device types\n')
    text = replace_once(text, '        num_federates = 5 + num_device_federates\n',
                        '        num_federates = 6 + num_device_federates # Austin: 5 -> 6 to add omf_observer_federate\n')
    text = replace_once(text, '        num_federates = 3 + num_device_federates\n',
                        '        num_federates = 4 + num_device_federates # Austin: 3 -> 4 to add omf_observer_federate\n')
    # - Create the observer thread right before the logger thread block
    text = replace_once(text, '    # Logger Federate\n',
                        '    # OMF Observer (subscribes to OpenDSS_Federate publications, writes nreca/*.csv).\n'
                        '    # Precompute static metadata HERE from the live dss singleton so the observer\n'
                        "    # thread doesn't need to touch dss at all (which would race with opendss_federate).\n"
                        '    _omf_base_kv_by_bus, _omf_reg_phase_letters = _omf_collect_static_metadata(dss, all_network_buses)\n'
                        '    omf_observer_thread = threading.Thread(\n'
                        '        target=omf_observer_federate.run_omf_observer_federate,\n'
                        '        args=(_omf_base_kv_by_bus, _omf_reg_phase_letters, all_network_buses,),\n'
                        '    )\n'
                        '\n'
                        '    # Logger Federate\n')
    # - Add the observer thread to both federate_threads lists (with and without inverters)
    text = replace_once(text, 'voltage_processor_thread, attack_thread',
                        'voltage_processor_thread, omf_observer_thread, attack_thread')
    text = replace_once(text, 'voltage_processor_thread, logger_thread]',
                        'voltage_processor_thread, omf_observer_thread, logger_thread]')
    TARGET.write_text(text)
    print(f'Patched {TARGET}', flush=True)


if __name__ == '__main__':
    main()
