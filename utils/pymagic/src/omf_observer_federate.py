import json
import os
from collections import defaultdict
import helics as h
import numpy as np
import pandas as pd
from pymagic import config
from pymagic.utils import get_cfg, get_output_dir


'''
- The OMF observer federate outputs 7 csvs:
    - base_voltage.csv: the base L-N kV voltage of each bus given its position in the circuit. This is computed once the dss object is loaded but before
      the simulation actually starts
        - For delta columns/loads, these values can be multiplied by √3 elsewhere to convert from pu to absolute voltages
    - bus_voltages.csv: each column contains a per-unit L-N (wye) or L-L (delta) voltage magnitude. The wiring is determined from bus_connection_type.json
        - Buses with wye-connected elements have a L-N voltage, so the column name only includes one letter (e.g. "150_a")
        - Buses with delta-connected elements have a L-L voltage, so the column name includes two letters (e.g. "799_ab")
    - meter_voltages.csv: contains the same information as bus_voltages.csv, but computes the min, max, and mean voltage of every phase of every bus that
      has a load for a given timestep
    - phase_imbalance.csv: computes the max deviation of any phase magnitude from the bus's mean phase magnitude by taking the max of the absolute value
      of the difference between the mean voltage magnitude and each phase magnitude, then by dividing that absolute value by the mean
    - regulator_taps.csv: tracks the tap positions of the voltage regulators
    - substation_power.csv: tracks the real power, reactive power, and apparent power at the substation. Also tracks total losses (lines + transformers)
      and DERs generation
    - transmission_voltage.csv: tracks the voltage at the swing bus (i.e. the voltage source, which is supposed to stay close to 1) and the downline bus
      (i.e. the bus on the feeder-side of the substation regulator)
'''


def omf_substation_snapshot(dss_module):
    '''Allow the opendss_federate.py to push data to the omf_observer_federate.py at each timestep to populate substation_power.csv'''
    total_power = dss_module.Circuit.TotalPower()
    total_losses = dss_module.Circuit.Losses()
    dg_w = 0.0
    n = dss_module.Generators.Count()
    if n > 0:
        dss_module.Generators.First()
        for _ in range(n):
            dg_w += dss_module.Generators.kW() * 1000.0
            dss_module.Generators.Next()
    n = dss_module.PVsystems.Count()
    if n > 0:
        dss_module.PVsystems.First()
        for _ in range(n):
            dg_w += dss_module.PVsystems.kW() * 1000.0
            dss_module.PVsystems.Next()
    if hasattr(dss_module, "Storages"):
        n = dss_module.Storages.Count()
        if n > 0:
            dss_module.Storages.First()
            for _ in range(n):
                if dss_module.Storages.kW() > 0:
                    dg_w += dss_module.Storages.kW() * 1000.0
                dss_module.Storages.Next()
    return {
        "P_sub_kW": total_power[0],
        "Q_sub_kVAR": total_power[1],
        "P_loss_W": total_losses[0],
        "DG_W": dg_w,
    }


def omf_regulator_taps_snapshot(dss_module):
    '''Allow the opendss_federate.py to push data to the omf_observer_federate.py at each timestep to populate regulator_taps.csv.'''
    taps = {}
    for reg_name in dss_module.RegControls.AllNames() or []:
        dss_module.RegControls.Name(reg_name)
        taps[reg_name] = dss_module.RegControls.TapNumber()
    return taps


def collect_static_metadata(dss_module, all_network_buses):
    '''Get the base kV of each bus, and the regulator phase letters from the OpenDSS object before the simulation starts.'''
    base_kv_by_bus = {}
    for bus in all_network_buses:
        dss_module.Circuit.SetActiveBus(bus)
        base_kv_by_bus[bus] = dss_module.Bus.kVBase()

    reg_phase_letters = {}
    # Maps DSS phase index (1, 2, 3) → A, B, C for column naming
    phase_letter = {1: "A", 2: "B", 3: "C"}
    for reg_name in dss_module.RegControls.AllNames() or []:
        dss_module.RegControls.Name(reg_name)
        xfm_name = dss_module.RegControls.Transformer()
        dss_module.Transformers.Name(xfm_name)
        bus_str = dss_module.CktElement.BusNames()[0]
        phase_parts = bus_str.split(".")[1:]
        letters = []
        for p in phase_parts:
            try:
                pi = int(p)
            except ValueError:
                continue
            letter = phase_letter.get(pi)
            if letter is not None and letter not in letters:
                letters.append(letter)
        if not letters:
            letters = ["A", "B", "C"]
        elif len(letters) != 3:
            # Mark the regulator with its primary phase letter only. Multi-phase regulators get all three letters.
            letters = letters[:1]
        reg_phase_letters[reg_name] = letters

    return base_kv_by_bus, reg_phase_letters


# Phase pairs, named in PyMAGIC's alphabetical order:
#   ab = phase 1 - phase 2
#   bc = phase 2 - phase 3
#   ac = phase 1 - phase 3
DELTA_PAIRS = (
    (1, 2, "ab"),
    (2, 3, "bc"),
    (1, 3, "ac"),
)

def get_bus_connection_type(path):
    '''Get the circuit metadata.'''
    with open(path) as f:
        data = json.load(f)
    if "default_connection" not in data:
        raise ValueError(f"{path}: missing required key 'default_connection' ('delta' or 'wye')")
    return BusConnectionMap(
        circuit_wiring=data["default_connection"],
        connection_exceptions=dict(data.get("connection_exceptions", {})),
        substation_buses=dict(data.get("substation_buses", {})),
    )


class BusConnectionMap:
    '''This class wraps bus_connection_type.json'''
    def __init__(self, circuit_wiring, substation_buses, connection_exceptions=None):
        if circuit_wiring not in ("delta", "wye"):
            raise ValueError(f"default_connection must be 'delta' or 'wye', got {repr(circuit_wiring)}")
        self.circuit_wiring = circuit_wiring
        self.connection_exceptions = dict(connection_exceptions or {})
        self.substation_buses = {k: v for k, v in substation_buses.items() if not k.startswith("_")}
        for pointer in ("swing", "downline"):
            if pointer not in self.substation_buses:
                raise ValueError(f"substation_buses must define {repr(pointer)}")
        for bus, conn in self.connection_exceptions.items():
            if conn not in ("delta", "wye"):
                raise ValueError(f"exception for bus {repr(bus)}: connection must be 'delta' or 'wye', got {repr(conn)}")

    def __getitem__(self, bus):
        return self.connection_exceptions.get(bus, self.circuit_wiring)

    @property
    def swing_bus(self):
        return self.substation_buses["swing"]

    @property
    def downline_bus(self):
        return self.substation_buses["downline"]


def index_voltages_by_bus_phase(columns):
    """Build {bus: {phase_int: column_name}} from column names "<bus>.<phase>"."""
    bus_phases = defaultdict(dict)
    for name in columns:
        parts = name.split(".")
        if len(parts) != 2:
            continue
        bus, phase_s = parts
        try:
            bus_phases[bus][int(phase_s)] = name
        except ValueError:
            continue
    return dict(bus_phases)


def compute_complex_voltages(df_v, bus_conn):
    '''Relabel the complex (i.e. real and imaginary components) voltage columns

    - For wye buses, there is no change in the complex voltage value
        - E.g. 150.1 becomes 150_a
    - For delta buses, convert L-N voltages into L-L voltages with (V_1 - V_2) / √3, (V_2 - V_3) / √3, and (V_1 - V_3) / √3
        - E.g. 799.1 and 799.2 become 799_ab, 799.2 and 799.3 become 799_bc, and 799.1 and 799.3 become 799_ac
    '''
    bus_phases = index_voltages_by_bus_phase(df_v.columns)
    out = {}
    num_to_letter = {1: "a", 2: "b", 3: "c"}
    for bus, phase_cols in bus_phases.items():
        conn = bus_conn[bus]
        if conn == "wye":
            for ph, col in phase_cols.items():
                letter = num_to_letter.get(ph)
                if letter is None:
                    continue
                out[f"{bus}_{letter}"] = df_v[col]
        else:  # delta
            for p1, p2, label in DELTA_PAIRS:
                if p1 in phase_cols and p2 in phase_cols:
                    out[f"{bus}_{label}"] = (df_v[phase_cols[p1]] - df_v[phase_cols[p2]]) / np.sqrt(3.0)
    return out


def compute_voltage_magnitudes(df_v, bus_conn):
    """Same as compute_complex_voltages but returns magnitudes (real Series).

    - .abs() of a complex Series computes the magnitude from the real and imaginary part
    """
    return {k: v.abs() for k, v in compute_complex_voltages(df_v, bus_conn).items()}


def compute_swing_and_downline_voltages(df_v, bus_conn):
    '''Compute data for the "Transmission Voltage" chart.'''
    bus_phases = index_voltages_by_bus_phase(df_v.columns)
    def _get_series(bus):
        phase_cols = bus_phases[bus]
        if bus_conn[bus] == "wye":
            return df_v[list(phase_cols.values())].abs().mean(axis=1, skipna=False)
        # delta: average the available L-L pair magnitudes
        pair_mags = [(df_v[phase_cols[p1]] - df_v[phase_cols[p2]]).abs() / np.sqrt(3.0)
                     for p1, p2, _label in DELTA_PAIRS
                     if p1 in phase_cols and p2 in phase_cols]
        if not pair_mags:
            raise ValueError(f"delta bus {repr(bus)} has no complete phase pair in the published voltages")
        return pd.concat(pair_mags, axis=1).mean(axis=1, skipna=False)

    return {
        "swingVoltage": _get_series(bus_conn.swing_bus),
        "downlineNodeVolts": _get_series(bus_conn.downline_bus),
    }


def compute_downline_node_volts_per_phase(df_v, bus_conn):
    '''Generate the per-phase data for the "Transmission Voltage" chart'''
    bus = bus_conn.downline_bus
    # raises KeyError if the downline bus never published
    phase_cols = index_voltages_by_bus_phase(df_v.columns)[bus]
    out = {}
    num_to_letter = {1: "a", 2: "b", 3: "c"}
    delta_to_phase = {"ab": "A", "bc": "B", "ac": "C"}
    if bus_conn[bus] == "wye":
        for ph_int, col in phase_cols.items():
            letter = num_to_letter.get(ph_int)
            if letter is None:
                continue
            out[letter.upper()] = df_v[col].abs()
    else:
        for p1, p2, label in DELTA_PAIRS:
            if p1 in phase_cols and p2 in phase_cols:
                out[delta_to_phase[label]] = (
                    (df_v[phase_cols[p1]] - df_v[phase_cols[p2]]).abs() / np.sqrt(3.0)
                )
    return out


def compute_meter_aggregates(df_v, bus_conn, meter_buses):
    '''Generate data for the "Triplex Meter Voltage Summary" chart.'''
    mags = compute_voltage_magnitudes(df_v, bus_conn)
    mags = {k: v for k, v in mags.items() if k.rsplit('_', 1)[0] in meter_buses}
    if not mags:
        raise ValueError("load_data.csv and circuit mismatch")
    M = pd.DataFrame(mags)
    return {
        "Min": M.min(axis=1, skipna=False).to_numpy(),
        "Mean": M.mean(axis=1, skipna=False).to_numpy(),
        "Max": M.max(axis=1, skipna=False).to_numpy(),
    }


# ---------------------------------------------------------------------------
# Voltage-translation CSV writing
# ---------------------------------------------------------------------------

def _get_buses_with_loads(data_dir):
    '''Get the buses that have loads, which never includes the downline and swing buses'''
    path = os.path.join(data_dir, "load_data.csv")
    return {c.rsplit("_", 1)[0] for c in pd.read_csv(path, nrows=0).columns}


def _voltage_rows_to_dataframe(voltage_rows):
    '''Get a list of {<node name>: <complex pu voltage>} and return a DataFrame

    - Each row of the DataFrame is a timestep
    - Each column of the DataFrame is a bus and phase, labeled "<bus>.<phase>" (e.g. "799.1")
    '''
    return pd.DataFrame(voltage_rows).astype(complex)


def _compute_phase_imbalance(df_v, all_network_buses, times):
    '''Compute data for phase_imbalance.csv'''
    mags = df_v.abs()
    bus_of_node = [c.rsplit(".", 1)[0] for c in mags.columns]
    g = mags.T.groupby(bus_of_node)
    bus_mean = g.mean().T
    phase_count = g.count().T
    max_dev = (mags - g.transform("mean").T).abs().T.groupby(bus_of_node).max().T
    ratio = (max_dev / bus_mean).where((phase_count >= 2) & (bus_mean > 0))
    ratio = ratio.reindex(columns=all_network_buses)
    ratio.insert(0, "time", times)
    return ratio


def _write_voltage_translation_csvs(nreca_dir, data_dir, times, voltage_rows):
    '''Compute + write the three voltage-translation CSVs: bus_voltages.csv, meter_voltages.csv, and transmission_voltage.csv.'''
    fed_name = "[OMF Observer Federate]"
    bus_conn = get_bus_connection_type(os.path.join(data_dir, "bus_connection_type.json"))

    df_v = _voltage_rows_to_dataframe(voltage_rows)
    mags = compute_voltage_magnitudes(df_v, bus_conn)
    if not mags:
        raise ValueError(f"no voltage keys could be derived from {len(df_v.columns)}")
    pd.DataFrame({"time": times, **mags}).to_csv(
        os.path.join(nreca_dir, "bus_voltages.csv"), index=False
    )
    agg = compute_meter_aggregates(df_v, bus_conn, meter_buses=_get_buses_with_loads(data_dir))
    pd.DataFrame({"time": times, **agg}).to_csv(
        os.path.join(nreca_dir, "meter_voltages.csv"), index=False
    )
    trans_cols = {"time": times}
    trans_cols.update(compute_swing_and_downline_voltages(df_v, bus_conn))
    for letter, series in compute_downline_node_volts_per_phase(df_v, bus_conn).items():
        trans_cols[f"downline_{letter}"] = series
    pd.DataFrame(trans_cols).to_csv(
        os.path.join(nreca_dir, "transmission_voltage.csv"), index=False
    )
    print(f"{fed_name} Wrote nreca/bus_voltages.csv, meter_voltages.csv, "
          f"transmission_voltage.csv")


def run_omf_observer_federate(base_kv_by_bus, reg_phase_letters, all_network_buses):
    '''Run the OMF observer federate.

    Tracks state from OpenDSS_Federate over HELICS, then writes the nreca CSVs after the simulation ends. We can't just use the opendss_federate.py
    because we want to keep our logic separate. HELICS only keeps the current timestep values, so a federate must record the simulation history.
    voltage_rows is duplicated from V_true.npy, but it's necessary not to couple opendss_federate.py and omf_observer_federate.py together
    '''
    cfg = get_cfg()
    delta_t = cfg.time_step
    simulation_time = cfg.simulation_time
    data_dir = os.path.join(config.DATA_DIR, cfg.data_folder)
    fed_name = "[OMF Observer Federate]"

    # --- HELICS federate setup -------------------------------------------------
    fedinfo = h.helicsCreateFederateInfo()
    h.helicsFederateInfoSetCoreName(fedinfo, "OMF_Observer_Federate")
    h.helicsFederateInfoSetCoreTypeFromString(fedinfo, "zmq")
    h.helicsFederateInfoSetTimeProperty(fedinfo, h.HELICS_PROPERTY_TIME_DELTA, delta_t)

    if cfg.real_time_mode:
        h.helicsFederateInfoSetFlagOption(fedinfo, h.HELICS_FLAG_REALTIME, True)
        h.helicsFederateInfoSetTimeProperty(fedinfo, h.HELICS_PROPERTY_TIME_RT_LAG, cfg.real_time_lag)
        h.helicsFederateInfoSetTimeProperty(fedinfo, h.HELICS_PROPERTY_TIME_RT_LEAD, cfg.real_time_lead)

    fed = h.helicsCreateValueFederate("OMF_Observer_Federate", fedinfo)

    # --- Subscriptions to OpenDSS_Federate -----------------------------------
    opendss_timestamp_sub = h.helicsFederateRegisterSubscription(
        fed, "OpenDSS_Federate/timestamp", ""
    )
    complex_voltage_sub = h.helicsFederateRegisterSubscription(
        fed, "OpenDSS_Federate/complex_voltage_out", ""
    )
    substation_power_sub = h.helicsFederateRegisterSubscription(
        fed, "OpenDSS_Federate/substation_power", ""
    )
    regulator_taps_sub = h.helicsFederateRegisterSubscription(
        fed, "OpenDSS_Federate/regulator_taps", ""
    )

    # Own timestamp publication so anyone downstream can synchronize on us too.
    timestamp_pub = h.helicsFederateRegisterPublication(
        fed, "timestamp", h.HELICS_DATA_TYPE_DOUBLE, ""
    )

    # --- Track values -----------------------------------------------
    times = []
    substation_rows = []
    tap_rows = []
    voltage_rows = []

    h.helicsFederateEnterExecutingMode(fed)

    # --- Main HELICS loop -----------------------------------
    current_time = 0.0
    request_mode = h.HELICS_ITERATION_REQUEST_ITERATE_IF_NEEDED

    while current_time < simulation_time:
        computed = False

        while True:
            # Wait until OpenDSS_Federate has published for the current step.
            opendss_ts = h.helicsInputGetDouble(opendss_timestamp_sub)

            if not computed and opendss_ts == current_time:
                # Read subscriptions.
                voltage_json = h.helicsInputGetString(complex_voltage_sub)
                substation_json = h.helicsInputGetString(substation_power_sub)
                taps_json = h.helicsInputGetString(regulator_taps_sub)
                try:
                    sub = json.loads(substation_json) if substation_json else {}
                except Exception as e:
                    print(f"{fed_name} substation_power parse failed at t={current_time}: {e}")
                    sub = {}
                times.append(current_time)
                substation_rows.append({k: sub.get(k, float("nan"))
                                        for k in ("P_sub_kW", "Q_sub_kVAR", "P_loss_W", "DG_W")})

                # - complex_voltage_out is a JSON dict {node_name: complex pu}
                #   - node_name is "<bus>.<phase_int>", e.g. "701.1"
                v_complex = {}
                try:
                    raw_v = json.loads(voltage_json) if voltage_json else {}
                    for node_name, val in raw_v.items():
                        v_complex[node_name] = complex(val["real"], val["imag"]) \
                            if isinstance(val, dict) else complex(val)
                except Exception as e:
                    v_complex = {}
                    print(f"{fed_name} complex_voltage parse failed at t={current_time}: {e}")
                voltage_rows.append(v_complex)

                # ---- regulator taps ----
                reg_row = {"time": current_time}
                try:
                    raw_taps = json.loads(taps_json) if taps_json else {}
                    for reg_name, tap in raw_taps.items():
                        for letter in reg_phase_letters[reg_name]:
                            reg_row[f"{reg_name}_{letter}"] = tap
                except Exception as e:
                    print(f"{fed_name} regulator_taps parse failed at t={current_time}: {e}")
                    for reg_name, letters in reg_phase_letters.items():
                        for letter in letters:
                            reg_row[f"{reg_name}_{letter}"] = float("nan")
                tap_rows.append(reg_row)

                h.helicsPublicationPublishDouble(timestamp_pub, current_time)
                computed = True

            current_time, iter_state = h.helicsFederateRequestTimeIterative(
                fed,
                current_time + delta_t,
                request_mode,
            )

            if iter_state == h.HELICS_ITERATION_RESULT_NEXT_STEP:
                break

    # --- Disconnect HELICS first; file writes happen after ---
    h.helicsFederateDisconnect(fed)
    h.helicsFederateFree(fed)

    # --- Write the four nreca CSVs --------------------------------------
    try:
        nreca_dir = os.path.join(get_output_dir(), "nreca")
        os.makedirs(nreca_dir, exist_ok=True)
        raw = pd.DataFrame(substation_rows, columns=["P_sub_kW", "Q_sub_kVAR", "P_loss_W", "DG_W"])
        # - OpenDSS sets power negative when delivering, so flip the sign
        p_w = -raw["P_sub_kW"] * 1000.0
        q_var = -raw["Q_sub_kVAR"] * 1000.0
        s_va = (p_w**2 + q_var**2) ** 0.5
        pd.DataFrame({
            "time": times,
            "Power Substation (W)": p_w,
            "Losses Total (W)": raw["P_loss_W"],
            "DG Output (W)": raw["DG_W"],
            "Reactive Power Substation (V)": q_var,
            "Apparent Power Substation (VA)": s_va,
            "Substation Power Factor (%)": (p_w / s_va).where(s_va > 0, 0.0),
        }).to_csv(os.path.join(nreca_dir, "substation_power.csv"), index=False)

        _compute_phase_imbalance(_voltage_rows_to_dataframe(voltage_rows), all_network_buses, times).to_csv(
            os.path.join(nreca_dir, "phase_imbalance.csv"), index=False
        )
        pd.DataFrame({
            "bus": list(base_kv_by_bus),
            "kV_base_LN": list(base_kv_by_bus.values()),
        }).to_csv(os.path.join(nreca_dir, "base_voltage.csv"), index=False)
        if tap_rows:
            pd.DataFrame(tap_rows).to_csv(
                os.path.join(nreca_dir, "regulator_taps.csv"), index=False
            )
        else:
            pd.DataFrame(columns=["time"]).to_csv(
                os.path.join(nreca_dir, "regulator_taps.csv"), index=False
            )
        print(f"{fed_name} Wrote nreca/substation_power.csv, "
              f"phase_imbalance.csv, base_voltage.csv, regulator_taps.csv")
    except Exception as e:
        print(f"{fed_name} [ERROR] Could not write nreca/* CSVs: {e}")

    # --- Write the voltage-translation CSVs ----------------------------------
    nreca_dir = os.path.join(get_output_dir(), "nreca")
    os.makedirs(nreca_dir, exist_ok=True)
    _write_voltage_translation_csvs(nreca_dir, data_dir, times, voltage_rows)
