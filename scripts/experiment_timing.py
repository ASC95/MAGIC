#!/usr/bin/env python

import opendssdirect as dss
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. Configuration & Setup
# ==========================================
master_file = '/Users/austinchang/programming/repositories/MAGIC/data/smartds/2018/GSO/rural/scenarios/base_timeseries/opendss/rhs2_1247/rhs2_1247--rdt1264/Master.dss'

# --- CONFIGURATIONS ---
# 1. Time Shift: Set to -15 to shift a 16:00 peak back to 15:45 (aligns to start of block)
time_shift_minutes = -15 

# 2. Substation Rating: Since the feeder uses an infinite bus (Vsource), define the rating here.
# Update this to match the exact rating from Transformers.dss (the Transformers.dss file right ABOVE the feeder I care about)
sub_trafo_kva = 16000.0

# Compile the OpenDSS model
dss.Command(f"Compile '{master_file}'")

# Find the Feeder Head (Vsource)
dss.Vsources.First()
source_name = dss.Vsources.Name()

# Add a monitor directly to the Vsource
dss.Command(f"New Monitor.Sub_Mon element=Vsource.{source_name} terminal=1 mode=1 ppolar=no")

# Print Substation Transformer Info to Terminal
print("\n" + "="*40)
print("SUBSTATION TRANSFORMER INFO")
print("="*40)
print(f"Feeder Head Node:   Vsource.{source_name}")
print(f"Rated Capacity:     {sub_trafo_kva:,.0f} kVA (User Configured)")
print("="*40 + "\n")

# ==========================================
# 2. Run the Yearly Simulation
# ==========================================
intervals = 35040
dss.Command("Set Mode=Yearly")
dss.Command("Set StepSize=15m")
dss.Command(f"Set Number={intervals}") 

print("Solving full year in OpenDSS engine...")
dss.Command("Solve")
print("Simulation complete. Extracting data...\n")

# ==========================================
# 3. Dynamic Data Extraction
# ==========================================
dss.Monitors.Name("Sub_Mon")
num_channels = dss.Monitors.NumChannels()

total_kw = np.zeros(intervals)
total_kvar = np.zeros(intervals)

# Dynamically sum all phases based on the Vsource
for i in range(1, num_channels + 1):
    channel_data = np.array(dss.Monitors.Channel(i))
    if i % 2 != 0:
        total_kw += channel_data
    else:
        total_kvar += channel_data

total_kva = np.sqrt(total_kw**2 + total_kvar**2)

# Generate the DatetimeIndex and apply the user-configured time shift
time_index = pd.date_range(start="2018-01-01 00:00:00", periods=intervals, freq="15min")
time_index = time_index + pd.Timedelta(minutes=time_shift_minutes)

df = pd.DataFrame({'Demand_kW': abs(total_kw), 'Demand_kVA': total_kva}, index=time_index)

# Calculate Percentage Loading based on the configured Substation kVA
df['Loading_Pct'] = (df['Demand_kVA'] / sub_trafo_kva) * 100

# ==========================================
# 4. Identify Peak
# ==========================================
annual_peak_kva = df['Demand_kVA'].max()
peak_timestamp = df['Demand_kVA'].idxmax()
peak_loading_pct = df.loc[peak_timestamp, 'Loading_Pct']

print("-" * 40)
print(f"Annual Peak Timestamp: {peak_timestamp}")
print(f"Annual Peak Demand:    {annual_peak_kva:,.2f} kVA")
print(f"Peak Loading:          {peak_loading_pct:.2f}% of Rated Capacity")
print("-" * 40 + "\n")

# ==========================================
# 5. Interactive Visualization (Plotly)
# ==========================================
start_window = peak_timestamp - pd.Timedelta(days=2, hours=12)
end_window = peak_timestamp + pd.Timedelta(days=2, hours=12)
df_window = df.loc[start_window:end_window]

# Create the hoverable Plotly figure
fig = go.Figure()

# Add Active Power (kW) trace
fig.add_trace(go.Scatter(
    x=df_window.index, 
    y=df_window['Demand_kW'],
    mode='lines',
    name='Active Power',
    hovertemplate='%{y:,.1f} kW',
    line=dict(color='blue', width=2)
))

# Add Apparent Power (kVA) trace with custom hover data for the loading percentage
fig.add_trace(go.Scatter(
    x=df_window.index, 
    y=df_window['Demand_kVA'],
    mode='lines',
    name='Apparent Power',
    customdata=df_window['Loading_Pct'],
    hovertemplate='%{y:,.1f} kVA<br><b>Substation Loading:</b> %{customdata:.2f}%',
    line=dict(color='red', width=2, dash='dash')
))

# Highlight the Peak
fig.add_annotation(
    x=peak_timestamp,
    y=annual_peak_kva,
    text=f"Annual Peak<br>{annual_peak_kva:,.0f} kVA<br>({peak_loading_pct:.1f}% Load)",
    showarrow=True,
    arrowhead=2,
    arrowsize=1.5,
    arrowwidth=2,
    arrowcolor="black",
    ax=50,
    ay=-60,
    font=dict(size=12, color="black"),
    bgcolor="white",
    opacity=0.9,
    bordercolor="black",
    borderwidth=1
)

# Format the interactive layout
fig.update_layout(
    title=f"Feeder rhs2_1247--rdt1264: 5-Day Demand Centered on Annual Peak",
    xaxis_title="Date / Time",
    yaxis_title="Power",
    hovermode="x unified", # Creates a single interactive vertical line with combined tooltips
    template="plotly_white",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

# Render the interactive HTML plot
fig.show()

# - Save it
fig.write_html("rhs2_1247_annual_peak.html")