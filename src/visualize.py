#!/usr/bin/env python
import yaml
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import paths
from data_processor import ShapeletProcessor


def main():
    # - Examine an appliance
    appliance_name = 'light16'
    with open(paths.APPLIANCES_FILE_PATH) as f:
        app_full_config = yaml.safe_load(f)
    with open(paths.TRAINING_CONFIG_FILE_PATH) as f:
        train_cfg = yaml.safe_load(f)
    graph_end = train_cfg['graph_end']
    app_config = app_full_config[appliance_name]

    processor = ShapeletProcessor(app_config)
    df_raw = processor.get_ampds2_data()
    real_shapelets, df_viz = processor.extract_shapelets(df_raw)
    graph_appliance_timeline(appliance_name, df_raw, df_viz, df_viz, graph_end)


def save_metrics_table_png(df: pd.DataFrame, filename: str = "evaluation_summary.png"):
    """
    Renders the metrics DataFrame as a high-quality table and saves to PNG.
    """
    if df.empty:
        print("No data to save for table.")
        return

    # --- TRANSPOSE & FORMAT ---
    # Transpose so columns become rows
    df_t = df.T
    # The original column headers are now the 'index'. 
    # We must reset_index() to move them into a standard data column.
    df_t.reset_index(inplace=True)
    # Rename that new column to "Metric" (or "Parameter")
    df_t.rename(columns={'index': 'Metric'}, inplace=True)
    # Use this processed dataframe for plotting
    plot_df = df_t

    # Define layout for table
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(plot_df.columns),
                    fill_color='paleturquoise',
                    align='left',
                    font=dict(size=12, color='black')),
        cells=dict(values=[plot_df[k].tolist() for k in plot_df.columns],
                   fill_color='lavender',
                   align='left',
                   font=dict(size=11, color='black'))
    )])

    fig.update_layout(
        title="<b>VAE Evaluation Summary Metrics</b>",
        width=1000, # Wide enough to fit all 12 columns
        height=400 + (len(df) * 30) # Dynamic height
    )

    output_path = paths.OUTPUTS_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Requires 'kaleido' package
    fig.write_image(str(output_path), scale=2)
    print(f"Metrics table saved to {output_path}")

def plot_best_worst_matches(appliance_name: str, match_data: dict):
    """
    Plots the Best (closest) and Worst (furthest) synthetic matches against their real counterparts.
    """
    if not match_data: return

    best_vae = match_data['best_vae']
    best_real = match_data['best_real']
    worst_vae = match_data['worst_vae']
    worst_real = match_data['worst_real']

    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=(f"Best Match (DTW: {match_data['best_dist']:.2f})", 
                                        f"Worst Match (DTW: {match_data['worst_dist']:.2f})"))

    # --- Plot Best Match ---
    fig.add_trace(go.Scatter(y=best_real, mode='lines', name='Nearest Real',
                             line=dict(color='blue', dash='dot'), showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(y=best_vae, mode='lines', name='Synthetic',
                             line=dict(color='green', width=2), showlegend=True), row=1, col=1)

    # --- Plot Worst Match ---
    fig.add_trace(go.Scatter(y=worst_real, mode='lines', name='Nearest Real',
                             line=dict(color='blue', dash='dot'), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(y=worst_vae, mode='lines', name='Synthetic',
                             line=dict(color='red', width=2), showlegend=True), row=1, col=2)

    fig.update_layout(
        title=f"<b>{appliance_name}: Synthetic Fidelity Check</b>",
        template='plotly_white',
        height=500,
        width=1200
    )

    fig.update_yaxes(rangemode="tozero")
    
    # Save or Show
    # output_path = paths.OUTPUTS_DIR / f"{appliance_name}_best_worst.png"
    # fig.write_image(str(output_path))
    fig.show()

def graph_appliance_timeline(appliance_name: str, 
                    df_actual: pd.DataFrame, 
                    df_shapelet: pd.DataFrame, 
                    df_prior: pd.DataFrame,
                    graph_end):
    """Preserved logic for the timeline view."""
    if graph_end is not None:
        df_actual = df_actual[:graph_end]
        df_shapelet = df_shapelet[:graph_end]
        df_prior = df_prior[:graph_end]

    traces = [
        {'name': f'{appliance_name} Actual', 'df': df_actual, 'color': 'blue'},
        {'name': f'{appliance_name} Extracted', 'df': df_shapelet, 'color': 'red'},
        {'name': f'{appliance_name} Generated', 'df': df_prior, 'color': 'green'},
    ]
    fig = go.Figure()
    for tc in traces:
        col = tc['df'].columns[0]
        fig.add_trace(go.Scatter(name=tc['name'], x=tc['df'].index, y=tc['df'][col],
                                 mode='lines', line=dict(color=tc['color'])))
    fig.update_layout(title=f'{appliance_name} Timeline Comparison', template='plotly_white')
    fig.show()


if __name__ == '__main__':
    main()