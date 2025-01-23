# requirements:
# pip install pandas plotly kaleido

import pandas as pd
import plotly.graph_objects as go

def plot_sankey(
    df: pd.DataFrame,
    out_file: str,
    source_col: str = 'Source',
    target_col: str = 'Target',
    value_col: str = 'Value',
    highlight_source: str = None,
    highlight_target: str = None
):
    """Sankey plot from sources to targets
    """
    source_totals = df.groupby(source_col)[value_col].sum()
    target_totals = df.groupby(target_col)[value_col].sum()
    source_nodes = source_totals.index.tolist()
    target_nodes = target_totals.index.tolist()
    nodes = source_nodes + target_nodes
    source_indices = {name: i for i, name in enumerate(nodes)}

    link_colors = ['rgba(0, 0, 255, 0.4)'] * len(df)
    if highlight_source and highlight_target:
        # Highlight the specified path in red
        highlight_mask = (df[source_col] == highlight_source) & (df[target_col] == highlight_target)
        for i, should_highlight in enumerate(highlight_mask):
            if should_highlight:
                link_colors[i] = 'rgba(255, 0, 0, 0.8)'


    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = nodes,
            color = "blue"
        ),
        link = dict(
            source = [source_indices[src] for src in df[source_col]],
            target = [source_indices[tgt] for tgt in df[target_col]],
            value = df[value_col],
            color = link_colors
        )
    )])

    fig.update_layout(
        title='Sankey Diagram',
        font=dict(size=14),
        annotations=[
            dict(
                x=0,
                y=1.1,
                xref="paper",
                yref="paper",
                text="Exporter",
                showarrow=False,
                font=dict(size=18)
            ),
            dict(
                x=1,
                y=1.1,
                xref="paper",
                yref="paper",
                text="Importer",
                showarrow=False,
                font=dict(size=18)
            )
        ]
    )
    fig.write_image(out_file)


# Input file path
input_file = 'test_oil_data.csv'
if input_file.endswith('.xlsx'):
    df = pd.read_excel(input_file)
else:
    df = pd.read_csv(input_file)

# output image filename
out_file = input_file.replace('.xlsx', '').replace('.csv', '') + '-sankey.png'

plot_sankey(
    df, 
    out_file=out_file, 
    source_col='Source', 
    target_col='Target', 
    value_col='Value',
    highlight_source="Saudi Arabia",
    highlight_target="China"
)
