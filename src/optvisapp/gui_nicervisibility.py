import dash
from dash import dcc, html, Input, Output, State, callback_context
import pandas as pd
import numpy as np
from optvisapp import targetvisibilitydetails  # your custom module

# Load data (already saved in Parquet format for speed)
df_nicer_vis_timeflt = pd.read_parquet('test_vis.parquet')
target_brightearth_all_df = pd.read_parquet('test_brightearth.parquet')
target_od_startend_times_all = pd.read_parquet('test_od_startend_times.parquet')
od_startend_times_all = pd.read_parquet('test_odbounds.parquet')

start_time = pd.to_datetime('2025-095T02:57:00', format='%Y-%jT%H:%M:%S', utc=True)
end_time = pd.to_datetime('2025-095T03:58:00', format='%Y-%jT%H:%M:%S', utc=True)

sources_per_page = 100
app = dash.Dash(__name__)

sunangle_min = df_nicer_vis_timeflt['sunangle_start'].min()
sunangle_max = df_nicer_vis_timeflt['sunangle_start'].max()

unique_targets = sorted(df_nicer_vis_timeflt['target_name'].unique())

app.layout = html.Div([
    html.Div(style={'display': 'flex', 'flexDirection': 'row', 'alignItems': 'flex-start', 'width': '100%'}, children=[
    html.Div([
    html.H1("NICER Target Visibilities"),

    html.Label("Filter by Sun Angle (start):"),
    dcc.RangeSlider(
        id='sunangle-slider',
        min=45,
        max=180,
        step=5,
        value=[45, 180],
        marks={i: str(i) for i in range(45, 185, 5)},
        tooltip={"placement": "bottom", "always_visible": True},
    ),

    html.Label("Sort targets by:"),
    dcc.Dropdown(
        id='sort-dropdown', style={'width': '300px'},
        options=[
            {'label': 'Sun Angle', 'value': 'sunangle_start'},
            {'label': 'Visibility Start', 'value': 'vis_start'},
            {'label': 'Visibility End', 'value': 'vis_end'}
        ],
        value='vis_start',  # default
        clearable=False
    ),

    html.Label("Sources per page:"),
    dcc.Dropdown(
        id='page-size-dropdown', style={'width': '200px'},
        options=[
            {'label': str(n), 'value': n} for n in [50, 100, 150, 200, 250, 300]
        ] + [{'label': 'Max', 'value': 'max'}],
        value=100,
        clearable=False
    ),

    dcc.Graph(id='visibility-plot'),

    html.A(
        href='https://heasarc.gsfc.nasa.gov/docs/nicer/',
        target='_blank',
        children=html.Img(
            src='https://heasarc.gsfc.nasa.gov/Images/nicer/NICER_crop_250.jpg',
        style={
            'position': 'absolute',
            'top': '10px',
            'right': '100px',
            'height': '270px',
            'zIndex': '1000'
        })
    ),

    html.Div([
        html.Button("Previous", id='prev-btn', n_clicks=0),
        html.Div(id='page-info', style={'margin': '0 15px'}),
        html.Button("Next", id='next-btn', n_clicks=0),
    ], style={'display': 'flex', 'alignItems': 'center'}),

    dcc.Store(id='current-page', data=1)
    ], style={'width': '72%', 'paddingRight': '20px'}),

    html.Div(
        children=[
            html.Button("Select All", id='select-all-btn', n_clicks=0),
            html.Button("Deselect All", id='deselect-all-btn', n_clicks=0),
            dcc.Checklist(
                id='target-selector',
                options=[{'label': name, 'value': name} for name in unique_targets],
                value=unique_targets,
                labelStyle={'display': 'block'},
                style={
                    'overflowY': 'scroll',
                    'height': '80vh',
                    'width': '300px',
                    'border': '1px solid #ccc',
                    'padding': '10px'
                }
            )
        ],
        style={'width': '28%', 'height': '100vh', 'overflowY': 'auto', 'paddingTop': '355px', 'marginLeft': '50px'}
    )])])


@app.callback(
    Output('target-selector', 'value'),
    [Input('select-all-btn', 'n_clicks'),
     Input('deselect-all-btn', 'n_clicks')],
    prevent_initial_call=True
)
def toggle_target_selection(select_all_clicks, deselect_all_clicks):
    ctx = callback_context
    if ctx.triggered and ctx.triggered[0]['prop_id'].startswith('select-all-btn'):
        return unique_targets
    elif ctx.triggered and ctx.triggered[0]['prop_id'].startswith('deselect-all-btn'):
        return []
    return dash.no_update


@app.callback(
    [Output('visibility-plot', 'figure'),
     Output('page-info', 'children'),
     Output('current-page', 'data')],
    [Input('prev-btn', 'n_clicks'),
     Input('next-btn', 'n_clicks'),
     Input('sunangle-slider', 'value'),
     Input('sort-dropdown', 'value'),
     Input('page-size-dropdown', 'value'),
     Input('target-selector', 'value')],
    [State('current-page', 'data')]
)
def update_plot(prev_clicks, next_clicks, sa_range, sort_field, page_size, selected_targets, current_page):
    ctx = callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    if button_id == 'next-btn':
        current_page += 1
    elif button_id == 'prev-btn':
        current_page -= 1

    # Filter by sunangle_start and selected targets
    df_filtered = df_nicer_vis_timeflt[
        (df_nicer_vis_timeflt['sunangle_start'] >= sa_range[0]) &
        (df_nicer_vis_timeflt['sunangle_start'] <= sa_range[1]) &
        (df_nicer_vis_timeflt['target_name'].isin(selected_targets))
    ]

    # Sort globally by selected column
    df_filtered = df_filtered.sort_values(by=sort_field, ascending=False)

    if page_size == 'max':
        page_size = len(df_filtered) if len(df_filtered) > 0 else 1

    max_page = max(1, int(np.ceil(len(df_filtered) / page_size)))
    current_page = max(1, min(current_page, max_page))

    start_idx = (current_page - 1) * page_size
    end_idx = current_page * page_size
    filtered_nicer_vis = df_filtered.iloc[start_idx:end_idx]

    filtered_targets = filtered_nicer_vis['target_name'].unique()
    filtered_brightearth = target_brightearth_all_df[
        target_brightearth_all_df['srcname'].isin(filtered_targets)]
    filtered_od_startend = target_od_startend_times_all[
        target_od_startend_times_all['target_name'].isin(filtered_targets)]

    fig = targetvisibilitydetails.visibilityplot_plotly(
        filtered_nicer_vis,
        filtered_brightearth,
        filtered_od_startend,
        od_startend_times_all,
        start_time,
        end_time,
        freq_bound=60,
        outputFile=None
    )

    info_text = f"Page {current_page} of {max_page}"
    return fig, info_text, current_page


if __name__ == '__main__':
    app.run(debug=True)
