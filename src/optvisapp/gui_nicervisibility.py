import argparse
import dash
from dash import dcc, html, Input, Output, State, callback_context
import pandas as pd
import numpy as np
from optvisapp import targetvisibilitydetails  # your custom module
import threading
import webbrowser
import os
import gdown

def create_app(context):
    """Create and return a Dash app using the data in the context."""
    app = dash.Dash(__name__)
    server = app.server  # Expose the Flask server if needed

    # Extract data from context for convenience
    unique_targets = context["unique_targets"]

    app.layout = html.Div([
        html.Div(style={'display': 'flex', 'flexDirection': 'row', 'alignItems': 'flex-start', 'width': '100%'}, children=[
            html.Div([
                html.H1("NICER Target Visibilities between {} and {}".format(context["start_time"].strftime("%Y-%m-%d %H:%M:%S"),
                                                                              context["end_time"].strftime("%Y-%m-%d %H:%M:%S"))),

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
                    id='sort-dropdown',
                    style={'width': '300px'},
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
                    id='page-size-dropdown',
                    style={'width': '200px'},
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
                            'top': '2%',
                            'right': '0.8%',
                            'height': '32%',
                            'zIndex': '1000'
                        }
                    )
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
                            'width': '20vw',
                            'border': '1px solid #ccc',
                            'padding': '10px'
                        }
                    )
                ],
                style={'width': '25%', 'height': '100vh', 'overflowY': 'auto', 'paddingTop': '21%', 'marginLeft': '5%'}
            )
        ])
    ])

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

        # Access data from the context dictionary
        df_nicer_vis_timeflt = context["df_nicer_vis_timeflt"]

        # Filter by sunangle_start and selected targets
        df_filtered = df_nicer_vis_timeflt[
            (df_nicer_vis_timeflt['sunangle_start'] >= sa_range[0]) &
            (df_nicer_vis_timeflt['sunangle_start'] <= sa_range[1]) &
            (df_nicer_vis_timeflt['target_name'].isin(selected_targets))
        ]
        df_filtered = df_filtered.sort_values(by=sort_field, ascending=False)

        if page_size == 'max':
            page_size = len(df_filtered) if len(df_filtered) > 0 else 1

        max_page = max(1, int(np.ceil(len(df_filtered) / page_size)))
        current_page = max(1, min(current_page, max_page))

        start_idx = (current_page - 1) * page_size
        end_idx = current_page * page_size
        filtered_nicer_vis = df_filtered.iloc[start_idx:end_idx]

        filtered_targets = filtered_nicer_vis['target_name'].unique()
        target_brightearth_all_df = context["target_brightearth_all_df"]
        filtered_brightearth = target_brightearth_all_df[
            target_brightearth_all_df['srcname'].isin(filtered_targets)]
        target_od_startend_times_all = context["target_od_startend_times_all"]
        filtered_od_startend = target_od_startend_times_all[
            target_od_startend_times_all['target_name'].isin(filtered_targets)]

        od_startend_times_all = context["od_startend_times_all"]
        start_time = context["start_time"]
        end_time = context["end_time"]

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

    return app

def main():
    parser = argparse.ArgumentParser(description="NICER Target Visibilities App")
    parser.add_argument("-vs", "--visibilities", help="Path to full target visibilities", default=None)
    parser.add_argument("-br", "--brightearth", help="Path to target bright-earth angle", default=None)
    parser.add_argument("-os", "--od_startend", help="Path to target start/end orbit-day visibilities",
                        default=None)
    parser.add_argument("-ob", "--odbounds", help="ISS orbit-day bounds", default=None)
    parser.add_argument("-st", "--start_time", help="Start time in format Y-jTH:M:S "
                                                    "(e.g., 2025-075T00:00:00)", default=None)
    parser.add_argument("-et", "--end_time", help="End time in format Y-jTH:M:S "
                                                  "(e.g., 2025-075T00:00:00)", default=None)
    parser.add_argument("-dp", "--downloadparquet", help="Download parquet file",
                        default=False, type=bool, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    if args.downloadparquet:
        folderid = "17hDfD34ljxtsz9JvBNXbSXY1ZyANtgWn?usp=sharing"
        gdown.download_folder(id=folderid)
        os.system('mv ./visibilities_files/*.parquet .')
        os.rmdir('visibilities_files')
        # making sure all parquet files are renamed to downloaded files
        args.visibilities = 'visibilities_vis.parquet'
        args.brightearth = 'visibilities_brightearth.parquet'
        args.od_startend = 'visibilities_od_startend_times.parquet'
        args.odbounds = 'visibilities_odbounds.parquet'
    else: # If download is false, parse through the parquet files
        # If no arguments are supplied, check if generic .parquet files are in directory
        if args.visibilities is None:
            args.visibilities = 'visibilities_vis.parquet'
        if args.brightearth is None:
            args.brightearth = 'visibilities_brightearth.parquet'
        if args.od_startend is None:
            args.od_startend = 'visibilities_od_startend_times.parquet'
        if args.odbounds is None:
            args.odbounds = 'visibilities_odbounds.parquet'

    # Except if any of the expected .parquet files are not found
    if ((not os.path.exists(args.visibilities)) or (not os.path.exists(args.brightearth)) or
            (not os.path.exists(args.od_startend)) or (not os.path.exists(args.odbounds))):
        raise Exception("Cannot find one or more of the .parquet files ({}, {}, {}, or {})".format(
            args.visibilities, args.brightearth, args.od_startend, args.odbounds))

    # Place .parquet files in dictionary
    context = {"df_nicer_vis_timeflt": pd.read_parquet(args.visibilities),
               "target_brightearth_all_df": pd.read_parquet(args.brightearth),
               "target_od_startend_times_all": pd.read_parquet(args.od_startend),
               "od_startend_times_all": pd.read_parquet(args.odbounds)}

    # Deals with start and end times
    if args.start_time is None:
        context["start_time"] = context["df_nicer_vis_timeflt"]['vis_start'].min()
    else:
        context["start_time"] = pd.to_datetime(args.start_time, format='%Y-%jT%H:%M:%S', utc=True)

    if args.end_time is None:
        context["end_time"] = context["df_nicer_vis_timeflt"]['vis_end'].max()
    else:
        context["end_time"] = pd.to_datetime(args.end_time, format='%Y-%jT%H:%M:%S', utc=True)

    context["sunangle_min"] = context["df_nicer_vis_timeflt"]['sunangle_start'].min()
    context["sunangle_max"] = context["df_nicer_vis_timeflt"]['sunangle_start'].max()
    context["unique_targets"] = sorted(context["df_nicer_vis_timeflt"]['target_name'].unique())

    app = create_app(context)

    def open_browser():
        webbrowser.open_new("http://127.0.0.1:8050")

    # Schedule browser to open 1 second after the server starts:
    threading.Timer(1, open_browser).start()

    app.run(debug=False, host='0.0.0.0', port=8050)

if __name__ == '__main__':
    main()
