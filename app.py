import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import pickle
import plotly.graph_objs as go
import base64
import io
import os
from functions import interactive_XRD_shift, extract_coordinates

# Load reference data
ref_path = r"Z:\P110143-phosphosulfides-Andrea\Data\Analysis\guidal\XRD\ref_database\reflections"
with open(os.path.join(ref_path, "reflections.pkl"), 'rb') as f:
    ref_peaks_df = pickle.load(f)

# Initialize the Dash app
app = dash.Dash(__name__)

# Initialize an empty dictionary for points
points = {}

# Define the layout of the app
app.layout = html.Div(children=[
    html.H1(children='Hello, World!'),
    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=False
        ),
        html.Button('Show Plot', id='show-plot-button', n_clicks=0),
        html.Div(id='file-content'),
        dcc.Graph(id='xrd-plot'),
        html.Div(id='points-checklist-container'),
        dcc.Input(id='phase-name', type='text', placeholder='Enter phase name'),
        html.Button('Add Phase', id='add-phase-button', n_clicks=0),
        html.Div(id='pickle-output')
    ])
])

# Callback to load the pickle file and show the plot
@app.callback(
    Output('file-content', 'children'),
    Output('xrd-plot', 'figure'),
    Output('points-checklist-container', 'children'),
    Input('upload-data', 'contents'),
    Input('show-plot-button', 'n_clicks'),
    State('upload-data', 'filename')
)
def load_file(contents, plot_clicks, filename):
    global points
    if contents is None:
        return html.Div(), go.Figure(), html.Div()

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    data = None

    try:
        data = pickle.load(io.BytesIO(decoded))
    except Exception as e:
        return html.Div([
            html.H3('Error Loading File'),
            html.Pre(str(e))
        ]), go.Figure(), html.Div()

    ctx = dash.callback_context
    if not ctx.triggered:
        return html.Div(), go.Figure(), html.Div()

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'show-plot-button' and plot_clicks > 0:
        try:
            # Define coordinates you want to plot
            x, y = extract_coordinates(data)
            
            # Assuming data contains the necessary variables for the function
            fig = interactive_XRD_shift(data, '2θ (°)', 'Corrected Intensity', 
                                        shift=400, x=x, y=y, ref_peaks_df=ref_peaks_df, title="mittma_")
            
            for i in range(len(x)):
                points[i] = {'coordinates': {'x': x[i], 'y': y[i]}, 'phases': []}

            checklist_options = [{'label': f'Point {i}', 'value': i} for i in points]

            # Create a 5x5 grid layout for the checklist
            grid_layout = []
            for i in range(0, len(checklist_options), 5):
                row = html.Div(
                    children=[
                        dcc.Checklist(
                            id=f'checklist-{i}',
                            options=checklist_options[i:i+5],
                            value=[j for j in range(i, min(i+5, len(points)))],
                            labelStyle={'display': 'block'}
                        )
                    ],
                    style={'display': 'flex', 'flexDirection': 'row'}
                )
                grid_layout.append(row)

            return html.Div([
                html.H3('File Loaded Successfully'),
                html.Pre(str(data))
            ]), fig, html.Div(grid_layout, style={'display': 'grid', 'gridTemplateColumns': 'repeat(5, 1fr)'})
        except Exception as e:
            return html.Div([
                html.H3('Error Generating Plot'),
                html.Pre(str(e))
            ]), go.Figure(), html.Div()

    return html.Div(), go.Figure(), html.Div()

# Callback to add phase to selected points and pickle the points dictionary
@app.callback(
    Output('pickle-output', 'children'),
    Input('add-phase-button', 'n_clicks'),
    State('phase-name', 'value'),
    State('points-checklist-container', 'children')
)
def add_phase(n_clicks, phase_name, checklist_container):
    global points
    if n_clicks > 0 and phase_name:
        selected_points = []
        for checklist in checklist_container:
            selected_points.extend(checklist['props']['children'][0]['props']['value'])

        for point_id in selected_points:
            points[point_id]['phases'].append(phase_name)

        # Pickle the updated points dictionary
        pickled_points = pickle.dumps(points)
        print(pickled_points)

        return html.Div([
            html.H3('Phase Added Successfully'),
            html.Pre(str(points))
        ])

    return html.Div()

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)