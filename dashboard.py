import json

import dash
import numpy as np
import pandas as pd
import plotly.express as px
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State


class DataExplorerDashboard:
    def __init__(self, dataframe):
        """
        Initialize the dashboard with a pandas DataFrame

        :param dataframe: Input pandas DataFrame to explore
        """
        self.df = dataframe
        self.app = dash.Dash(__name__)

        # Categorize columns by type
        self.categorize_columns()

        # Generate DataFrame description
        self.df_description = self.generate_dataframe_description()

        # Build the layout
        self.create_layout()

        # Set up callbacks
        self.create_callbacks()

    def categorize_columns(self):
        """
        Categorize columns by their data types for dynamic UI generation
        """
        self.numeric_cols = list(self.df.select_dtypes(include=[np.number]).columns)
        self.categorical_cols = list(self.df.select_dtypes(include=['object', 'category']).columns)
        self.datetime_cols = list(self.df.select_dtypes(include=['datetime64']).columns)

        # Additional categorization for datetime flexibility
        self.datetime_resolutions = {
            'day': lambda x: x.dt.date,
            'week': lambda x: x.dt.to_period('W').dt.start_time,
            'month': lambda x: x.dt.to_period('M').dt.start_time,
            'year': lambda x: x.dt.to_period('Y').dt.start_time
        }

    def generate_dataframe_description(self):
        """
        Generate a comprehensive description of the DataFrame

        :return: Dictionary with DataFrame description
        """
        description = {
            "overall": {
                "total_rows": int(len(self.df)),
                "total_columns": int(len(self.df.columns))
            },
            "columns": {}
        }

        for col in self.df.columns:
            col_info = {
                "dtype": str(self.df[col].dtype),
                "non_null_count": int(self.df[col].count()),
                "null_count": int(self.df[col].isnull().sum())
            }

            # Additional details based on data type
            if isinstance(self.df[col].dtype, pd.CategoricalDtype) or \
                    pd.api.types.is_object_dtype(self.df[col]):
                unique_values = self.df[col].unique().tolist()
                col_info["unique_values"] = unique_values[:10]  # Limit to 10 unique values
                col_info["total_unique_values"] = int(len(unique_values))

            elif pd.api.types.is_numeric_dtype(self.df[col]):
                col_info.update({
                    "min": float(self.df[col].min()),
                    "max": float(self.df[col].max()),
                    "mean": float(self.df[col].mean()),
                    "median": float(self.df[col].median()),
                    "std": float(self.df[col].std())
                })

            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                col_info.update({
                    "min_date": str(self.df[col].min()),
                    "max_date": str(self.df[col].max()),
                    "date_range": str(self.df[col].max() - self.df[col].min())
                })

            description["columns"][col] = col_info

        return description

    def create_layout(self):
        """
        Create the Dash layout with dynamic dropdowns, controls, and description modal
        """
        self.app.layout = html.Div([
            # Description Modal (Moved to top of the layout)
            html.Div([
                html.Div([
                    html.Div([
                        html.H2('DataFrame Description', style={'display': 'inline-block', 'marginRight': '20px'}),
                        html.Button('×', id='close-description-modal',
                                    style={
                                        'display': 'inline-block',
                                        'float': 'right',
                                        'fontSize': '24px',
                                        'background': 'none',
                                        'border': 'none',
                                        'cursor': 'pointer'
                                    })
                    ], style={'borderBottom': '1px solid #ccc', 'marginBottom': '15px', 'paddingBottom': '10px'}),

                    # Overall DataFrame Info
                    html.Div([
                        html.H3('Overall DataFrame'),
                        html.P(f"Total Rows: {self.df_description['overall']['total_rows']}"),
                        html.P(f"Total Columns: {self.df_description['overall']['total_columns']}")
                    ]),

                    # Scrollable Columns Description
                    html.Div(
                        [html.Div([
                            html.H4(col),
                            html.Pre(json.dumps(details, indent=2))
                        ]) for col, details in self.df_description['columns'].items()],
                        style={
                            'maxHeight': '400px',
                            'overflowY': 'scroll',
                            'border': '1px solid #ccc',
                            'padding': '10px',
                            'backgroundColor': '#f9f9f9'
                        }
                    )
                ], style={
                    'backgroundColor': 'white',
                    'padding': '20px',
                    'borderRadius': '5px',
                    'maxWidth': '800px',
                    'margin': 'auto',
                    'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
                    'position': 'relative'
                })
            ], id='description-modal-content', style={'display': 'none',
                                                      'position': 'fixed',
                                                      'zIndex': 1000,
                                                      'left': 0,
                                                      'top': 0,
                                                      'width': '100%',
                                                      'height': '100%',
                                                      'overflow': 'auto',
                                                      'backgroundColor': 'rgba(0,0,0,0.4)'}),

            # DataFrame Description Button
            html.Button('Show DataFrame Description',
                        id='open-description-modal',
                        style={'margin': '10px'}),

            # Rest of the dashboard layout remains the same
            html.H1('Interactive Data Explorer', style={'textAlign': 'center'}),

            # Plot Type Selection
            html.Div([
                html.Label('Plot Type:'),
                dcc.Dropdown(
                    id='plot-type-dropdown',
                    options=[
                        {'label': 'Scatter', 'value': 'scatter'},
                        {'label': 'Line', 'value': 'line'},
                        {'label': 'Bar', 'value': 'bar'},
                        {'label': 'Boxplot', 'value': 'boxplot'},
                        {'label': 'Histogram', 'value': 'histogram'}  # Added histogram
                    ],
                    value='scatter'
                )
            ], style={'width': '48%', 'display': 'inline-block'}),

            # X-Axis Column Selection
            html.Div([
                html.Label('X-Axis Column:'),
                dcc.Dropdown(
                    id='x-axis-dropdown',
                    options=[
                        {'label': col, 'value': col} for col in
                        sorted(self.numeric_cols + self.categorical_cols + self.datetime_cols)
                    ],
                    value=self.numeric_cols[0] if self.numeric_cols else None
                )
            ], style={'width': '48%', 'display': 'inline-block'}),

            # Y-Axis Column Selection
            html.Div([
                html.Label('Y-Axis Column:'),
                dcc.Dropdown(
                    id='y-axis-dropdown',
                    options=[
                        {'label': col, 'value': col} for col in
                        sorted(self.numeric_cols + self.categorical_cols)
                    ],
                    value=self.numeric_cols[1] if len(self.numeric_cols) > 1 else None
                )
            ], style={'width': '48%', 'display': 'inline-block'}),

            # Categorical Breakdown Dropdown
            html.Div([
                html.Label('Breakdown Column:'),
                dcc.Dropdown(
                    id='breakdown-dropdown',
                    options=[
                        {'label': col, 'value': col} for col in
                        self.categorical_cols
                    ],
                    value=self.categorical_cols[0] if self.categorical_cols else None
                )
            ], style={'width': '48%', 'display': 'inline-block'}),

            # Datetime Resolution Dropdown (conditionally rendered)
            html.Div([
                html.Label('Datetime Resolution:'),
                dcc.Dropdown(
                    id='datetime-resolution-dropdown',
                    options=[
                        {'label': res.capitalize(), 'value': res} for res in self.datetime_resolutions.keys()
                    ],
                    value='month'
                )
            ], id='datetime-resolution-container', style={'width': '48%', 'display': 'none'}),

            # Aggregation Method Dropdown (conditionally rendered)
            html.Div([
                html.Label('Aggregation Method:'),
                dcc.Dropdown(
                    id='aggregation-dropdown',
                    options=[
                        {'label': 'None', 'value': 'None'},
                        {'label': 'Count', 'value': 'count'},
                        {'label': 'Distinct Count', 'value': 'nunique'},
                        {'label': 'Sum', 'value': 'sum'},
                        {'label': 'Average', 'value': 'mean'},
                        {'label': 'Min', 'value': 'min'},
                        {'label': 'Max', 'value': 'max'}
                    ],
                    value='None'
                )
            ], id='aggregation-container', style={'width': '48%', 'display': 'none'}),
            # Bin Selection (conditionally rendered)
            html.Div([
                html.Label('Number of Bins:'),
                dcc.Input(
                    id='histogram-bins-input',
                    type='number',
                    min=1,
                    max=100,
                    step=1,
                    value=10
                )
            ], id='histogram-bins-container', style={'width': '48%', 'display': 'none'}),

            # Plot
            html.Div([
                # Parent container with flex layout
                html.Div([
                    # Left content: Graph
                    html.Div([
                        dcc.Graph(id='data-plot')
                    ], style={'flex': '3', 'padding': '10px'}),  # Adjust flex ratio for width

                    # Right content: Filters pane
                    html.Div([
                        html.H3('Filters', style={'textAlign': 'center'}),

                        html.Div([
                            html.Div([
                                html.Label(f'Filter: {col}'),
                                dcc.Dropdown(
                                    id=f'filter-{col}',
                                    options=[{'label': str(value), 'value': value} for value in
                                             self.df[col].dropna().unique()],
                                    multi=True
                                )
                            ], style={'marginBottom': '10px'}) for col in sorted(self.df.columns) if
                            self.df[col].nunique() > 1 and self.df[col].nunique() <= 10
                        ])

                    ], style={
                        'flex': '1',  # Adjust flex ratio for width
                        'padding': '10px',
                        'backgroundColor': '#f9f9f9',
                        'borderLeft': '1px solid #ccc',
                        'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
                    })
                ], style={
                    'display': 'flex',
                    'flexDirection': 'row',  # Arrange children in a row
                    'height': '100vh'  # Full height of the viewport
                })
            ]),
        ])

    def create_callbacks(self):
        """
        Create Dash callbacks for dynamic interactions and modal
        """

        # Modal interaction callback
        @self.app.callback(
            Output('description-modal-content', 'style'),
            [Input('open-description-modal', 'n_clicks'),
             Input('close-description-modal', 'n_clicks')],
            [State('description-modal-content', 'style')]
        )
        def toggle_modal(open_clicks, close_clicks, current_style):
            """
            Toggle the visibility of the description modal
            """
            # Ensure proper initialization
            if open_clicks is None:
                open_clicks = 0
            if close_clicks is None:
                close_clicks = 0

            # If open button is clicked more times than close button, show the modal
            if open_clicks > close_clicks:
                return {
                    'display': 'block',
                    'position': 'fixed',
                    'zIndex': 1000,
                    'left': 0,
                    'top': 0,
                    'width': '100%',
                    'height': '100%',
                    'overflow': 'auto',
                    'backgroundColor': 'rgba(0,0,0,0.4)'
                }
            # Otherwise, hide the modal
            return {'display': 'none'}

        # Combined callback for plot updates and interaction
        @self.app.callback(
            Output('data-plot', 'figure', allow_duplicate=True),
            [
                Input('plot-type-dropdown', 'value'),
                Input('x-axis-dropdown', 'value'),
                Input('y-axis-dropdown', 'value'),
                Input('breakdown-dropdown', 'value'),
                Input('datetime-resolution-dropdown', 'value'),
                Input('aggregation-dropdown', 'value'),
                Input('data-plot', 'clickData'),
                Input('histogram-bins-input', 'value')  # Add bins input as a trigger
            ] +
            [Input(f'filter-{col}', 'value') for col in self.df.columns if
             self.df[col].nunique() > 1 and self.df[col].nunique() <= 10],
            [State('data-plot', 'figure')],
            prevent_initial_call=True
        )
        def update_plot_and_interaction(plot_type, x_column, y_column, breakdown_column,
                                        datetime_resolution, aggregation_method, clickData,
                                        bins, *args):
            """
            Dynamically generate plots based on user selections, filters, and handle trace visibility.
            """
            # Split args into filter_values and figure
            filter_values = args[:-1]  # All arguments except the last are filter values
            figure = args[-1]  # The last argument is the figure from State

            ctx = dash.callback_context
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

            # Initialize working DataFrame
            working_df = self.df.copy()

            # Apply filters dynamically
            for col, values in zip(self.df.columns, filter_values):
                if values:  # If there are selected values for the column
                    working_df = working_df[working_df[col].isin(values)]

            # Validate breakdown_column
            if breakdown_column not in working_df.columns:
                breakdown_column = None  # Set to None if the column is not valid

            # Plot generation logic
            if trigger_id in ['plot-type-dropdown', 'x-axis-dropdown', 'y-axis-dropdown',
                              'breakdown-dropdown', 'datetime-resolution-dropdown', 'aggregation-dropdown',
                              'histogram-bins-input'] + \
                    [f'filter-{col}' for col in self.df.columns]:
                # Handle datetime resolution if applicable
                if x_column in self.datetime_cols:
                    resolution_func = self.datetime_resolutions.get(datetime_resolution,
                                                                    self.datetime_resolutions['month'])
                    working_df['x_resolved'] = resolution_func(working_df[x_column])
                    x_column = 'x_resolved'

                # Handle categorical x-axis with aggregation
                if (x_column in self.categorical_cols or plot_type == "line") and aggregation_method != "None":
                    grouped = working_df.groupby([x_column, breakdown_column] if breakdown_column else [x_column])
                    agg_func = getattr(grouped[y_column], aggregation_method)
                    grouped_df = agg_func().reset_index()

                    # Calculate error bars (standard deviation as an example)
                    grouped_df['error'] = grouped[y_column].std().reset_index()[y_column]
                else:
                    grouped_df = working_df
                    grouped_df['error'] = None

                # Plotting logic based on plot type
                if plot_type == 'scatter':
                    fig = px.scatter(grouped_df, x=x_column, y=y_column, color=breakdown_column)
                elif plot_type == 'line':  # Line plot with aggregated values and error bars
                    fig = px.line(
                        grouped_df,
                        x=x_column,
                        y=y_column,
                        color=breakdown_column,
                        error_y='error'
                    )
                elif plot_type == 'boxplot':  # Boxplot
                    fig = px.box(
                        working_df,
                        x=x_column,
                        y=y_column,
                        color=breakdown_column
                    )
                elif plot_type == 'histogram':
                    fig = px.histogram(
                        self.df,
                        x=x_column,
                        color=breakdown_column,  # Group histograms by breakdown column
                        nbins=bins,
                        histnorm='density',  # Normalize to density
                        opacity=0.5,  # Set transparency for overlapping histograms
                        barmode='overlay'  # Overlay histograms
                    )

                else:  # bar (grouped bar chart with error bars)
                    fig = px.bar(
                        grouped_df,
                        x=x_column,
                        y=y_column,
                        color=breakdown_column,
                        barmode='group',
                        error_y='error'
                    )

                # Customize plot layout
                fig.update_layout(clickmode='event+select')
                return fig

            # Trace visibility logic
            elif trigger_id == 'data-plot' and clickData is not None:
                point = clickData['points'][0]
                trace_index = point['curveNumber']

                for i, trace in enumerate(figure['data']):
                    if i == trace_index:
                        trace['visible'] = True
                    else:
                        trace['visible'] = 'legendonly'

                return figure

            # Default return if no specific trigger
            return figure

        # Existing datetime/aggregation toggle callback
        @self.app.callback(
            [Output('datetime-resolution-container', 'style'),
             Output('aggregation-container', 'style')],
            [Input('x-axis-dropdown', 'value'),
             Input('plot-type-dropdown', 'value')]
        )
        def toggle_datetime_resolution(x_column, plot_type):
            """
            Dynamically show/hide datetime resolution and aggregation dropdowns
            """
            if x_column in self.datetime_cols:
                return {'width': '48%', 'display': 'inline-block'}, {'width': '48%', 'display': 'inline-block'}
            elif x_column in self.categorical_cols or plot_type == 'line':
                return {'display': 'none'}, {'width': '48%', 'display': 'inline-block'}
            else:
                return {'display': 'none'}, {'display': 'none'}

        @self.app.callback(
            Output('histogram-bins-container', 'style'),
            [Input('plot-type-dropdown', 'value')]
        )
        def toggle_bins_input(plot_type):
            """
            Show or hide the bins input based on the selected plot type.
            """
            if plot_type == 'histogram':
                return {'width': '48%', 'display': 'inline-block'}
            return {'display': 'none'}

    def run(self, debug=True, port=8050):
        """
        Run the Dash application

        :param debug: Enable debug mode
        :param port: Port to run the application
        """
        self.app.run_server(debug=debug, port=port)


# Example usage. Replace this with loading and preprocessing of your data
def main():
    # Create a sample DataFrame for demonstration
    np.random.seed(42)
    # df = pd.DataFrame({
    #     'Date': pd.date_range(start='2023-01-01', periods=100),
    #     'Sales': np.random.randint(100, 1000, 100),
    #     'Region': np.random.choice(['North', 'South', 'East', 'West'], 100),
    #     'Product': np.random.choice(['A', 'B', 'C'], 100),
    #     'Price': np.random.uniform(10, 100, 100)
    # })
    df = pd.read_csv('StoryDBfullPublic_withTM.csv')
    df.drop(columns=['Unnamed: 0'], inplace=True)
    # replace '.' with '_' in column names
    df.columns = df.columns.str.replace('.', '_')
    # Initialize and run the dashboard
    dashboard = DataExplorerDashboard(df)
    dashboard.run()


if __name__ == '__main__':
    main()
