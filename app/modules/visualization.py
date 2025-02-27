import plotly.express as px
import plotly.graph_objects as go
from dash import html, dcc, callback_context, no_update
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, ALL
import pandas as pd
from typing import Dict, List
import logging
from datetime import datetime, timedelta
import numpy as np
import random
import io
import base64
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

logger = logging.getLogger(__name__)

# Mock data generation for SOC activities
class SOCDataGenerator:
    def __init__(self):
        self.threat_types = ['Malware', 'Phishing', 'DDoS', 'Ransomware', 'Data Exfiltration', 'SQL Injection', 
                           'Zero-Day Exploit', 'APT', 'Insider Threat', 'Supply Chain Attack']
        self.threat_sources = ['External', 'Internal', 'Unknown', 'Nation State', 'Cybercrime Group', 
                             'Hacktivist', 'Malicious Insider', 'Third Party']
        self.attack_vectors = ['Email', 'Web', 'Network', 'USB', 'Social Engineering', 'Cloud Services', 
                             'Remote Access', 'IoT Devices']
        self.locations = {
            'New York': {'lat': 40.7128, 'lon': -74.0060},
            'London': {'lat': 51.5074, 'lon': -0.1278},
            'Tokyo': {'lat': 35.6762, 'lon': 139.6503},
            'Sydney': {'lat': -33.8688, 'lon': 151.2093},
            'Moscow': {'lat': 55.7558, 'lon': 37.6173},
            'Singapore': {'lat': 1.3521, 'lon': 103.8198},
            'Dubai': {'lat': 25.2048, 'lon': 55.2708},
            'Paris': {'lat': 48.8566, 'lon': 2.3522},
            'Berlin': {'lat': 52.5200, 'lon': 13.4050},
            'Mumbai': {'lat': 19.0760, 'lon': 72.8777},
            'Toronto': {'lat': 43.6532, 'lon': -79.3832},
            'São Paulo': {'lat': -23.5505, 'lon': -46.6333},
            'Mexico City': {'lat': 19.4326, 'lon': -99.1332},
            'Cairo': {'lat': 30.0444, 'lon': 31.2357},
            'Lagos': {'lat': 6.5244, 'lon': 3.3792},
            'Istanbul': {'lat': 41.0082, 'lon': 28.9784},
            'Seoul': {'lat': 37.5665, 'lon': 126.9780},
            'Beijing': {'lat': 39.9042, 'lon': 116.4074},
            'Shanghai': {'lat': 31.2304, 'lon': 121.4737},
            'Hong Kong': {'lat': 22.3193, 'lon': 114.1694},
            'Bangkok': {'lat': 13.7563, 'lon': 100.5018},
            'Jakarta': {'lat': -6.2088, 'lon': 106.8456},
            'Manila': {'lat': 14.5995, 'lon': 120.9842},
            'Melbourne': {'lat': -37.8136, 'lon': 144.9631},
            'Auckland': {'lat': -36.8509, 'lon': 174.7645},
            'Amsterdam': {'lat': 52.3676, 'lon': 4.9041},
            'Brussels': {'lat': 50.8503, 'lon': 4.3517},
            'Vienna': {'lat': 48.2082, 'lon': 16.3738},
            'Stockholm': {'lat': 59.3293, 'lon': 18.0686},
            'Oslo': {'lat': 59.9139, 'lon': 10.7522},
            'Copenhagen': {'lat': 55.6761, 'lon': 12.5683},
            'Warsaw': {'lat': 52.2297, 'lon': 21.0122},
            'Prague': {'lat': 50.0755, 'lon': 14.4378},
            'Budapest': {'lat': 47.4979, 'lon': 19.0402},
            'Rome': {'lat': 41.9028, 'lon': 12.4964},
            'Madrid': {'lat': 40.4168, 'lon': -3.7038},
            'Lisbon': {'lat': 38.7223, 'lon': -9.1393},
            'Athens': {'lat': 37.9838, 'lon': 23.7275},
            'Bucharest': {'lat': 44.4268, 'lon': 26.1025},
            'Kiev': {'lat': 50.4501, 'lon': 30.5234},
            'Tel Aviv': {'lat': 32.0853, 'lon': 34.7818},
            'Riyadh': {'lat': 24.7136, 'lon': 46.6753},
            'Abu Dhabi': {'lat': 24.4539, 'lon': 54.3773},
            'Doha': {'lat': 25.2854, 'lon': 51.5310},
            'Kuwait City': {'lat': 29.3759, 'lon': 47.9774},
            'Tehran': {'lat': 35.6892, 'lon': 51.3890},
            'Karachi': {'lat': 24.8607, 'lon': 67.0011},
            'New Delhi': {'lat': 28.6139, 'lon': 77.2090},
            'Bangalore': {'lat': 12.9716, 'lon': 77.5946},
            'Chennai': {'lat': 13.0827, 'lon': 80.2707},
            'Colombo': {'lat': 6.9271, 'lon': 79.8612},
            'Dhaka': {'lat': 23.8103, 'lon': 90.4125},
            'Bangkok': {'lat': 13.7563, 'lon': 100.5018},
            'Ho Chi Minh City': {'lat': 10.8231, 'lon': 106.6297},
            'Kuala Lumpur': {'lat': 3.1390, 'lon': 101.6869},
            'Manila': {'lat': 14.5995, 'lon': 120.9842},
            'Taipei': {'lat': 25.0330, 'lon': 121.5654},
            'Guangzhou': {'lat': 23.1291, 'lon': 113.2644},
            'Shenzhen': {'lat': 22.5431, 'lon': 114.0579},
            'Chengdu': {'lat': 30.5728, 'lon': 104.0668},
            'Vancouver': {'lat': 49.2827, 'lon': -123.1207},
            'Montreal': {'lat': 45.5017, 'lon': -73.5673},
            'Chicago': {'lat': 41.8781, 'lon': -87.6298},
            'Los Angeles': {'lat': 34.0522, 'lon': -118.2437},
            'San Francisco': {'lat': 37.7749, 'lon': -122.4194},
            'Seattle': {'lat': 47.6062, 'lon': -122.3321},
            'Boston': {'lat': 42.3601, 'lon': -71.0589},
            'Washington DC': {'lat': 38.9072, 'lon': -77.0369},
            'Miami': {'lat': 25.7617, 'lon': -80.1918},
            'Houston': {'lat': 29.7604, 'lon': -95.3698},
            'Dallas': {'lat': 32.7767, 'lon': -96.7970},
            'Mexico City': {'lat': 19.4326, 'lon': -99.1332},
            'Monterrey': {'lat': 25.6866, 'lon': -100.3161},
            'Guadalajara': {'lat': 20.6597, 'lon': -103.3496},
            'Panama City': {'lat': 8.9824, 'lon': -79.5199},
            'Bogota': {'lat': 4.7110, 'lon': -74.0721},
            'Lima': {'lat': -12.0464, 'lon': -77.0428},
            'Santiago': {'lat': -33.4489, 'lon': -70.6693},
            'Buenos Aires': {'lat': -34.6037, 'lon': -58.3816},
            'Rio de Janeiro': {'lat': -22.9068, 'lon': -43.1729},
            'Brasilia': {'lat': -15.7975, 'lon': -47.8919},
            'Johannesburg': {'lat': -26.2041, 'lon': 28.0473},
            'Cape Town': {'lat': -33.9249, 'lon': 18.4241},
            'Nairobi': {'lat': -1.2921, 'lon': 36.8219},
            'Addis Ababa': {'lat': 9.0320, 'lon': 38.7422},
            'Casablanca': {'lat': 33.5731, 'lon': -7.5898},
            'Tunis': {'lat': 36.8065, 'lon': 10.1815},
            'Algiers': {'lat': 36.7538, 'lon': 3.0588},
            'Accra': {'lat': 5.6037, 'lon': -0.1870},
            'Dakar': {'lat': 14.7167, 'lon': -17.4677},
            'Luanda': {'lat': -8.8399, 'lon': 13.2894},
            'Kinshasa': {'lat': -4.4419, 'lon': 15.2663},
            'Dar es Salaam': {'lat': -6.7924, 'lon': 39.2083},
            'Khartoum': {'lat': 15.5007, 'lon': 32.5599},
            'Alexandria': {'lat': 31.2001, 'lon': 29.9187},
            'Beirut': {'lat': 33.8938, 'lon': 35.5018},
            'Amman': {'lat': 31.9454, 'lon': 35.9284},
            'Baghdad': {'lat': 33.3152, 'lon': 44.3661},
            'Muscat': {'lat': 23.5880, 'lon': 58.3829},
            'Baku': {'lat': 40.4093, 'lon': 49.8671},
            'Tashkent': {'lat': 41.2995, 'lon': 69.2401},
            'Almaty': {'lat': 43.2220, 'lon': 76.8512},
            'Islamabad': {'lat': 33.6007, 'lon': 73.0679},
            'Kabul': {'lat': 34.5553, 'lon': 69.2075},
            'Hanoi': {'lat': 21.0285, 'lon': 105.8542},
            'Phnom Penh': {'lat': 11.5564, 'lon': 104.9282},
            'Vientiane': {'lat': 17.9757, 'lon': 102.6331},
            'Yangon': {'lat': 16.8661, 'lon': 96.1951},
            'Perth': {'lat': -31.9505, 'lon': 115.8605},
            'Brisbane': {'lat': -27.4698, 'lon': 153.0251},
            'Adelaide': {'lat': -34.9285, 'lon': 138.6007},
            'Wellington': {'lat': -41.2866, 'lon': 174.7756}
        }
        self.status_options = ['Active', 'Investigating', 'Mitigated', 'Resolved', 'Escalated']
        self.severity_levels = ['Critical', 'High', 'Medium', 'Low']

    def generate_threat_data(self, n_threats=50):
        threats = []
        current_time = datetime.now()
        
        for i in range(n_threats):
            threat_time = current_time - timedelta(minutes=random.randint(0, 60))
            severity = np.random.choice(self.severity_levels, p=[0.1, 0.2, 0.4, 0.3])
            status = np.random.choice(self.status_options, p=[0.3, 0.3, 0.2, 0.1, 0.1])
            
            threat = {
                'id': f'THR-{i+1:04d}',
                'type': random.choice(self.threat_types),
                'source': random.choice(self.threat_sources),
                'vector': random.choice(self.attack_vectors),
                'severity': severity,
                'status': status,
                'time': threat_time.strftime('%Y-%m-%d %H:%M:%S'),
                'location': random.choice(list(self.locations.keys())),
                'ip_address': f'{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}',
                'affected_systems': random.randint(1, 10),
                'detection_method': random.choice(['SIEM', 'IDS', 'EDR', 'User Report', 'Threat Intel']),
                'confidence': random.randint(60, 100)
            }
            threats.append(threat)
        
        return pd.DataFrame(threats)

# Enhanced DashboardManager class
class DashboardManager:
    def __init__(self, app):
        self.app = app
        self.data_generator = SOCDataGenerator()
        self.current_page = 'dashboard'
        self._cached_threats = None
        self._cache_timestamp = None
        self._setup_callbacks()
        self._inject_custom_css()
        logger.info("DashboardManager initialized successfully")

    def _get_threat_data(self, force_refresh=False):
        """Get threat data from cache or generate new data if needed"""
        current_time = datetime.now()
        # Refresh cache if it's older than 30 seconds or doesn't exist
        if (force_refresh or 
            self._cached_threats is None or 
            self._cache_timestamp is None or 
            (current_time - self._cache_timestamp).total_seconds() > 30):
            self._cached_threats = self.data_generator.generate_threat_data(n_threats=50)
            self._cache_timestamp = current_time
        return self._cached_threats

    def create_main_layout(self):
        return html.Div([
            dcc.Location(id='url', refresh=False),
            self._create_navbar(),
            html.Div(id='page-content', className="px-4 py-3")
        ])

    def _create_navbar(self):
        """Create the navbar"""
        return dbc.Navbar(
            dbc.Container([
                html.A(
                    dbc.Row([
                        dbc.Col(html.I(className="fas fa-shield-alt mr-2")),
                        dbc.Col(dbc.NavbarBrand("SOC Platform", className="ml-2")),
                    ], align="center", className="g-0"),
                    href="/",
                    style={"textDecoration": "none"},
                ),
                dbc.NavbarToggler(id="navbar-toggler"),
                dbc.Collapse(
                    dbc.Nav([
                        dbc.NavItem(dbc.NavLink("Dashboard", href="/dashboard")),
                        dbc.NavItem(dbc.NavLink("Analytics", href="/analytics")),
                        dbc.NavItem(dbc.NavLink("Reports", href="/reports")),
                        dbc.NavItem(dbc.NavLink("Settings", href="/settings")),
                    ], className="ml-auto", navbar=True),
                    id="navbar-collapse",
                    navbar=True,
                ),
            ]),
            color="dark",
            dark=True,
            className="mb-4",
        )

    def _setup_callbacks(self):
        """Set up all callbacks for the dashboard"""
        # Navbar toggle callback
        @self.app.callback(
            Output("navbar-collapse", "is_open"),
            [Input("navbar-toggler", "n_clicks")],
            [State("navbar-collapse", "is_open")],
        )
        def toggle_navbar_collapse(n, is_open):
            if n:
                return not is_open
            return is_open

        # Page routing callback
        @self.app.callback(
            Output('page-content', 'children'),
            [Input('url', 'pathname')]
        )
        def display_page(pathname):
            try:
                if pathname == '/dashboard' or pathname == '/':
                    return self._create_dashboard_page()
                elif pathname == '/analytics':
                    return self._create_analytics_page()
                elif pathname == '/reports':
                    return self._create_reports_page()
                elif pathname == '/settings':
                    return self._create_settings_page()
                else:
                    return self._create_dashboard_page()
            except Exception as e:
                logger.error(f"Error in page routing: {e}")
                return html.Div("Error loading page")

        # Dashboard callbacks
        @self.app.callback(
            [Output("active-threats-count", "children"),
             Output("risk-level", "children"),
             Output("mitigated-count", "children"),
             Output("alerts-count", "children")],
            [Input("interval-component", "n_intervals")]
        )
        def update_summary_cards(n):
            try:
                df = self._get_threat_data()
                active_threats = len(df[df['status'] == 'Active'])
                risk_level = "High" if active_threats > 10 else "Medium" if active_threats > 5 else "Low"
                mitigated = len(df[df['status'] == 'Mitigated'])
                alerts = len(df[df['severity'].isin(['Critical', 'High'])])
                return str(active_threats), risk_level, str(mitigated), str(alerts)
            except Exception as e:
                logger.error(f"Error updating summary cards: {e}")
                return "N/A", "N/A", "N/A", "N/A"

        @self.app.callback(
            Output("threat-map", "figure"),
            [Input("interval-component", "n_intervals")]
        )
        def update_threat_map(n):
            try:
                df = self._get_threat_data()
                locations = self.data_generator.locations
                
                fig = go.Figure()
                
                # Add threat points
                fig.add_trace(go.Scattergeo(
                    lon=[locations[loc]['lon'] for loc in df['location']],
                    lat=[locations[loc]['lat'] for loc in df['location']],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=['red' if s == 'Critical' else 'orange' if s == 'High' else 'yellow' if s == 'Medium' else 'green' 
                               for s in df['severity']],
                        opacity=0.8,
                        symbol='circle',
                        line=dict(width=1, color='white')
                    ),
                    text=df.apply(lambda x: f"Location: {x['location']}<br>Type: {x['type']}<br>Severity: {x['severity']}", axis=1),
                    hoverinfo='text',
                    name='Threats'
                ))
                
                # Update layout with fixed constraints
                fig.update_layout(
                    geo=dict(
                        scope='world',
                        projection_type='equirectangular',
                        showland=True,
                        showcountries=True,
                        showocean=True,
                        countrywidth=0.5,
                        landcolor='rgb(43, 49, 55)',
                        oceancolor='rgb(52, 58, 64)',
                        showcoastlines=True,
                        coastlinecolor='rgba(255, 255, 255, 0.2)',
                        showframe=False,
                        bgcolor='rgba(0,0,0,0)',
                        # Set map center and zoom
                        center=dict(
                            lon=0,
                            lat=20
                        ),
                        # Control zoom level
                        projection=dict(
                            scale=1.3  # Default zoom level
                        ),
                        # Set fixed ranges for lat/lon without grid
                        lonaxis=dict(
                            range=[-180, 180],
                            showgrid=False  # Remove longitude grid
                        ),
                        lataxis=dict(
                            range=[-60, 85],
                            showgrid=False  # Remove latitude grid
                        )
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=False,
                    height=500,
                    autosize=True,
                    hovermode='closest',
                    dragmode='pan',  # Set default interaction mode to pan
                    transition=dict(
                        duration=500,
                        easing='cubic-in-out'
                    )
                )
                
                return fig
            except Exception as e:
                logger.error(f"Error updating threat map: {e}")
                # Return a basic world map if there's an error
                return go.Figure(
                    data=[],
                    layout=go.Layout(
                        geo=dict(
                            scope='world',
                            projection_type='equirectangular',
                            showland=True,
                            showcountries=True,
                            projection=dict(
                                scale=1.0  # Set default zoom level
                            ),
                            lonaxis=dict(showgrid=False),  # Remove grid in error state too
                            lataxis=dict(showgrid=False)
                        ),
                        height=500
                    )
                )

        @self.app.callback(
            Output("trend-analysis", "figure"),
            [Input("interval-component", "n_intervals")]
        )
        def update_trend_analysis(n):
            try:
                df = self._get_threat_data()
                df['time'] = pd.to_datetime(df['time'])
                hourly_counts = df.groupby([pd.Grouper(key='time', freq='h'), 'severity']).size().reset_index(name='count')
                
                fig = go.Figure()
                
                for severity in df['severity'].unique():
                    severity_data = hourly_counts[hourly_counts['severity'] == severity]
                    fig.add_trace(go.Scatter(
                        x=severity_data['time'],
                        y=severity_data['count'],
                        name=severity,
                        mode='lines+markers',
                        line=dict(width=2),
                        marker=dict(size=8)
                    ))
                
                fig.update_layout(
                    title='24-Hour Threat Activity',
                    xaxis_title='Time',
                    yaxis_title='Number of Threats',
                    showlegend=True,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    height=300
                )
                
                return fig
            except Exception as e:
                logger.error(f"Error updating trend analysis: {e}")
                return go.Figure()

        @self.app.callback(
            Output("threat-distribution", "figure"),
            [Input("threat-distribution-interval", "n_intervals")]
        )
        def update_threat_distribution(n):
            try:
                df = self._get_threat_data()
                
                # Ensure all threat types are represented
                threat_types = [
                    'Malware', 'Data Exfiltration', 'APT', 'DDoS', 
                    'Zero-Day Exploit', 'SQL Injection', 'Phishing', 
                    'Ransomware', 'Insider Threat'
                ]
                
                # Count occurrences of each threat type
                type_counts = df['type'].value_counts()
                
                # Create a dictionary with all threat types, defaulting to 0 if not present
                counts_dict = {threat: type_counts.get(threat, 0) for threat in threat_types}
                
                # Create the pie chart with fixed dimensions and softer colors
                fig = go.Figure(data=[go.Pie(
                    labels=list(counts_dict.keys()),
                    values=list(counts_dict.values()),
                    textinfo='label+percent',
                    hoverinfo='label+value+percent',
                    textposition='inside',
                    hole=0.3,
                    marker=dict(
                        colors=[
                            '#FF9999',  # Light red
                            '#66B2FF',  # Light blue
                            '#99FF99',  # Light green
                            '#FFCC99',  # Light orange
                            '#FF99FF',  # Light magenta
                            '#99FFFF',  # Light cyan
                            '#FFB366',  # Light brown
                            '#FF99CC',  # Light pink
                            '#99CC00'   # Lime green
                        ]
                    ),
                    direction='clockwise',
                    sort=False,
                    showlegend=False,
                    textfont=dict(
                        size=10,
                        color='white'
                    ),
                    insidetextorientation='horizontal',
                    pull=[0.02] * len(threat_types),
                    hovertemplate='%{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
                    # Add uirevision to maintain state between updates
                    uirevision='constant'
                )])
                
                # Fixed layout configuration
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    autosize=False,
                    width=350,
                    height=350,
                    margin=dict(
                        l=10,
                        r=10,
                        t=10,
                        b=10,
                        pad=0
                    ),
                    font=dict(
                        color='white',
                        size=10
                    ),
                    uniformtext=dict(
                        minsize=8,
                        mode='hide'
                    ),
                    # Enhanced transition settings
                    transition={
                        'duration': 750,
                        'easing': 'cubic-in-out'
                    },
                    xaxis=dict(
                        showgrid=False,
                        showticklabels=False,
                        zeroline=False,
                        domain=[0, 1],
                        fixedrange=True
                    ),
                    yaxis=dict(
                        showgrid=False,
                        showticklabels=False,
                        zeroline=False,
                        domain=[0, 1],
                        fixedrange=True
                    ),
                    # Add uirevision to maintain state between updates
                    uirevision='constant'
                )
                
                return fig
            except Exception as e:
                logger.error(f"Error updating threat distribution: {e}")
                return go.Figure()

        @self.app.callback(
            [Output("alert-details-content", "children"),
             Output("alert-details-content", "style")],
            [Input({"type": "alert-details-btn", "index": ALL}, "n_clicks")],
            [State("alert-details-content", "style")]
        )
        def toggle_alert_details(view_clicks, current_style):
            ctx = callback_context
            if not ctx.triggered or all(v is None for v in view_clicks):
                return None, {'display': 'none', 'opacity': '0'}
            
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            
            try:
                # Get the alert ID from the button that was clicked
                alert_id = eval(button_id)["index"]
                
                # Generate fresh data to simulate real-time updates
                df = self._get_threat_data()
                
                # Find the specific alert details
                alert_details = df[df['id'] == alert_id].iloc[0]
                
                # Create detailed content for the alert
                details_content = dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.H5([
                                html.I(className="fas fa-exclamation-triangle mr-2"),
                                f"Alert Details - {alert_id}"
                            ], className="mb-0 d-inline"),
                            dbc.Button(
                                "×",
                                id="close-alert-details",
                                className="close",
                                n_clicks=0,
                                style={"color": "white"}
                            ),
                        ], className="d-flex justify-content-between align-items-center")
                    ], className="bg-dark text-white"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                # Alert Information
                                html.Div([
                                    html.H6([
                                        html.I(className="fas fa-info-circle mr-2"),
                                        "Alert Information"
                                    ], className="mb-3"),
                                    dbc.ListGroup([
                                        dbc.ListGroupItem([
                                            html.Strong("Type: "),
                                            alert_details['type']
                                        ], className="bg-dark text-white"),
                                        dbc.ListGroupItem([
                                            html.Strong("Detection Time: "),
                                            alert_details['time']
                                        ], className="bg-dark text-white"),
                                        dbc.ListGroupItem([
                                            html.Strong("Location: "),
                                            alert_details['location']
                                        ], className="bg-dark text-white"),
                                        dbc.ListGroupItem([
                                            html.Strong("Status: "),
                                            dbc.Badge(
                                                alert_details['status'],
                                                color="danger" if alert_details['status'] == 'Active'
                                              else "warning" if alert_details['status'] == 'Investigating'
                                              else "info" if alert_details['status'] == 'Contained'
                                              else "success"
                                            )
                                        ], className="bg-dark text-white")
                                    ], flush=True, className="mb-4")
                                ]),
                                
                                # Technical Details
                                html.Div([
                                    html.H6([
                                        html.I(className="fas fa-code mr-2"),
                                        "Technical Details"
                                    ], className="mb-3"),
                                    dbc.ListGroup([
                                        dbc.ListGroupItem([
                                            html.Strong("Attack Vector: "),
                                            alert_details['vector']
                                        ], className="bg-dark text-white"),
                                        dbc.ListGroupItem([
                                            html.Strong("Source: "),
                                            alert_details['source']
                                        ], className="bg-dark text-white"),
                                        dbc.ListGroupItem([
                                            html.Strong("IP Address: "),
                                            alert_details['ip_address']
                                        ], className="bg-dark text-white")
                                    ], flush=True, className="mb-4")
                                ])
                            ], md=6),
                            dbc.Col([
                                # Impact Assessment
                                html.Div([
                                    html.H6([
                                        html.I(className="fas fa-chart-bar mr-2"),
                                        "Impact Assessment"
                                    ], className="mb-3"),
                                    dbc.ListGroup([
                                        dbc.ListGroupItem([
                                            html.Strong("Severity: "),
                                            dbc.Badge(
                                                alert_details['severity'],
                                                color="danger" if alert_details['severity'] == 'Critical'
                                              else "warning" if alert_details['severity'] == 'High'
                                              else "info" if alert_details['severity'] == 'Medium'
                                              else "success"
                                            )
                                        ], className="bg-dark text-white"),
                                        dbc.ListGroupItem([
                                            html.Strong("Affected Systems: "),
                                            str(alert_details['affected_systems'])
                                        ], className="bg-dark text-white"),
                                        dbc.ListGroupItem([
                                            html.Strong("Detection Method: "),
                                            alert_details['detection_method']
                                        ], className="bg-dark text-white"),
                                        dbc.ListGroupItem([
                                            html.Strong("Confidence Score: "),
                                            dbc.Progress(
                                                value=alert_details['confidence'],
                                                color="success" if alert_details['confidence'] >= 80
                                              else "warning" if alert_details['confidence'] >= 60
                                              else "danger",
                                                className="mb-0",
                                                style={"height": "0.5rem"}
                                            )
                                        ], className="bg-dark text-white")
                                    ], flush=True)
                                ])
                            ], md=6)
                        ])
                    ], className="bg-dark text-white")
                ], className="border-info mb-3")
                
                # Show the details section with animation
                return details_content, {
                    'display': 'block',
                    'opacity': '1',
                    'transition': 'opacity 0.3s ease-in-out',
                    'margin-bottom': '1rem'
                }
            except Exception as e:
                logger.error(f"Error displaying alert details: {e}")
                return None, {'display': 'none', 'opacity': '0'}

        @self.app.callback(
            [Output("alert-details-content", "children", allow_duplicate=True),
             Output("alert-details-content", "style", allow_duplicate=True)],
            [Input("close-alert-details", "n_clicks")],
            prevent_initial_call=True
        )
        def close_alert_details(n_clicks):
            if n_clicks:
                return None, {'display': 'none', 'opacity': '0'}
            return no_update, no_update

        @self.app.callback(
            Output("recent-alerts-content", "children"),
            [Input("interval-component", "n_intervals")]
        )
        def update_recent_alerts(n):
            try:
                df = self._get_threat_data()
                # Convert time column to datetime
                df['time'] = pd.to_datetime(df['time'])
                # Sort by time and take most recent 5
                recent_alerts = df.sort_values('time', ascending=False).head(5)
                
                alert_items = []
                for _, alert in recent_alerts.iterrows():
                    severity_colors = {
                        "Critical": "danger",
                        "High": "warning",
                        "Medium": "info",
                        "Low": "success"
                    }
                    alert_items.append(
                        dbc.ListGroupItem(
                            html.Div([
                                html.Div([
                                    html.H6(alert['type'], className="mb-1 text-white"),
                                    dbc.Badge(
                                        alert['severity'],
                                        color=severity_colors.get(alert['severity'], "primary"),
                                        className="ml-2"
                                    )
                                ], className="d-flex justify-content-between align-items-center"),
                                html.P(
                                    [
                                        html.I(className="fas fa-map-marker-alt mr-2"),
                                        f"Location: {alert['location']}"
                                    ],
                                    className="mb-1 small text-white"
                                ),
                                html.Small(
                                    [
                                        html.I(className="fas fa-clock mr-2"),
                                        alert['time']
                                    ],
                                    className="text-white"
                                ),
                                html.Div([
                                    html.Small(
                                        [
                                            html.I(className="fas fa-shield-alt mr-2"),
                                            f"Status: {alert['status']}"
                                        ],
                                        className="text-white"
                                    ),
                                    dbc.Button(
                                        [
                                            html.I(className="fas fa-eye mr-1"),
                                            "View Details"
                                        ],
                                        id={"type": "alert-details-btn", "index": alert['id']},
                                        color="primary",
                                        size="sm",
                                        className="ml-2"
                                    )
                                ], className="d-flex justify-content-between align-items-center mt-2")
                            ]),
                            className="bg-dark border-light-subtle mb-2"
                        )
                    )
                
                if not alert_items:
                    return html.Div("No recent alerts", className="text-white text-center p-3")
                
                return dbc.ListGroup(alert_items, className="alert-list bg-dark")
            except Exception as e:
                logger.error(f"Error updating recent alerts: {e}")
                return html.Div([
                    html.I(className="fas fa-exclamation-circle text-danger mr-2"),
                    "Error loading alerts"
                ], className="text-white text-center p-3")

        @self.app.callback(
            Output("threat-table-content", "children"),
            [Input("interval-component", "n_intervals")]
        )
        def update_threat_table(n):
            try:
                df = self._get_threat_data()
                
                table_header = [
                    html.Thead(html.Tr([
                        html.Th("ID"),
                        html.Th("Type"),
                        html.Th("Source"),
                        html.Th("Severity"),
                        html.Th("Status"),
                        html.Th("Location"),
                        html.Th("Time")
                    ]))
                ]
                
                rows = []
                for _, threat in df.iterrows():
                    severity_colors = {
                        "Critical": "danger",
                        "High": "warning",
                        "Medium": "info",
                        "Low": "success"
                    }
                    status_colors = {
                        "Active": "danger",
                        "Investigating": "warning",
                        "Contained": "info",
                        "Resolved": "success"
                    }
                    rows.append(html.Tr([
                        html.Td(threat['id']),
                        html.Td(threat['type']),
                        html.Td(threat['source']),
                        html.Td(dbc.Badge(
                            threat['severity'],
                            color=severity_colors.get(threat['severity'], "primary")
                        )),
                        html.Td(dbc.Badge(
                            threat['status'],
                            color=status_colors.get(threat['status'], "primary")
                        )),
                        html.Td(threat['location']),
                        html.Td(threat['time'])
                    ]))
                
                table_body = [html.Tbody(rows)]
                
                return dbc.Table(
                    table_header + table_body,
                    bordered=True,
                    hover=True,
                    responsive=True,
                    striped=True,
                    className="align-middle"
                )
            except Exception as e:
                logger.error(f"Error updating threat table: {e}")
                return html.Div("Error loading threat data")

        # Analytics page callbacks
        @self.app.callback(
            [Output("threat-intel-summary", "figure"),
             Output("attack-vector-chart", "figure"),
             Output("severity-chart", "figure"),
             Output("detection-methods-chart", "figure"),
             Output("threat-timeline", "figure")],
            [Input("analytics-interval", "n_intervals")]
        )
        def update_analytics_charts(n):
            try:
                df = self._get_threat_data()
                return (
                    self._create_threat_intel_visualization(df),
                    self._create_attack_vector_visualization(df),
                    self._create_severity_visualization(df),
                    self._create_detection_visualization(df),
                    self._create_timeline_visualization(df)
                )
            except Exception as e:
                logger.error(f"Error updating analytics charts: {e}")
                return tuple(go.Figure() for _ in range(5))

        # Report generation and download callback
        @self.app.callback(
            Output("download-report", "data"),
            [Input("generate-report-btn", "n_clicks")],
            [State("report-type", "value"),
             State("time-range", "value"),
             State("report-sections", "value"),
             State("export-format", "value")]
        )
        def generate_and_download_report(n_clicks, report_type, time_range, sections, export_format):
            if not n_clicks:
                return None

            try:
                # Generate threat data
                df = self._get_threat_data(force_refresh=True)  # Force refresh on interval
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename_base = f"security_report_{current_time}"

                if export_format == "csv":
                    # Generate CSV with all data
                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)
                    return dict(
                        content=csv_buffer.getvalue(),
                        filename=f"{filename_base}.csv",
                        type="text/csv",
                    )

                elif export_format == "excel":
                    # Generate Excel with multiple sheets
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        # Write main data
                        df.to_excel(writer, sheet_name='Raw Data', index=False)
                        
                        # Add selected sections
                        if "summary" in sections:
                            summary_data = self._generate_summary_data(df)
                            summary_data.to_excel(writer, sheet_name='Executive Summary', index=False)
                        
                        if "threats" in sections:
                            threat_data = self._generate_threat_analysis(df)
                            threat_data.to_excel(writer, sheet_name='Threat Analysis', index=False)
                        
                        if "incidents" in sections:
                            incident_data = self._generate_incident_details(df)
                            incident_data.to_excel(writer, sheet_name='Incident Details', index=False)
                        
                        if "actions" in sections:
                            action_data = self._generate_mitigation_actions(df)
                            action_data.to_excel(writer, sheet_name='Mitigation Actions', index=False)
                        
                        if "recommendations" in sections:
                            recommendations_data = self._generate_recommendations(df)
                            recommendations_data.to_excel(writer, sheet_name='Recommendations', index=False)

                    excel_buffer.seek(0)
                    return dict(
                        content=base64.b64encode(excel_buffer.getvalue()).decode('utf-8'),
                        filename=f"{filename_base}.xlsx",
                        type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        base64=True
                    )

                else:  # PDF
                    # Generate PDF report
                    pdf_buffer = io.BytesIO()
                    self._generate_pdf_report(df, report_type, time_range, sections, pdf_buffer)
                    pdf_buffer.seek(0)
                    return dict(
                        content=base64.b64encode(pdf_buffer.getvalue()).decode('utf-8'),
                        filename=f"{filename_base}.pdf",
                        type="application/pdf",
                        base64=True
                    )

            except Exception as e:
                logger.error(f"Error generating report: {str(e)}")
                return None

        # Add new callback for alert collapse
        @self.app.callback(
            Output("alert-collapse", "is_open"),
            [Input("alert-collapse-button", "n_clicks")],
            [State("alert-collapse", "is_open")],
        )
        def toggle_alert_collapse(n_clicks, is_open):
            if n_clicks:
                return not is_open
            return is_open

        @self.app.callback(
            [Output("threat-details-collapse", "is_open"),
             Output("threat-details-content", "children")],
            [Input({"type": "view-details-btn", "index": ALL}, "n_clicks"),
             Input("close-details", "n_clicks")],
            [State("threat-details-collapse", "is_open")]
        )
        def toggle_threat_details(view_clicks, close_clicks, is_open):
            ctx = callback_context
            if not ctx.triggered:
                return False, None
            
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            
            if button_id == "close-details":
                return False, None
            
            try:
                # Get the threat ID from the button that was clicked
                threat_id = eval(button_id)["index"]
                
                # Use cached data instead of generating new data
                df = self._get_threat_data()
                
                # Find the specific threat details
                threat_details = df[df['id'] == threat_id].iloc[0]
                
                # Create detailed content based on threat type
                threat_type_details = {
                    'Malware': {
                        'icon': 'fas fa-virus',
                        'sections': [
                            ('Malware Details', ['Malware Family', 'Infection Vector', 'Known Variants']),
                            ('System Impact', ['Affected Files', 'System Changes', 'Network Activity'])
                        ]
                    },
                    'Phishing': {
                        'icon': 'fas fa-envelope',
                        'sections': [
                            ('Email Details', ['Sender', 'Subject', 'Attachments']),
                            ('Target Information', ['Recipients', 'Delivery Time', 'Email Content'])
                        ]
                    },
                    'DDoS': {
                        'icon': 'fas fa-network-wired',
                        'sections': [
                            ('Attack Details', ['Attack Type', 'Traffic Volume', 'Target Services']),
                            ('Network Impact', ['Bandwidth Usage', 'Service Status', 'Mitigation Status'])
                        ]
                    },
                    'Data Exfiltration': {
                        'icon': 'fas fa-database',
                        'sections': [
                            ('Data Details', ['Data Types', 'Volume', 'Destination']),
                            ('Access Information', ['Source IP', 'Protocols Used', 'Time Period'])
                        ]
                    }
                }
                
                type_info = threat_type_details.get(threat_details['type'], {
                    'icon': 'fas fa-exclamation-triangle',
                    'sections': [
                        ('Threat Details', ['Type', 'Source', 'Target']),
                        ('Impact Details', ['Severity', 'Scope', 'Status'])
                    ]
                })
                
                details_content = dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.H5([
                                html.I(className=f"{type_info['icon']} mr-2"),
                                f"Threat Details - {threat_id}"
                            ], className="mb-0 d-inline"),
                            dbc.Button(
                                "×",
                                id="close-details",
                                className="close",
                                n_clicks=0,
                                style={"color": "white"}
                            ),
                        ], className="d-flex justify-content-between align-items-center")
                    ], className="bg-dark text-white"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                # Primary Information
                                html.Div([
                                    html.H6([
                                        html.I(className="fas fa-info-circle mr-2"),
                                        "Primary Information"
                                    ], className="mb-3"),
                                    dbc.ListGroup([
                                        dbc.ListGroupItem([
                                            html.Strong("Type: "),
                                            threat_details['type']
                                        ], className="bg-dark text-white"),
                                        dbc.ListGroupItem([
                                            html.Strong("Detection Time: "),
                                            threat_details['time']
                                        ], className="bg-dark text-white"),
                                        dbc.ListGroupItem([
                                            html.Strong("Location: "),
                                            threat_details['location']
                                        ], className="bg-dark text-white"),
                                        dbc.ListGroupItem([
                                            html.Strong("Status: "),
                                            dbc.Badge(
                                                threat_details['status'],
                                                color="danger" if threat_details['status'] == 'Active'
                                              else "warning" if threat_details['status'] == 'Investigating'
                                              else "info" if threat_details['status'] == 'Contained'
                                              else "success"
                                            )
                                        ], className="bg-dark text-white")
                                    ], flush=True, className="mb-4")
                                ]),
                                
                                # Type-Specific Details
                                html.Div([
                                    html.H6([
                                        html.I(className=f"{type_info['icon']} mr-2"),
                                        f"{threat_details['type']} Specific Details"
                                    ], className="mb-3"),
                                    dbc.ListGroup([
                                        dbc.ListGroupItem([
                                            html.Strong("Attack Vector: "),
                                            threat_details['vector']
                                        ], className="bg-dark text-white"),
                                        dbc.ListGroupItem([
                                            html.Strong("Source: "),
                                            threat_details['source']
                                        ], className="bg-dark text-white"),
                                        dbc.ListGroupItem([
                                            html.Strong("Affected Systems: "),
                                            str(threat_details['affected_systems'])
                                        ], className="bg-dark text-white")
                                    ], flush=True, className="mb-4")
                                ])
                            ], md=6),
                            dbc.Col([
                                # Technical Details
                                html.Div([
                                    html.H6([
                                        html.I(className="fas fa-code mr-2"),
                                        "Technical Details"
                                    ], className="mb-3"),
                                    dbc.ListGroup([
                                        dbc.ListGroupItem([
                                            html.Strong("IP Address: "),
                                            threat_details['ip_address']
                                        ], className="bg-dark text-white"),
                                        dbc.ListGroupItem([
                                            html.Strong("Detection Method: "),
                                            threat_details['detection_method']
                                        ], className="bg-dark text-white")
                                    ], flush=True, className="mb-4")
                                ]),
                                
                                # Risk Assessment
                                html.Div([
                                    html.H6([
                                        html.I(className="fas fa-chart-bar mr-2"),
                                        "Risk Assessment"
                                    ], className="mb-3"),
                                    dbc.ListGroup([
                                        dbc.ListGroupItem([
                                            html.Strong("Severity: "),
                                            dbc.Badge(
                                                threat_details['severity'],
                                                color="danger" if threat_details['severity'] == 'Critical'
                                              else "warning" if threat_details['severity'] == 'High'
                                              else "info" if threat_details['severity'] == 'Medium'
                                              else "success"
                                            )
                                        ], className="bg-dark text-white"),
                                        dbc.ListGroupItem([
                                            html.Strong("Confidence Score: "),
                                            dbc.Progress(
                                                value=threat_details['confidence'],
                                                color="success" if threat_details['confidence'] >= 80
                                              else "warning" if threat_details['confidence'] >= 60
                                              else "danger",
                                                className="mb-0",
                                                style={"height": "0.5rem"}
                                            )
                                        ], className="bg-dark text-white")
                                    ], flush=True)
                                ])
                            ], md=6)
                        ])
                    ], className="bg-dark text-white")
                ], className="border-info")
                
                return True, details_content
            except Exception as e:
                logger.error(f"Error displaying threat details: {e}")
                return False, None

        @self.app.callback(
            [Output("active-threats-list", "children"),
             Output("high-risk-count", "children")],
            [Input("active-threats-interval", "n_intervals")]
        )
        def update_active_threats(n):
            try:
                df = self._get_threat_data(force_refresh=True)  # Force refresh on interval
                # Filter for high-risk and critical threats
                high_risk_threats = df[
                    (df['severity'].isin(['Critical', 'High'])) & 
                    (df['status'] == 'Active')
                ].sort_values('time', ascending=False).head(5)

                threat_items = []
                for _, threat in high_risk_threats.iterrows():
                    threat_items.append(self._create_active_threat_item(
                        threat['id'],
                        threat['type'],
                        threat['severity'],
                        threat['status'],
                        threat['location'],
                        threat['time'],
                        f"Alert: {threat['type']} detected from {threat['source']}"
                    ))

                high_risk_count = f"{len(high_risk_threats)} High Risk"
                
                return dbc.ListGroup(threat_items, flush=True, className="text-white"), high_risk_count
            except Exception as e:
                logger.error(f"Error updating active threats: {e}")
                return html.Div("Error loading threats"), "Error"

    def _get_recent_reports(self):
        """Get list of recent reports with actual timestamps"""
        current_time = datetime.now()
        recent_reports = [
            self._create_report_item(
                "Security Report [PDF]",
                "Daily overview of security incidents and threats",
                (current_time - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"),
                "System"
            ),
            self._create_report_item(
                "Security Report [CSV]",
                "Detailed analysis of detected threats and patterns",
                (current_time - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
                "John Doe"
            ),
            self._create_report_item(
                "Security Report [PDF]",
                "Summary of incident response activities",
                (current_time - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
                "Jane Smith"
            )
        ]
        return dbc.ListGroup(recent_reports, className="bg-dark")

    def _generate_summary_data(self, df):
        """Generate summary data for reports"""
        summary_data = {
            'Metric': [
                'Total Threats',
                'Critical Threats',
                'High Threats',
                'Medium Threats',
                'Low Threats',
                'Active Threats',
                'Mitigated Threats',
                'Detection Rate',
                'Average Response Time'
            ],
            'Value': [
                len(df),
                len(df[df['severity'] == 'Critical']),
                len(df[df['severity'] == 'High']),
                len(df[df['severity'] == 'Medium']),
                len(df[df['severity'] == 'Low']),
                len(df[df['status'] == 'Active']),
                len(df[df['status'] == 'Mitigated']),
                f"{(len(df[df['detection_method'] == 'SIEM']) / len(df) * 100):.1f}%",
                "15 minutes"
            ]
        }
        return pd.DataFrame(summary_data)

    def _generate_threat_analysis(self, df):
        """Generate threat analysis data for reports"""
        # Group by multiple columns for detailed analysis
        threat_analysis = df.groupby(['type', 'severity', 'status', 'source']).agg({
            'id': 'count',
            'confidence': 'mean'
        }).reset_index()
        
        # Rename columns for clarity
        threat_analysis.columns = ['Threat Type', 'Severity', 'Status', 'Source', 'Count', 'Confidence Score']
        
        # Format confidence score
        threat_analysis['Confidence Score'] = threat_analysis['Confidence Score'].round(2)
        
        return threat_analysis

    def _generate_pdf_report(self, df, report_type, time_range, sections, buffer):
        """Generate PDF report with selected sections"""
        try:
            # Initialize document
            doc = SimpleDocTemplate(
                buffer,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            # Styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                textColor=colors.HexColor('#2c3e50')
            )
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=18,
                spaceAfter=12,
                textColor=colors.HexColor('#34495e')
            )
            body_style = ParagraphStyle(
                'CustomBody',
                parent=styles['Normal'],
                fontSize=12,
                spaceAfter=12
            )
            
            # Build story
            story = []
            
            # Title
            story.append(Paragraph(f"Security Report - {report_type}", title_style))
            story.append(Spacer(1, 20))
            
            # Report metadata
            story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
            story.append(Paragraph(f"Time Range: {time_range}", body_style))
            story.append(Spacer(1, 20))
            
            # Add sections based on selection
            if "summary" in sections:
                story.append(Paragraph("Executive Summary", heading_style))
                summary_data = self._generate_summary_data(df)
                # Convert DataFrame to list of lists for table
                table_data = [summary_data.columns.tolist()] + summary_data.values.tolist()
                summary_table = Table(table_data, colWidths=[4*inch, 2*inch])
                summary_table.setStyle(self._get_table_style())
                story.append(summary_table)
                story.append(Spacer(1, 20))
            
            if "threats" in sections:
                story.append(Paragraph("Threat Analysis", heading_style))
                threat_data = self._generate_threat_analysis(df)
                # Convert DataFrame to list of lists for table
                table_data = [threat_data.columns.tolist()] + threat_data.values.tolist()
                threat_table = Table(table_data, colWidths=[2*inch, 1*inch, 1*inch, 1.5*inch, 1*inch, 1*inch])
                threat_table.setStyle(self._get_table_style())
                story.append(threat_table)
                story.append(Spacer(1, 20))
            
            if "incidents" in sections:
                story.append(Paragraph("Incident Details", heading_style))
                incident_data = self._generate_incident_details(df)
                # Convert DataFrame to list of lists for table
                table_data = [incident_data.columns.tolist()] + incident_data.values.tolist()
                incident_table = Table(table_data, colWidths=[1*inch, 1.5*inch, 1*inch, 1*inch, 1.5*inch, 1.5*inch])
                incident_table.setStyle(self._get_table_style())
                story.append(incident_table)
                story.append(Spacer(1, 20))
            
            if "actions" in sections:
                story.append(Paragraph("Mitigation Actions", heading_style))
                action_data = self._generate_mitigation_actions(df)
                # Convert DataFrame to list of lists for table
                table_data = [action_data.columns.tolist()] + action_data.values.tolist()
                action_table = Table(table_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 2*inch])
                action_table.setStyle(self._get_table_style())
                story.append(action_table)
                story.append(Spacer(1, 20))
            
            if "recommendations" in sections:
                story.append(Paragraph("Recommendations", heading_style))
                recommendations_data = self._generate_recommendations(df)
                # Convert DataFrame to list of lists for table
                table_data = [recommendations_data.columns.tolist()] + recommendations_data.values.tolist()
                recommendations_table = Table(table_data, colWidths=[1.5*inch, 3*inch, 1*inch, 1.5*inch])
                recommendations_table.setStyle(self._get_table_style())
                story.append(recommendations_table)
            
            # Build PDF
            doc.build(story)
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            raise

    def _get_table_style(self):
        """Get consistent table style for reports"""
        return TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ])

    def _create_header(self, title):
        """Create page header with title"""
        return html.Div([
            html.H1(title, className="header-title fade-in"),
            html.P("Real-time monitoring and analysis", className="text-center mb-4 text-muted fade-in")
        ], className="text-center mb-5")

    def _inject_custom_css(self):
        """Inject custom CSS into the app"""
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>SOC Platform</title>
                {%favicon%}
                {%css%}
                <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
                <style>
                    :root {
                        --primary-color: #2c3e50;
                        --secondary-color: #34495e;
                        --accent-color: #3498db;
                        --background-color: #1a1a1a;
                        --card-bg: rgba(40, 44, 52, 0.95);
                        --text-color: #ecf0f1;
                        --text-muted: #95a5a6;
                        --success-color: #27ae60;
                        --warning-color: #f39c12;
                        --danger-color: #c0392b;
                        --info-color: #2980b9;
                    }
                    
                    body {
                        font-family: 'Inter', sans-serif;
                        background: var(--background-color);
                        color: var(--text-color);
                    }
                    
                    .card {
                        background: var(--card-bg);
                        border: 1px solid rgba(255, 255, 255, 0.1);
                        border-radius: 10px;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    }
                    
                    .card-header {
                        background: rgba(0, 0, 0, 0.2);
                        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                        padding: 1rem;
                        color: var(--text-color);
                    }
                    
                    .card-body {
                        background: var(--card-bg);
                        color: var(--text-color);
                    }
                    
                    .table {
                        color: var(--text-color) !important;
                    }
                    
                    .table-dark {
                        background-color: var(--card-bg) !important;
                    }
                    
                    .table-dark td,
                    .table-dark th {
                        color: var(--text-color) !important;
                        border-color: rgba(255, 255, 255, 0.1) !important;
                    }
                    
                    .list-group-item {
                        background: var(--card-bg) !important;
                        border-color: rgba(255, 255, 255, 0.1) !important;
                        color: var(--text-color) !important;
                    }
                    
                    .form-control {
                        background-color: rgba(255, 255, 255, 0.1) !important;
                        border-color: rgba(255, 255, 255, 0.1) !important;
                        color: var(--text-color) !important;
                    }
                    
                    .form-control:focus {
                        background-color: rgba(255, 255, 255, 0.15) !important;
                        border-color: var(--accent-color) !important;
                        color: var(--text-color) !important;
                    }
                    
                    .custom-control-label {
                        color: var(--text-color) !important;
                    }
                    
                    .custom-switch .custom-control-label::before {
                        background-color: rgba(255, 255, 255, 0.1) !important;
                    }
                    
                    .text-light {
                        color: var(--text-color) !important;
                    }
                    
                    .bg-dark {
                        background-color: var(--card-bg) !important;
                    }
                    
                    /* Additional styles for better visibility */
                    .checklist-option {
                        color: var(--text-color) !important;
                    }
                    
                    .custom-control-label::before,
                    .custom-control-label::after {
                        background-color: var(--accent-color) !important;
                    }
                    
                    .custom-switch .custom-control-input:checked ~ .custom-control-label::before {
                        background-color: var(--accent-color) !important;
                    }
                    
                    /* Ensure text contrast */
                    h1, h2, h3, h4, h5, h6, p, span, div {
                        color: var(--text-color);
                    }
                    
                    /* Active Threats Styles */
                    .active-threats-container {
                        max-height: 400px;
                        overflow-y: auto;
                    }
                    
                    .border-left-thick {
                        border-left: 4px solid;
                        transition: all 0.3s ease;
                    }
                    
                    .border-left-thick:hover {
                        transform: translateX(5px);
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    }
                    
                    .list-group-item.border-left-thick {
                        border-left-color: var(--danger);
                    }
                    
                    .alert-collapse-button:hover {
                        background-color: rgba(255,255,255,0.1) !important;
                    }
                    
                    .active-threats-container::-webkit-scrollbar {
                        width: 8px;
                    }
                    
                    .active-threats-container::-webkit-scrollbar-track {
                        background: rgba(0,0,0,0.1);
                        border-radius: 4px;
                    }
                    
                    .active-threats-container::-webkit-scrollbar-thumb {
                        background: rgba(255,255,255,0.2);
                        border-radius: 4px;
                    }
                    
                    .active-threats-container::-webkit-scrollbar-thumb:hover {
                        background: rgba(255,255,255,0.3);
                    }
                    
                    /* Alert Details Styles */
                    .alert-details-container {
                        background: var(--card-bg);
                        border-radius: 8px;
                        overflow: hidden;
                        transition: all 0.3s ease-in-out;
                        opacity: 0;
                    }
                    
                    .alert-details-container.show {
                        opacity: 1;
                        margin-bottom: 1rem;
                    }
                    
                    .alert-details-container .card {
                        border: 1px solid rgba(255, 255, 255, 0.1);
                        margin: 0;
                    }
                    
                    .alert-details-container .card-header {
                        background: rgba(0, 0, 0, 0.2);
                        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                    }
                    
                    .alert-details-container .list-group-item {
                        background: var(--card-bg);
                        border-color: rgba(255, 255, 255, 0.1);
                        padding: 0.75rem 1rem;
                    }
                    
                    .alert-details-container .progress {
                        background-color: rgba(0, 0, 0, 0.2);
                    }
                    
                    .alert-details-container .close:hover {
                        color: #fff;
                        opacity: 0.75;
                    }
                    
                    /* Collapse Animation */
                    .collapse-enter {
                        max-height: 0;
                        overflow: hidden;
                        transition: max-height 0.3s ease-in-out;
                    }
                    
                    .collapse-enter.collapse-enter-active {
                        max-height: 1000px;
                    }
                    
                    .collapse-exit {
                        max-height: 1000px;
                        overflow: hidden;
                        transition: max-height 0.3s ease-in-out;
                    }
                    
                    .collapse-exit.collapse-exit-active {
                        max-height: 0;
                    }
                </style>
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''

    def _create_dashboard_page(self):
        """Create the main dashboard page"""
        try:
            return dbc.Container([
                self._create_header("Security Operations Dashboard"),
                self._create_alert_section(),
                self._create_summary_cards(),
                dbc.Row([
                    dbc.Col([
                        self._create_threat_map(),
                        self._create_trend_analysis(),
                    ], md=8),
                    dbc.Col([
                        self._create_threat_distribution(),
                        self._create_recent_alerts(),
                    ], md=4),
                ], className="mb-4"),
                dbc.Row([
                    dbc.Col(self._create_threat_table(), md=12),
                ]),
                dcc.Interval(
                    id='interval-component',
                    interval=5*1000,
                    n_intervals=0
                ),
            ], fluid=True)
        except Exception as e:
            logger.error(f"Error creating dashboard page: {e}")
            return html.Div("Error loading dashboard")

    def _create_analytics_page(self):
        """Create the analytics page"""
        try:
            return dbc.Container([
                self._create_header("Threat Analytics"),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Threat Intelligence Overview"),
                            dbc.CardBody([
                                dcc.Graph(
                                    id='threat-intel-summary',
                                    config={'displayModeBar': False}
                                )
                            ])
                        ], className="mb-4"),
                        dbc.Card([
                            dbc.CardHeader("Attack Vector Analysis"),
                            dbc.CardBody([
                                dcc.Graph(
                                    id='attack-vector-chart',
                                    config={'displayModeBar': False}
                                )
                            ])
                        ], className="mb-4")
                    ], md=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Severity Distribution"),
                            dbc.CardBody([
                                dcc.Graph(
                                    id='severity-chart',
                                    config={'displayModeBar': False}
                                )
                            ])
                        ], className="mb-4"),
                        dbc.Card([
                            dbc.CardHeader("Detection Methods"),
                            dbc.CardBody([
                                dcc.Graph(
                                    id='detection-methods-chart',
                                    config={'displayModeBar': False}
                                )
                            ])
                        ], className="mb-4")
                    ], md=6)
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Threat Timeline"),
                            dbc.CardBody([
                                dcc.Graph(
                                    id='threat-timeline',
                                    config={'displayModeBar': False}
                                )
                            ])
                        ], className="mb-4")
                    ], md=12)
                ]),
                dcc.Interval(
                    id='analytics-interval',
                    interval=10*1000,
                    n_intervals=0
                )
            ], fluid=True)
        except Exception as e:
            logger.error(f"Error creating analytics page: {e}")
            return html.Div("Error loading analytics")

    def _create_reports_page(self):
        """Create the reports page with download functionality"""
        try:
            return dbc.Container([
                self._create_header("Security Reports"),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Generate Report", className="bg-dark text-light"),
                            dbc.CardBody([
                                dbc.Form([
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("Report Type", className="text-light"),
                                            dbc.Select(
                                                id="report-type",
                                                options=[
                                                    {"label": "Executive Summary", "value": "executive"},
                                                    {"label": "Incident Report", "value": "incident"},
                                                    {"label": "Threat Analysis", "value": "threat"},
                                                    {"label": "Compliance Report", "value": "compliance"},
                                                    {"label": "Performance Metrics", "value": "performance"}
                                                ],
                                                value="executive",
                                                className="mb-3"
                                            )
                                        ], md=6),
                                        dbc.Col([
                                            dbc.Label("Time Range", className="text-light"),
                                            dbc.Select(
                                                id="time-range",
                                                options=[
                                                    {"label": "Last 24 Hours", "value": "24h"},
                                                    {"label": "Last 7 Days", "value": "7d"},
                                                    {"label": "Last 30 Days", "value": "30d"}
                                                ],
                                                value="24h",
                                                className="mb-3"
                                            )
                                        ], md=6)
                                    ]),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("Include Sections", className="text-light"),
                                            dbc.Checklist(
                                                id="report-sections",
                                                options=[
                                                    {"label": "Executive Summary", "value": "summary"},
                                                    {"label": "Threat Analysis", "value": "threats"},
                                                    {"label": "Incident Details", "value": "incidents"},
                                                    {"label": "Mitigation Actions", "value": "actions"},
                                                    {"label": "Recommendations", "value": "recommendations"}
                                                ],
                                                value=["summary", "threats", "incidents"],
                                                className="text-light mb-3"
                                            )
                                        ])
                                    ]),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("Export Format", className="text-light"),
                                            dbc.RadioItems(
                                                id="export-format",
                                                options=[
                                                    {"label": "PDF Report", "value": "pdf"},
                                                    {"label": "CSV Data", "value": "csv"},
                                                    {"label": "Excel Workbook", "value": "excel"}
                                                ],
                                                value="pdf",
                                                inline=True,
                                                className="text-light mb-3"
                                            )
                                        ])
                                    ]),
                                    dbc.Button(
                                        [html.I(className="fas fa-file-export mr-2"), "Generate & Download"],
                                        id="generate-report-btn",
                                        color="primary",
                                        className="mt-2"
                                    ),
                                    dcc.Download(id="download-report")
                                ])
                            ], className="bg-dark")
                        ], className="mb-4 border-0")
                    ], md=12)
                ])
            ], fluid=True)
        except Exception as e:
            logger.error(f"Error creating reports page: {e}")
            return html.Div("Error loading reports page", className="text-light")

    def _create_report_item(self, title, description, time, author):
        """Create a report list item"""
        return dbc.ListGroupItem([
            dbc.Row([
                dbc.Col([
                    html.H5(title, className="mb-1"),
                    html.P(description, className="mb-1"),
                    html.Small([
                        html.I(className="fas fa-clock mr-1"),
                        f"Generated {time} by {author}"
                    ], className="text-muted")
                ], md=9),
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button([
                            html.I(className="fas fa-share-alt mr-1"),
                            "Share"
                        ], color="info", size="sm", className="mr-2"),
                        dbc.Button([
                            html.I(className="fas fa-trash-alt mr-1"),
                            "Delete"
                        ], color="danger", size="sm")
                    ], className="float-right")
                ], md=3, className="d-flex align-items-center")
            ])
        ])

    def _create_threat_hunting_page(self):
        """Create threat hunting page with enhanced functionality"""
        return dbc.Container([
            self._create_header("Threat Hunting"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Hunt Query Builder"),
                        dbc.CardBody([
                            dbc.Form([
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Data Source"),
                                        dbc.Select(
                                            id="data-source",
                                            options=[
                                                {"label": "Network Logs", "value": "network"},
                                                {"label": "Endpoint Logs", "value": "endpoint"},
                                                {"label": "SIEM Events", "value": "siem"},
                                                {"label": "Threat Intel", "value": "intel"}
                                            ]
                                        )
                                    ], md=4),
                                    dbc.Col([
                                        dbc.Label("Time Range"),
                                        dbc.Select(
                                            id="hunt-time-range",
                                            options=[
                                                {"label": "Last Hour", "value": "1h"},
                                                {"label": "Last 24 Hours", "value": "24h"},
                                                {"label": "Last 7 Days", "value": "7d"},
                                                {"label": "Custom", "value": "custom"}
                                            ]
                                        )
                                    ], md=4),
                                    dbc.Col([
                                        dbc.Label("Hunt Type"),
                                        dbc.Select(
                                            id="hunt-type",
                                            options=[
                                                {"label": "IOC Search", "value": "ioc"},
                                                {"label": "Behavior Analysis", "value": "behavior"},
                                                {"label": "Pattern Match", "value": "pattern"},
                                                {"label": "Anomaly Detection", "value": "anomaly"}
                                            ]
                                        )
                                    ], md=4)
                                ], className="mb-3"),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Search Query"),
                                        dbc.Textarea(
                                            id="hunt-query",
                                            placeholder="Enter your hunt query...",
                                            style={"height": "100px"}
                                        )
                                    ])
                                ], className="mb-3"),
                                dbc.Button([
                                    html.I(className="fas fa-search mr-2"),
                                    "Start Hunt"
                                ], color="primary")
                            ])
                        ])
                    ], className="mb-4")
                ], md=12)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Active Hunts"),
                        dbc.CardBody([
                            dbc.ListGroup([
                                self._create_hunt_item(
                                    "Ransomware Behavior Detection",
                                    "Hunting for signs of ransomware activity",
                                    "In Progress",
                                    "75"
                                ),
                                self._create_hunt_item(
                                    "C2 Communication Pattern",
                                    "Detecting potential command & control traffic",
                                    "Completed",
                                    "100"
                                ),
                                self._create_hunt_item(
                                    "Data Exfiltration",
                                    "Identifying unusual data transfers",
                                    "In Progress",
                                    "45"
                                )
                            ])
                        ])
                    ])
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Hunt Results"),
                        dbc.CardBody([
                            dbc.Tabs([
                                dbc.Tab([
                                    html.Div([
                                        html.H5("Findings", className="mb-3"),
                                        dbc.ListGroup([
                                            self._create_finding_item(
                                                "Suspicious PowerShell Activity",
                                                "Multiple encoded commands detected",
                                                "High"
                                            ),
                                            self._create_finding_item(
                                                "Unusual Network Connection",
                                                "Connection to known malicious IP",
                                                "Critical"
                                            ),
                                            self._create_finding_item(
                                                "Registry Modification",
                                                "Persistence mechanism detected",
                                                "Medium"
                                            )
                                        ])
                                    ], className="mt-3")
                                ], label="Findings"),
                                dbc.Tab([
                                    dcc.Graph(
                                        figure=self._create_hunt_metrics(),
                                        config={'displayModeBar': False}
                                    )
                                ], label="Metrics")
                            ])
                        ])
                    ])
                ], md=6)
            ])
        ], fluid=True)

    def _create_hunt_item(self, title, description, status, progress):
        """Create a hunt list item"""
        status_colors = {
            "In Progress": "warning",
            "Completed": "success",
            "Failed": "danger"
        }
        return dbc.ListGroupItem([
            html.Div([
                html.H5(title, className="mb-1"),
                html.P(description, className="mb-1"),
                dbc.Progress(
                    value=int(progress),
                    color=status_colors.get(status, "primary"),
                    className="mb-2",
                    style={"height": "4px"}
                ),
                html.Div([
                    dbc.Badge(status, color=status_colors.get(status, "primary"), className="mr-2"),
                    html.Small(f"Progress: {progress}%", className="text-muted")
                ])
            ])
        ])

    def _create_finding_item(self, title, description, severity):
        """Create a finding list item"""
        severity_colors = {
            "Critical": "danger",
            "High": "warning",
            "Medium": "info",
            "Low": "success"
        }
        return dbc.ListGroupItem([
            html.Div([
                html.Div([
                    html.H6(title, className="mb-1"),
                    html.Small(description, className="text-muted")
                ]),
                dbc.Badge(severity, color=severity_colors.get(severity, "primary"))
            ], className="d-flex justify-content-between align-items-center")
        ])

    def _create_hunt_metrics(self):
        """Create hunt metrics visualization"""
        # Sample data for hunt metrics
        categories = ['IOCs Found', 'False Positives', 'True Positives', 'Pending Review']
        values = [45, 15, 25, 5]
        
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=values,
                marker_color=['#3498db', '#e74c3c', '#2ecc71', '#f1c40f']
            )
        ])
        
        fig.update_layout(
            title='Hunt Metrics Overview',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=False
        )
        
        return fig

    def _create_incidents_page(self):
        """Create incidents page with enhanced functionality"""
        return dbc.Container([
            self._create_header("Incident Management"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Active Incidents"),
                        dbc.CardBody([
                            dbc.Tabs([
                                dbc.Tab([
                                    self._create_incident_table()
                                ], label="All Incidents"),
                                dbc.Tab([
                                    self._create_incident_timeline()
                                ], label="Timeline"),
                                dbc.Tab([
                                    self._create_incident_metrics()
                                ], label="Metrics")
                            ])
                        ])
                    ], className="mb-4")
                ], md=12)
            ]),
            dbc.Row([
                dbc.Col([
                    self._create_incident_details()
                ], md=8),
                dbc.Col([
                    self._create_incident_actions()
                ], md=4)
            ])
        ], fluid=True)

    def _create_incident_table(self):
        """Create incident table with filtering and sorting"""
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Input(
                        id="incident-search",
                        placeholder="Search incidents...",
                        type="text",
                        className="mb-3"
                    )
                ], md=6),
                dbc.Col([
                    dbc.Select(
                        id="incident-filter",
                        options=[
                            {"label": "All Incidents", "value": "all"},
                            {"label": "Critical", "value": "critical"},
                            {"label": "High", "value": "high"},
                            {"label": "Medium", "value": "medium"},
                            {"label": "Low", "value": "low"}
                        ],
                        value="all",
                        className="mb-3"
                    )
                ], md=3),
                dbc.Col([
                    dbc.Select(
                        id="incident-sort",
                        options=[
                            {"label": "Newest First", "value": "newest"},
                            {"label": "Oldest First", "value": "oldest"},
                            {"label": "Highest Severity", "value": "severity"},
                            {"label": "Status", "value": "status"}
                        ],
                        value="newest",
                        className="mb-3"
                    )
                ], md=3)
            ]),
            dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("ID"),
                        html.Th("Title"),
                        html.Th("Severity"),
                        html.Th("Status"),
                        html.Th("Assigned To"),
                        html.Th("Last Updated"),
                        html.Th("Actions")
                    ])
                ]),
                html.Tbody([
                    self._create_incident_row(
                        "INC-001",
                        "Ransomware Attack",
                        "Critical",
                        "Active",
                        "John Doe",
                        "10 min ago"
                    ),
                    self._create_incident_row(
                        "INC-002",
                        "Phishing Campaign",
                        "High",
                        "Investigating",
                        "Jane Smith",
                        "30 min ago"
                    ),
                    self._create_incident_row(
                        "INC-003",
                        "Data Exfiltration",
                        "High",
                        "Contained",
                        "Mike Johnson",
                        "1 hour ago"
                    )
                ])
            ], bordered=True, hover=True, responsive=True, striped=True)
        ])

    def _create_incident_row(self, id, title, severity, status, assigned, updated):
        """Create an incident table row"""
        severity_colors = {
            "Critical": "danger",
            "High": "warning",
            "Medium": "info",
            "Low": "success"
        }
        status_colors = {
            "Active": "danger",
            "Investigating": "warning",
            "Contained": "info",
            "Resolved": "success"
        }
        return html.Tr([
            html.Td(id),
            html.Td(title),
            html.Td(dbc.Badge(severity, color=severity_colors.get(severity, "primary"))),
            html.Td(dbc.Badge(status, color=status_colors.get(status, "primary"))),
            html.Td(assigned),
            html.Td(updated),
            html.Td(
                dbc.ButtonGroup([
                    dbc.Button("View", color="primary", size="sm", className="mr-1"),
                    dbc.Button("Update", color="warning", size="sm", className="mr-1"),
                    dbc.Button("Close", color="danger", size="sm")
                ])
            )
        ])

    def _create_incident_metrics(self):
        """Create incident metrics visualization"""
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        figure=self._create_incident_severity_chart(),
                        config={'displayModeBar': False}
                    )
                ], md=6),
                dbc.Col([
                    dcc.Graph(
                        figure=self._create_incident_status_chart(),
                        config={'displayModeBar': False}
                    )
                ], md=6)
            ], className="mb-4"),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        figure=self._create_incident_trend_chart(),
                        config={'displayModeBar': False}
                    )
                ], md=12)
            ])
        ])

    def _create_incident_severity_chart(self):
        """Create incident severity distribution chart"""
        labels = ['Critical', 'High', 'Medium', 'Low']
        values = [5, 12, 25, 8]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.6,
            marker=dict(colors=['#e74c3c', '#f39c12', '#3498db', '#2ecc71'])
        )])
        
        fig.update_layout(
            title='Incident Severity Distribution',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=True
        )
        
        return fig

    def _create_incident_status_chart(self):
        """Create incident status distribution chart"""
        x = ['Active', 'Investigating', 'Contained', 'Resolved']
        y = [8, 15, 10, 20]
        
        fig = go.Figure(data=[go.Bar(
            x=x,
            y=y,
            marker_color=['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
        )])
        
        fig.update_layout(
            title='Incident Status Distribution',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=False
        )
        
        return fig

    def _create_incident_trend_chart(self):
        """Create incident trend chart"""
        dates = pd.date_range(start='2024-01-01', periods=14)
        incidents = np.random.randint(5, 20, size=14)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=incidents,
            mode='lines+markers',
            line=dict(color='#3498db', width=2),
            fill='tozeroy'
        ))
        
        fig.update_layout(
            title='Incident Trend (Last 14 Days)',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=False,
            xaxis_title='Date',
            yaxis_title='Number of Incidents'
        )
        
        return fig

    def _create_incident_actions(self):
        """Create incident actions panel"""
        return dbc.Card([
            dbc.CardHeader("Response Actions"),
            dbc.CardBody([
                dbc.ButtonGroup([
                    dbc.Button([
                        html.I(className="fas fa-shield-alt mr-2"),
                        "Isolate System"
                    ], color="warning", className="mb-2 w-100"),
                    dbc.Button([
                        html.I(className="fas fa-ban mr-2"),
                        "Block IOCs"
                    ], color="danger", className="mb-2 w-100"),
                    dbc.Button([
                        html.I(className="fas fa-envelope mr-2"),
                        "Send Alert"
                    ], color="info", className="mb-2 w-100")
                ], vertical=True),
                html.Hr(),
                html.H6("Playbook Steps", className="mb-3"),
                dbc.ListGroup([
                    self._create_playbook_step(
                        "1. Initial Response",
                        "Isolate affected systems",
                        "Completed"
                    ),
                    self._create_playbook_step(
                        "2. Investigation",
                        "Collect and analyze evidence",
                        "In Progress"
                    ),
                    self._create_playbook_step(
                        "3. Containment",
                        "Implement containment measures",
                        "Pending"
                    ),
                    self._create_playbook_step(
                        "4. Eradication",
                        "Remove threat from systems",
                        "Pending"
                    )
                ], flush=True)
            ])
        ])

    def _create_playbook_step(self, title, description, status):
        """Create a playbook step item"""
        status_colors = {
            "Completed": "success",
            "In Progress": "warning",
            "Pending": "secondary",
            "Failed": "danger"
        }
        return dbc.ListGroupItem([
            html.Div([
                html.Div([
                    html.H6(title, className="mb-1"),
                    html.Small(description, className="text-muted")
                ]),
                dbc.Badge(status, color=status_colors.get(status, "primary"))
            ], className="d-flex justify-content-between align-items-center")
        ])

    def _create_training_page(self):
        """Create training page with enhanced functionality"""
        return dbc.Container([
            self._create_header("SOC Training"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Training Modules"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    self._create_training_card(
                                        "Incident Response",
                                        "Learn effective incident response procedures",
                                        ["IR Process", "Evidence Collection", "Documentation"],
                                        "4 hours",
                                        "Beginner"
                                    )
                                ], md=6),
                                dbc.Col([
                                    self._create_training_card(
                                        "Threat Hunting",
                                        "Advanced threat hunting techniques",
                                        ["IOC Analysis", "YARA Rules", "Memory Forensics"],
                                        "8 hours",
                                        "Advanced"
                                    )
                                ], md=6)
                            ], className="mb-4"),
                            dbc.Row([
                                dbc.Col([
                                    self._create_training_card(
                                        "Malware Analysis",
                                        "Learn malware analysis fundamentals",
                                        ["Static Analysis", "Dynamic Analysis", "Reverse Engineering"],
                                        "12 hours",
                                        "Intermediate"
                                    )
                                ], md=6),
                                dbc.Col([
                                    self._create_training_card(
                                        "Digital Forensics",
                                        "Master digital forensics techniques",
                                        ["Disk Forensics", "Network Forensics", "Memory Analysis"],
                                        "16 hours",
                                        "Advanced"
                                    )
                                ], md=6)
                            ])
                        ])
                    ], className="mb-4")
                ], md=12)
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Training Progress"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.H5("Overall Progress", className="mb-3"),
                                    dbc.Progress([
                                        dbc.Progress(value=40, color="success", bar=True),
                                        dbc.Progress(value=35, color="warning", bar=True),
                                        dbc.Progress(value=25, color="danger", bar=True)
                                    ], className="mb-3", style={"height": "2rem"}),
                                    html.Div([
                                        dbc.Badge("Completed (40%)", color="success", className="mr-2"),
                                        dbc.Badge("In Progress (35%)", color="warning", className="mr-2"),
                                        dbc.Badge("Not Started (25%)", color="danger")
                                    ], className="mb-4")
                                ])
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dcc.Graph(
                                        figure=self._create_training_metrics(),
                                        config={'displayModeBar': False}
                                    )
                                ])
                            ])
                        ])
                    ])
                ], md=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Upcoming Sessions"),
                        dbc.CardBody([
                            dbc.ListGroup([
                                self._create_session_item(
                                    "Incident Response Workshop",
                                    "Jan 15, 2024 10:00 AM",
                                    "John Doe",
                                    15
                                ),
                                self._create_session_item(
                                    "Threat Hunting Lab",
                                    "Jan 16, 2024 2:00 PM",
                                    "Jane Smith",
                                    8
                                ),
                                self._create_session_item(
                                    "Malware Analysis Demo",
                                    "Jan 17, 2024 11:00 AM",
                                    "Mike Johnson",
                                    12
                                )
                            ])
                        ])
                    ])
                ], md=4)
            ])
        ], fluid=True)

    def _create_session_item(self, title, time, instructor, spots):
        """Create a training session item"""
        return dbc.ListGroupItem([
            html.Div([
                html.H6(title, className="mb-1"),
                html.Small([
                    html.I(className="fas fa-clock mr-1"),
                    time
                ], className="text-muted d-block"),
                html.Small([
                    html.I(className="fas fa-user mr-1"),
                    f"Instructor: {instructor}"
                ], className="text-muted d-block"),
                html.Small([
                    html.I(className="fas fa-users mr-1"),
                    f"Available Spots: {spots}"
                ], className="text-muted d-block")
            ], className="mb-2"),
            dbc.Button("Register", color="primary", size="sm")
        ])

    def _create_training_metrics(self):
        """Create training metrics visualization"""
        categories = ['Completed', 'In Progress', 'Scheduled', 'Available']
        values = [12, 8, 5, 15]
        
        fig = go.Figure(data=[go.Bar(
            x=categories,
            y=values,
            marker_color=['#2ecc71', '#f1c40f', '#3498db', '#95a5a6']
        )])
        
        fig.update_layout(
            title='Training Module Status',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=False
        )
        
        return fig

    def _create_settings_page(self):
        """Create settings page with enhanced functionality"""
        try:
            notification_settings = dbc.Card([
                dbc.CardHeader("Notification Settings", className="bg-dark text-light"),
                dbc.CardBody([
                    html.H5("Alert Notifications", className="mb-3 text-light"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Critical Alerts", className="text-light"),
                            dbc.Checklist(
                                id="critical-alerts",
                                options=[
                                    {"label": "Email", "value": "email"},
                                    {"label": "SMS", "value": "sms"},
                                    {"label": "Slack", "value": "slack"},
                                    {"label": "Teams", "value": "teams"}
                                ],
                                value=["email", "sms", "slack"],
                                switch=True,
                                className="text-light"
                            ),
                        ], width=12, className="mb-3"),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("High Priority Alerts", className="text-light"),
                            dbc.Checklist(
                                id="high-alerts",
                                options=[
                                    {"label": "Email", "value": "email"},
                                    {"label": "SMS", "value": "sms"},
                                    {"label": "Slack", "value": "slack"},
                                    {"label": "Teams", "value": "teams"}
                                ],
                                value=["email", "slack"],
                                switch=True,
                                className="text-light"
                            ),
                        ], width=12, className="mb-3"),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Medium Priority Alerts", className="text-light"),
                            dbc.Checklist(
                                id="medium-alerts",
                                options=[
                                    {"label": "Email", "value": "email"},
                                    {"label": "Slack", "value": "slack"},
                                    {"label": "Teams", "value": "teams"}
                                ],
                                value=["email"],
                                switch=True,
                                className="text-light"
                            ),
                        ], width=12, className="mb-3"),
                    ]),
                ], className="bg-dark")
            ], className="mb-4 border-0")

            return dbc.Container([
                self._create_header("System Settings"),
                dbc.Row([
                    dbc.Col(notification_settings, md=6),
                    dbc.Col(self._create_integration_settings(), md=6)
                ]),
                dbc.Row([
                    dbc.Col(self._create_user_management(), md=12)
                ])
            ], fluid=True)
        except Exception as e:
            logger.error(f"Error creating settings page: {str(e)}")
            return html.Div([
                html.H4("Error Loading Settings Page", className="text-light text-center mb-3"),
                html.P(f"An error occurred: {str(e)}", className="text-light text-center"),
                dbc.Button("Refresh Page", color="primary", className="d-block mx-auto", id="refresh-settings")
            ], className="p-5")

    def _create_integration_settings(self):
        """Create integration settings card"""
        return dbc.Card([
            dbc.CardHeader("Integration Settings", className="bg-dark text-light"),
            dbc.CardBody([
                html.H5("External Integrations", className="mb-3 text-light"),
                dbc.ListGroup([
                    self._create_integration_item(
                        "SIEM Integration",
                        "Connected to Splunk Enterprise",
                        "Connected",
                        "success"
                    ),
                    self._create_integration_item(
                        "Threat Intel Platform",
                        "Connected to AlienVault OTX",
                        "Connected",
                        "success"
                    ),
                    self._create_integration_item(
                        "Ticketing System",
                        "Connected to ServiceNow",
                        "Connected",
                        "success"
                    ),
                    self._create_integration_item(
                        "Email Security",
                        "Connection Failed",
                        "Error",
                        "danger"
                    )
                ], flush=True, className="bg-dark")
            ], className="bg-dark")
        ], className="mb-4 border-0")

    def _create_user_management(self):
        """Create user management card"""
        return dbc.Card([
            dbc.CardHeader("User Management", className="bg-dark text-light"),
            dbc.CardBody([
                html.H5("Active Users", className="mb-3 text-light"),
                dbc.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("User", className="text-light"),
                            html.Th("Role", className="text-light"),
                            html.Th("Status", className="text-light"),
                            html.Th("Last Active", className="text-light"),
                            html.Th("Actions", className="text-light")
                        ], className="bg-dark")
                    ]),
                    html.Tbody([
                        self._create_user_row(
                            "John Doe",
                            "Admin",
                            "Active",
                            "5 min ago"
                        ),
                        self._create_user_row(
                            "Jane Smith",
                            "Analyst",
                            "Active",
                            "10 min ago"
                        ),
                        self._create_user_row(
                            "Mike Johnson",
                            "Analyst",
                            "Inactive",
                            "1 hour ago"
                        )
                    ])
                ], bordered=True, hover=True, responsive=True, striped=True, 
                className="text-light bg-dark table-dark")
            ], className="bg-dark")
        ], className="border-0")

    def _create_integration_item(self, title, description, status, status_color):
        """Create an integration list item"""
        return dbc.ListGroupItem([
            html.Div([
                html.Div([
                    html.H6(title, className="mb-1 text-light"),
                    html.Small(description, className="text-muted")
                ]),
                html.Div([
                    dbc.Badge(status, color=status_color, className="mr-2"),
                    dbc.Button("Configure", color="primary", size="sm")
                ])
            ], className="d-flex justify-content-between align-items-center")
        ], className="bg-dark")

    def _create_user_row(self, name, role, status, last_active):
        """Create a user table row"""
        status_colors = {
            "Active": "success",
            "Inactive": "secondary"
        }
        return html.Tr([
            html.Td([
                html.I(className="fas fa-user-circle mr-2"),
                name
            ], className="text-light"),
            html.Td(role, className="text-light"),
            html.Td(dbc.Badge(status, color=status_colors[status])),
            html.Td(last_active, className="text-light"),
            html.Td([
                dbc.ButtonGroup([
                    dbc.Button("Edit", color="primary", size="sm", className="mr-1"),
                    dbc.Button("Disable", color="danger", size="sm")
                ])
            ])
        ])

    def _create_alert_section(self):
        """Create alert section with real-time alerts and expandable details"""
        return dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader(
                        dbc.Button(
                            [
                                html.I(className="fas fa-exclamation-triangle mr-2"),
                                "Active Threats Detected ",
                                html.Span(id="high-risk-count", className="ml-2 badge bg-danger"),
                                html.I(className="fas fa-chevron-down ml-2")
                            ],
                            id="alert-collapse-button",
                            color="warning",
                            className="w-100 text-left d-flex align-items-center justify-content-between text-white",
                        )
                    ),
                    dbc.Collapse(
                        dbc.CardBody([
                            # Threat Details Container
                            dbc.Collapse(
                                dbc.Card([
                                    dbc.CardHeader([
                                        html.H5("Threat Details", className="mb-0 d-inline"),
                                        dbc.Button(
                                            "×",
                                            id="close-details",
                                            className="close float-right",
                                            n_clicks=0,
                                        ),
                                    ], className="bg-dark text-white"),
                                    dbc.CardBody(id="threat-details-content", className="bg-dark text-white")
                                ], className="mb-3 border-info"),
                                id="threat-details-collapse",
                                is_open=False,
                            ),
                            # Active Threats List
                            html.Div(
                                id="active-threats-list",
                                className="active-threats-container"
                            ),
                            # Add interval component for updating active threats
                            dcc.Interval(
                                id='active-threats-interval',
                                interval=30*1000,  # 30 seconds
                                n_intervals=0
                            )
                        ], className="text-white"),
                        id="alert-collapse",
                        is_open=False,
                    )
                ], className="mb-4 border-danger"),
            )
        ])

    def _create_active_threat_item(self, threat_id, threat_type, severity, status, location, time, description):
        """Create an individual threat item for the expandable alert section"""
        severity_colors = {
            "Critical": "danger",
            "High": "warning",
            "Medium": "info",
            "Low": "success"
        }
        status_colors = {
            "Active": "danger",
            "Investigating": "warning",
            "Contained": "info",
            "Resolved": "success"
        }
        
        return dbc.ListGroupItem([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H5([
                            html.I(className="fas fa-bug mr-2"),
                            threat_id,
                            dbc.Badge(
                                severity,
                                color=severity_colors.get(severity, "primary"),
                                className="ml-2"
                            )
                        ], className="mb-1 text-white"),
                        html.H6(threat_type, className="mb-2 text-white"),
                        html.P(description, className="mb-2 text-white"),
                        html.Div([
                            dbc.Badge(
                                [html.I(className="fas fa-map-marker-alt mr-1"), location],
                                color="dark",
                                className="mr-2 text-white"
                            ),
                            dbc.Badge(
                                [html.I(className="fas fa-clock mr-1"), time],
                                color="dark",
                                className="mr-2 text-white"
                            ),
                            dbc.Badge(
                                status,
                                color=status_colors.get(status, "primary"),
                                className="mr-2"
                            )
                        ])
                    ])
                ], md=9),
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button([
                            html.I(className="fas fa-eye mr-1"),
                            "Details"
                        ], id={"type": "view-details-btn", "index": threat_id},
                        color="dark", size="sm")
                    ], className="float-right")
                ], md=3, className="d-flex align-items-center justify-content-end")
            ])
        ], className="border-left-thick", style={"color": "white"})

    def _create_summary_cards(self):
        """Create summary cards with key metrics"""
        return dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-exclamation-circle fa-2x text-danger mb-2"),
                            html.H2(id="active-threats-count", className="stat-value"),
                            html.P("Active Threats", className="stat-label")
                        ], className="text-center")
                    ])
                ], className="mb-4"),
                md=3
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-chart-line fa-2x text-warning mb-2"),
                            html.H2(id="risk-level", className="stat-value"),
                            html.P("Risk Level", className="stat-label")
                        ], className="text-center")
                    ])
                ], className="mb-4"),
                md=3
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-shield-alt fa-2x text-success mb-2"),
                            html.H2(id="mitigated-count", className="stat-value"),
                            html.P("Threats Mitigated", className="stat-label")
                        ], className="text-center")
                    ])
                ], className="mb-4"),
                md=3
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-bell fa-2x text-info mb-2"),
                            html.H2(id="alerts-count", className="stat-value"),
                            html.P("Active Alerts", className="stat-label")
                        ], className="text-center")
                    ])
                ], className="mb-4"),
                md=3
            )
        ])

    def _create_threat_map(self):
        """Create threat map visualization"""
        return dbc.Card([
            dbc.CardHeader([
                html.H3("Global Threat Distribution", className="mb-0"),
                html.Small("Real-time geographic threat monitoring", className="text-muted")
            ]),
            dbc.CardBody([
                dcc.Graph(
                    id="threat-map",
                    config={
                        'displayModeBar': True,
                        'scrollZoom': True,  # Enable scroll zoom
                        'modeBarButtonsToRemove': [
                            'lasso2d', 
                            'select2d', 
                            'autoScale2d',
                            'resetScale2d',
                            'zoom2d',
                            'pan2d',
                            'zoomIn2d',
                            'zoomOut2d'
                        ],  # Remove zoom and pan buttons
                        'doubleClick': False  # Disable double click zoom
                    },
                    style={'height': '500px'}
                )
            ])
        ], className="mb-4")

    def _create_trend_analysis(self):
        """Create trend analysis visualization"""
        return dbc.Card([
            dbc.CardHeader([
                html.H3("Threat Trend Analysis", className="mb-0"),
                html.Small("24-hour threat activity pattern", className="text-muted")
            ]),
            dbc.CardBody([
                dcc.Graph(
                    id="trend-analysis",
                    config={'displayModeBar': False}
                )
            ])
        ])

    def _create_threat_distribution(self):
        """Create threat distribution visualization"""
        return dbc.Card([
            dbc.CardHeader([
                html.H3("Threat Categories", className="mb-0 text-white"),
                html.Small("Distribution by type", className="text-white")
            ], className="bg-dark"),
            dbc.CardBody([
                html.Div([
                    dcc.Graph(
                        id="threat-distribution",
                        config={'displayModeBar': False},
                        style={
                            'height': '350px',
                            'width': '350px',  # Fixed width
                            'margin': '0 auto',
                            'display': 'block',
                            'position': 'relative'
                        }
                    )
                ], style={
                    'height': '350px',
                    'width': '100%',
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'center',
                    'overflow': 'hidden'
                }),
                dcc.Interval(
                    id='threat-distribution-interval',
                    interval=5*1000,
                    n_intervals=0
                )
            ], className="bg-dark", style={
                'height': '350px',
                'padding': '10px',
                'display': 'flex',
                'flexDirection': 'column',
                'justifyContent': 'center'
            })
        ], className="bg-dark border-0")

    def _create_recent_alerts(self):
        """Create recent alerts panel with collapsible details"""
        return dbc.Card([
            dbc.CardHeader([
                html.Div([
                    html.H3("Recent Alerts", className="mb-0 text-white"),
                    dbc.Badge("Live", color="danger", className="ml-2 pulse")
                ], className="d-flex align-items-center"),
                html.Small("Last 5 security alerts", className="text-white")
            ], className="bg-dark"),
            dbc.CardBody([
                # Alert Details Section (Hidden by default)
                html.Div(
                    id="alert-details-content",
                    className="alert-details-container",
                    style={'display': 'none', 'opacity': '0', 'transition': 'opacity 0.3s ease-in-out'}
                ),
                # Recent Alerts List
                html.Div(
                    id="recent-alerts-content",
                    className="bg-dark text-white"
                ),
                dcc.Loading(
                    id="alerts-loading",
                    type="default",
                    children=[
                        dcc.Interval(
                            id='interval-component',
                            interval=5*1000,
                            n_intervals=0
                        )
                    ]
                )
            ], className="bg-dark p-2")
        ], className="bg-dark border-0 h-100")

    def _create_threat_table(self):
        """Create threat table"""
        return dbc.Card([
            dbc.CardHeader([
                html.H3("Detailed Threat Analysis", className="mb-0"),
                html.Small("Comprehensive threat information", className="text-muted")
            ]),
            dbc.CardBody([
                html.Div(id="threat-table-content")
            ])
        ])

    def _create_threat_intel_visualization(self, df):
        """Create threat intelligence visualization"""
        try:
            # Create bubble chart for threat types and sources
            source_type_counts = df.groupby(['source', 'type']).size().reset_index(name='count')
            
            fig = px.scatter(source_type_counts,
                           x='source',
                           y='type',
                           size='count',
                           color='count',
                           color_continuous_scale='Viridis',
                           title='Threat Intelligence Overview')
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                showlegend=False
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error creating threat intel visualization: {e}")
            return go.Figure()

    def _create_attack_vector_visualization(self, df):
        """Create attack vector visualization"""
        try:
            # Create sunburst chart for attack vectors
            vector_severity = df.groupby(['vector', 'severity']).size().reset_index(name='count')
            
            fig = px.sunburst(vector_severity,
                            path=['vector', 'severity'],
                            values='count',
                            color='count',
                            color_continuous_scale='Viridis',
                            title='Attack Vector Distribution')
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error creating attack vector visualization: {e}")
            return go.Figure()

    def _create_severity_visualization(self, df):
        """Create severity visualization"""
        try:
            # Create stacked bar chart for severity and status
            severity_status = df.groupby(['severity', 'status']).size().reset_index(name='count')
            
            fig = px.bar(severity_status,
                        x='severity',
                        y='count',
                        color='status',
                        title='Threat Severity Distribution',
                        barmode='stack')
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error creating severity visualization: {e}")
            return go.Figure()

    def _create_detection_visualization(self, df):
        """Create detection methods visualization"""
        try:
            # Create scatter plot for detection methods
            detection_metrics = df.groupby('detection_method').agg({
                'confidence': 'mean',
                'id': 'count'
            }).reset_index()
            
            detection_metrics.columns = ['method', 'avg_confidence', 'count']
            
            fig = px.scatter(detection_metrics,
                           x='avg_confidence',
                           y='count',
                           text='method',
                           size='count',
                           color='avg_confidence',
                           title='Detection Methods Effectiveness')
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error creating detection visualization: {e}")
            return go.Figure()

    def _create_timeline_visualization(self, df):
        """Create timeline visualization"""
        try:
            # Create line chart for threats over time
            df['time'] = pd.to_datetime(df['time'])
            timeline_data = df.groupby(['time', 'severity']).size().reset_index(name='count')
            
            fig = px.line(timeline_data,
                         x='time',
                         y='count',
                         color='severity',
                         title='Threat Activity Timeline')
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                showlegend=False,
                xaxis_title='Date',
                yaxis_title='Number of Threats'
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error creating timeline visualization: {e}")
            return go.Figure()

    def _create_training_card(self, title, description, topics, duration, level):
        """Create a training module card"""
        level_colors = {
            "Beginner": "success",
            "Intermediate": "warning",
            "Advanced": "danger"
        }
        return dbc.Card([
            dbc.CardBody([
                html.H5(title, className="mb-2"),
                dbc.Badge(level, color=level_colors.get(level, "primary"), className="mb-2"),
                html.P(description, className="mb-3"),
                html.H6("Topics Covered:", className="mb-2"),
                html.Ul([html.Li(topic) for topic in topics], className="mb-3"),
                html.Div([
                    html.Small(f"Duration: {duration}", className="text-muted"),
                    dbc.Button("Start Training", color="primary", size="sm")
                ], className="d-flex justify-content-between align-items-center")
            ])
        ], className="mb-4")

    def _create_incident_timeline(self):
        """Create incident timeline visualization"""
        events = [
            {
                "title": "Initial Detection",
                "time": "2024-01-14 15:30:00",
                "description": "Suspicious activity detected in network logs",
                "type": "detection"
            },
            {
                "title": "Investigation Started",
                "time": "2024-01-14 15:35:00",
                "description": "SOC team began initial investigation",
                "type": "investigation"
            },
            {
                "title": "Containment Measures",
                "time": "2024-01-14 15:40:00",
                "description": "Implemented network isolation for affected systems",
                "type": "containment"
            },
            {
                "title": "Root Cause Analysis",
                "time": "2024-01-14 15:45:00",
                "description": "Identified compromised credentials as entry point",
                "type": "analysis"
            }
        ]
        
        return dbc.Card([
            dbc.CardHeader("Incident Timeline"),
            dbc.CardBody([
                dbc.ListGroup([
                    self._create_timeline_event(**event) for event in events
                ], flush=True)
            ])
        ], className="mb-4")

    def _create_timeline_event(self, title, time, description, type):
        """Create a timeline event item"""
        type_icons = {
            "detection": "fas fa-exclamation-circle text-danger",
            "investigation": "fas fa-search text-info",
            "containment": "fas fa-shield-alt text-warning",
            "analysis": "fas fa-microscope text-success"
        }
        return dbc.ListGroupItem([
            html.Div([
                html.Div([
                    html.I(className=type_icons.get(type, "fas fa-circle"), style={"width": "20px"}),
                    html.Div([
                        html.H6(title, className="mb-1"),
                        html.Small(time, className="text-muted d-block"),
                        html.P(description, className="mb-0 mt-2")
                    ], className="ml-3")
                ], className="d-flex")
            ], className="p-2")
        ])

    def _create_incident_details(self):
        """Create incident details interface"""
        return dbc.Card([
            dbc.CardHeader("Incident Details"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H5("Current Incident", className="mb-3"),
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.H4("INC-2024-001", className="mb-2"),
                                    dbc.Badge("Critical", color="danger", className="mb-3"),
                                    html.P("Ransomware Attack Attempt", className="mb-2"),
                                    html.Small("Detected: 2024-01-14 15:30:00", className="text-muted d-block"),
                                    html.Small("Status: Active Investigation", className="text-muted d-block"),
                                    html.Hr(),
                                    html.H6("Affected Systems", className="mb-2"),
                                    dbc.ListGroup([
                                        dbc.ListGroupItem("WS-001 (192.168.1.100)"),
                                        dbc.ListGroupItem("WS-002 (192.168.1.101)"),
                                        dbc.ListGroupItem("SRV-001 (192.168.1.10)")
                                    ], flush=True)
                                ])
                            ])
                        ])
                    ], md=6),
                    dbc.Col([
                        html.H5("Impact Assessment", className="mb-3"),
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    self._create_impact_metric("Systems Affected", "3", "warning"),
                                    self._create_impact_metric("Data at Risk", "250GB", "danger"),
                                    self._create_impact_metric("Services Impacted", "2", "warning"),
                                    self._create_impact_metric("Users Affected", "15", "info")
                                ])
                            ])
                        ]),
                        html.H5("Risk Assessment", className="mb-3 mt-4"),
                        dbc.Progress(value=75, color="danger", className="mb-2",
                                   label="Business Impact"),
                        dbc.Progress(value=60, color="warning", className="mb-2",
                                   label="Data Loss Risk"),
                        dbc.Progress(value=40, color="info", className="mb-2",
                                   label="Spread Risk"),
                        html.Hr(),
                        dbc.Row([
                            dbc.Col([
                                html.H5("Technical Details", className="mb-3"),
                                dbc.Card([
                                    dbc.CardBody([
                                        html.Pre("""
Incident Type: Ransomware Attack Attempt
Initial Vector: Phishing Email
Malware Family: LockBit
C2 Servers: 
- 185.234.xxx.xxx
- 192.168.xxx.xxx
Encrypted Extensions: .locked, .encrypted
IOCs:
- Hash: d41d8cd98f00b204e9800998ecf8427e
- Domain: evil-domain.com
- IP: 192.168.1.100
                                        """, className="mb-0")
                                    ])
                                ])
                            ])
                        ])
                    ], md=6)
                ])
            ])
        ], className="mb-4")

    def _create_impact_metric(self, label, value, color):
        """Create an impact metric display"""
        return html.Div([
            html.H6(label, className="mb-1"),
            html.H4([
                value,
                dbc.Badge("", className=f"ml-2 bg-{color}", 
                         style={"width": "10px", "height": "10px", "border-radius": "50%"})
            ], className="mb-3")
        ])

    def _generate_incident_details(self, df):
        """Generate incident details data for reports"""
        # Filter for active and high-severity threats
        incident_data = df[
            (df['status'] == 'Active') & 
            (df['severity'].isin(['Critical', 'High']))
        ].copy()
        
        # Add risk score based on severity
        severity_scores = {'Critical': 100, 'High': 80, 'Medium': 60, 'Low': 40}
        incident_data['Risk Score'] = incident_data['severity'].map(severity_scores)
        
        # Select and rename columns
        incident_data = incident_data[[
            'id', 'type', 'severity', 'location', 'detection_method', 'time'
        ]].rename(columns={
            'id': 'Incident ID',
            'type': 'Threat Type',
            'severity': 'Severity',
            'location': 'Location',
            'detection_method': 'Detection Method',
            'time': 'Timestamp'
        })
        
        return incident_data

    def _generate_mitigation_actions(self, df):
        """Generate mitigation actions data for reports"""
        # Create a DataFrame with predefined mitigation actions
        actions = {
            'Action': [
                'System Isolation',
                'Network Segmentation',
                'Credential Reset',
                'Patch Application',
                'Malware Removal'
            ],
            'Status': [
                'Completed',
                'In Progress',
                'Pending',
                'Completed',
                'In Progress'
            ],
            'Priority': [
                'High',
                'Critical',
                'Medium',
                'High',
                'Critical'
            ],
            'Assigned To': [
                'SOC Team',
                'Network Team',
                'Identity Team',
                'IT Team',
                'Security Team'
            ]
        }
        return pd.DataFrame(actions)

    def _generate_recommendations(self, df):
        """Generate recommendations data for reports"""
        # Create a DataFrame with recommendations based on threat analysis
        recommendations = {
            'Category': [
                'Access Control',
                'Network Security',
                'Endpoint Protection',
                'Security Awareness',
                'Incident Response'
            ],
            'Recommendation': [
                'Implement multi-factor authentication across all systems',
                'Enhance network segmentation and monitoring',
                'Update endpoint detection and response solutions',
                'Conduct regular security awareness training',
                'Review and update incident response procedures'
            ],
            'Priority': [
                'High',
                'Critical',
                'High',
                'Medium',
                'High'
            ],
            'Timeline': [
                '1 week',
                'Immediate',
                '2 weeks',
                '1 month',
                '2 weeks'
            ]
        }
        return pd.DataFrame(recommendations)

    def _create_incident_timeline(self):
        """Create incident timeline visualization"""
        events = [
            {
                "title": "Initial Detection",
                "time": "2024-01-14 15:30:00",
                "description": "Suspicious activity detected in network logs",
                "type": "detection"
            },
            {
                "title": "Investigation Started",
                "time": "2024-01-14 15:35:00",
                "description": "SOC team began initial investigation",
                "type": "investigation"
            },
            {
                "title": "Containment Measures",
                "time": "2024-01-14 15:40:00",
                "description": "Implemented network isolation for affected systems",
                "type": "containment"
            },
            {
                "title": "Root Cause Analysis",
                "time": "2024-01-14 15:45:00",
                "description": "Identified compromised credentials as entry point",
                "type": "analysis"
            }
        ]
        
        return dbc.Card([
            dbc.CardHeader("Incident Timeline"),
            dbc.CardBody([
                dbc.ListGroup([
                    self._create_timeline_event(**event) for event in events
                ], flush=True)
            ])
        ], className="mb-4")

    def _create_timeline_event(self, title, time, description, type):
        """Create a timeline event item"""
        type_icons = {
            "detection": "fas fa-exclamation-circle text-danger",
            "investigation": "fas fa-search text-info",
            "containment": "fas fa-shield-alt text-warning",
            "analysis": "fas fa-microscope text-success"
        }
        return dbc.ListGroupItem([
            html.Div([
                html.Div([
                    html.I(className=type_icons.get(type, "fas fa-circle"), style={"width": "20px"}),
                    html.Div([
                        html.H6(title, className="mb-1"),
                        html.Small(time, className="text-muted d-block"),
                        html.P(description, className="mb-0 mt-2")
                    ], className="ml-3")
                ], className="d-flex")
            ], className="p-2")
        ])

    def _create_incident_details(self):
        """Create incident details interface"""
        return dbc.Card([
            dbc.CardHeader("Incident Details"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H5("Current Incident", className="mb-3"),
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.H4("INC-2024-001", className="mb-2"),
                                    dbc.Badge("Critical", color="danger", className="mb-3"),
                                    html.P("Ransomware Attack Attempt", className="mb-2"),
                                    html.Small("Detected: 2024-01-14 15:30:00", className="text-muted d-block"),
                                    html.Small("Status: Active Investigation", className="text-muted d-block"),
                                    html.Hr(),
                                    html.H6("Affected Systems", className="mb-2"),
                                    dbc.ListGroup([
                                        dbc.ListGroupItem("WS-001 (192.168.1.100)"),
                                        dbc.ListGroupItem("WS-002 (192.168.1.101)"),
                                        dbc.ListGroupItem("SRV-001 (192.168.1.10)")
                                    ], flush=True)
                                ])
                            ])
                        ])
                    ], md=6),
                    dbc.Col([
                        html.H5("Impact Assessment", className="mb-3"),
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    self._create_impact_metric("Systems Affected", "3", "warning"),
                                    self._create_impact_metric("Data at Risk", "250GB", "danger"),
                                    self._create_impact_metric("Services Impacted", "2", "warning"),
                                    self._create_impact_metric("Users Affected", "15", "info")
                                ])
                            ])
                        ]),
                        html.H5("Risk Assessment", className="mb-3 mt-4"),
                        dbc.Progress(value=75, color="danger", className="mb-2",
                                   label="Business Impact"),
                        dbc.Progress(value=60, color="warning", className="mb-2",
                                   label="Data Loss Risk"),
                        dbc.Progress(value=40, color="info", className="mb-2",
                                   label="Spread Risk"),
                        html.Hr(),
                        dbc.Row([
                            dbc.Col([
                                html.H5("Technical Details", className="mb-3"),
                                dbc.Card([
                                    dbc.CardBody([
                                        html.Pre("""
Incident Type: Ransomware Attack Attempt
Initial Vector: Phishing Email
Malware Family: LockBit
C2 Servers: 
- 185.234.xxx.xxx
- 192.168.xxx.xxx
Encrypted Extensions: .locked, .encrypted
IOCs:
- Hash: d41d8cd98f00b204e9800998ecf8427e
- Domain: evil-domain.com
- IP: 192.168.1.100
                                        """, className="mb-0")
                                    ])
                                ])
                            ])
                        ])
                    ], md=6)
                ])
            ])
        ], className="mb-4")

    def _create_impact_metric(self, label, value, color):
        """Create an impact metric display"""
        return html.Div([
            html.H6(label, className="mb-1"),
            html.H4([
                value,
                dbc.Badge("", className=f"ml-2 bg-{color}", 
                         style={"width": "10px", "height": "10px", "border-radius": "50%"})
            ], className="mb-3")
        ]) 