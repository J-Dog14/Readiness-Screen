import sqlite3
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.colors as plotly_colors

# Load data
db_path = r'D:/Readiness Screen 3/Readiness_Screen_Data_v2.db'
conn = sqlite3.connect(db_path)

df_cmj = pd.read_sql("""
    SELECT Name, Creation_Date,
           Jump_Height AS Jump_Height_CMJ,
           PP_FORCEPLATE AS PP_FORCEPLATE_CMJ,
           Force_at_PP AS Force_at_PP_CMJ,
           Vel_at_PP AS Vel_at_PP_CMJ
    FROM CMJ
""", conn)

df_ppu = pd.read_sql("""
    SELECT Name, Creation_Date,
           Jump_Height AS Jump_Height_PPU,
           PP_FORCEPLATE AS PP_FORCEPLATE_PPU,
           Force_at_PP AS Force_at_PP_PPU,
           Vel_at_PP AS Vel_at_PP_PPU
    FROM PPU
""", conn)

df_i = pd.read_sql("SELECT Name, Creation_Date, Avg_Force AS Avg_Force_I FROM I", conn)
df_y = pd.read_sql("SELECT Name, Creation_Date, Avg_Force AS Avg_Force_Y FROM Y", conn)
df_t = pd.read_sql("SELECT Name, Creation_Date, Avg_Force AS Avg_Force_T FROM T", conn)
df_ir = pd.read_sql("SELECT Name, Creation_Date, Avg_Force AS Avg_Force_IR90 FROM IR90", conn)

# Merge data
df_merged = (
    df_cmj.merge(df_ppu, on=["Name", "Creation_Date"], how="outer")
          .merge(df_i, on=["Name", "Creation_Date"], how="outer")
          .merge(df_y, on=["Name", "Creation_Date"], how="outer")
          .merge(df_t, on=["Name", "Creation_Date"], how="outer")
          .merge(df_ir, on=["Name", "Creation_Date"], how="outer")
)

df_merged["Creation_Date"] = pd.to_datetime(df_merged["Creation_Date"])
df_merged.sort_values("Creation_Date", inplace=True)

participants = sorted(df_merged["Name"].dropna().unique())
conn.close()

print(f"Loaded {len(participants)} participants")

# Create minimal dashboard
app = Dash(__name__)

app.layout = html.Div([
    html.H2("Minimal Readiness Dashboard"),
    
    html.Div([
        html.Label("Select athlete:"),
        dcc.Dropdown(
            id="athlete",
            options=[{"label": n, "value": n} for n in participants],
            value=participants[0] if participants else None,
            clearable=False
        ),
    ], style={"width": "400px", "marginBottom": "20px"}),
    
    dcc.Graph(id="force-lines"),
    dcc.Graph(id="cmj-jump"),
])

@app.callback(
    Output("force-lines", "figure"),
    Output("cmj-jump", "figure"),
    Input("athlete", "value")
)
def update(name):
    print(f"Callback triggered for: {name}")
    
    dff = df_merged[df_merged["Name"] == name].sort_values("Creation_Date")
    dates_cat = dff["Creation_Date"].dt.strftime("%Y-%m-%d")
    
    print(f"Filtered data shape: {dff.shape}")
    print(f"Date categories: {dates_cat.tolist()}")
    
    # Force lines
    fig_force = go.Figure()
    for col, label in [("Avg_Force_I", "I"), ("Avg_Force_T", "T"),
                      ("Avg_Force_Y", "Y"), ("Avg_Force_IR90", "IR90")]:
        if dff[col].notna().any():
            print(f"Adding {label} trace with data: {dff[col].tolist()}")
            fig_force.add_trace(go.Scatter(
                x=dates_cat, y=dff[col], mode="lines+markers", name=label))
    
    fig_force.update_layout(
        title="Avg Force (I / T / Y / IR90)",
        template="plotly_dark",
        xaxis=dict(type="category"),
        xaxis_title="Session date",
        yaxis_title="Avg Force (N)"
    )
    
    # CMJ jump height
    fig_cmj = go.Figure()
    if dff["Jump_Height_CMJ"].notna().any():
        print(f"Adding CMJ trace with data: {dff['Jump_Height_CMJ'].tolist()}")
        fig_cmj.add_trace(go.Scatter(
            x=dates_cat, y=dff["Jump_Height_CMJ"],
            mode="lines+markers", name="CMJ Jump Height"))
    
    fig_cmj.update_layout(
        title="CMJ Jump Height",
        template="plotly_dark",
        xaxis=dict(type="category"),
        xaxis_title="Session date",
        yaxis_title="JH (cm)"
    )
    
    print(f"Returning figures with {len(fig_force.data)} and {len(fig_cmj.data)} traces")
    return fig_force, fig_cmj

if __name__ == "__main__":
    print("Starting minimal dashboard...")
    app.run_server(port=8052, debug=True, use_reloader=False)

