import sqlite3
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.colors as plotly_colors

# Test a minimal version of the dashboard
db_path = r'D:/Readiness Screen 3/Readiness_Screen_Data_v2.db'

try:
    conn = sqlite3.connect(db_path)
    
    # Load data exactly like the dashboard does
    df_cmj = pd.read_sql("""
        SELECT Name, Creation_Date,
               Jump_Height             AS Jump_Height_CMJ,
               PP_FORCEPLATE           AS PP_FORCEPLATE_CMJ,
               Force_at_PP             AS Force_at_PP_CMJ,
               Vel_at_PP               AS Vel_at_PP_CMJ
        FROM CMJ
    """, conn)

    df_ppu = pd.read_sql("""
        SELECT Name, Creation_Date,
               Jump_Height             AS Jump_Height_PPU,
               PP_FORCEPLATE           AS PP_FORCEPLATE_PPU,
               Force_at_PP             AS Force_at_PP_PPU,
               Vel_at_PP               AS Vel_at_PP_PPU
        FROM PPU
    """, conn)

    df_i   = pd.read_sql("SELECT Name, Creation_Date, Avg_Force AS Avg_Force_I   FROM I"   , conn)
    df_y   = pd.read_sql("SELECT Name, Creation_Date, Avg_Force AS Avg_Force_Y   FROM Y"   , conn)
    df_t   = pd.read_sql("SELECT Name, Creation_Date, Avg_Force AS Avg_Force_T   FROM T"   , conn)
    df_ir  = pd.read_sql("SELECT Name, Creation_Date, Avg_Force AS Avg_Force_IR90 FROM IR90", conn)
    
    # Merge data exactly like dashboard
    df_merged = (
        df_cmj.merge(df_ppu, on=["Name", "Creation_Date"], how="outer")
              .merge(df_i , on=["Name", "Creation_Date"], how="outer")
              .merge(df_y , on=["Name", "Creation_Date"], how="outer")
              .merge(df_t , on=["Name", "Creation_Date"], how="outer")
              .merge(df_ir, on=["Name", "Creation_Date"], how="outer")
    )
    
    # Convert dates exactly like dashboard
    df_merged["Creation_Date"] = pd.to_datetime(df_merged["Creation_Date"])
    df_merged.sort_values("Creation_Date", inplace=True)
    
    participants = sorted(df_merged["Name"].dropna().unique())
    
    print(f"Loaded {len(participants)} participants")
    print(f"First participant: {participants[0]}")
    
    # Test the callback function directly
    def test_update(name):
        print(f"\n=== Testing update function for '{name}' ===")
        
        # Recreate reference dataframes
        cmj_ref = df_cmj[(df_cmj["Force_at_PP_CMJ"].notna()) & (df_cmj["Vel_at_PP_CMJ"].notna())].copy()
        ppu_ref = df_ppu[(df_ppu["Force_at_PP_PPU"].notna()) & (df_ppu["Vel_at_PP_PPU"].notna())].copy()
        
        dff = df_merged[df_merged["Name"] == name].sort_values("Creation_Date")
        dates_cat = dff["Creation_Date"].dt.strftime("%Y-%m-%d")
        
        print(f"Filtered data shape: {dff.shape}")
        print(f"Date categories: {dates_cat.tolist()}")
        
        # Test force lines
        fig_force = go.Figure()
        for col,label in [("Avg_Force_I","I"),("Avg_Force_T","T"),
                          ("Avg_Force_Y","Y"),("Avg_Force_IR90","IR90")]:
            if dff[col].notna().any():
                print(f"Adding {label} trace with data: {dff[col].tolist()}")
                fig_force.add_trace(go.Scatter(
                    x=dates_cat,y=dff[col],mode="lines+markers",name=label))
        
        print(f"Force figure has {len(fig_force.data)} traces")
        
        # Test CMJ
        fig_cmj = go.Figure()
        if dff["Jump_Height_CMJ"].notna().any():
            print(f"Adding CMJ trace with data: {dff['Jump_Height_CMJ'].tolist()}")
            fig_cmj.add_trace(go.Scatter(
                x=dates_cat,y=dff["Jump_Height_CMJ"],
                mode="lines+markers",name="CMJ Jump Height"))
        
        print(f"CMJ figure has {len(fig_cmj.data)} traces")
        
        return fig_force, fig_cmj
    
    # Test with first participant
    if participants:
        fig_force, fig_cmj = test_update(participants[0])
        print(f"\nFinal results:")
        print(f"Force figure traces: {len(fig_force.data)}")
        print(f"CMJ figure traces: {len(fig_cmj.data)}")
    
    conn.close()
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

