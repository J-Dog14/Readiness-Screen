import sqlite3
import pandas as pd
import numpy as np

# Test the callback logic exactly as it appears in the dashboard
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
    
    # Test the callback function logic
    def test_callback(name):
        print(f"\n=== Testing callback for '{name}' ===")
        
        # Filter data exactly like the callback does
        dff = df_merged[df_merged["Name"] == name].sort_values("Creation_Date")
        print(f"Filtered data shape: {dff.shape}")
        
        if dff.empty:
            print("ERROR: No data found for this participant!")
            return
        
        # Test date formatting
        dates_cat = dff["Creation_Date"].dt.strftime("%Y-%m-%d")
        print(f"Date categories: {dates_cat.tolist()}")
        
        # Test each plot data
        print("\n--- Force Lines Test ---")
        for col, label in [("Avg_Force_I","I"),("Avg_Force_T","T"),
                          ("Avg_Force_Y","Y"),("Avg_Force_IR90","IR90")]:
            has_data = dff[col].notna().any()
            print(f"{label}: {has_data} - {dff[col].tolist()}")
        
        print("\n--- CMJ Jump Height Test ---")
        has_cmj = dff["Jump_Height_CMJ"].notna().any()
        print(f"CMJ Jump Height: {has_cmj} - {dff['Jump_Height_CMJ'].tolist()}")
        
        print("\n--- PPU Jump Height Test ---")
        has_ppu = dff["Jump_Height_PPU"].notna().any()
        print(f"PPU Jump Height: {has_ppu} - {dff['Jump_Height_PPU'].tolist()}")
        
        # Test reference data
        cmj_ref = df_cmj[(df_cmj["Force_at_PP_CMJ"].notna()) & (df_cmj["Vel_at_PP_CMJ"].notna())].copy()
        ppu_ref = df_ppu[(df_ppu["Force_at_PP_PPU"].notna()) & (df_ppu["Vel_at_PP_PPU"].notna())].copy()
        
        print(f"\n--- Reference Data ---")
        print(f"CMJ reference shape: {cmj_ref.shape}")
        print(f"PPU reference shape: {ppu_ref.shape}")
        
        # Test scatter plot data
        sel_cmj = cmj_ref[cmj_ref["Name"]==name]
        sel_ppu = ppu_ref[ppu_ref["Name"]==name]
        
        print(f"Selected CMJ data: {sel_cmj.shape}")
        print(f"Selected PPU data: {sel_ppu.shape}")
        
        if not sel_cmj.empty:
            print(f"CMJ dates: {sel_cmj['Creation_Date'].unique()}")
        if not sel_ppu.empty:
            print(f"PPU dates: {sel_ppu['Creation_Date'].unique()}")
    
    # Test with first few participants
    for participant in participants[:3]:
        test_callback(participant)
    
    conn.close()
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

