import sqlite3
import pandas as pd

# Test the exact same data loading as the dashboard
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
    
    print("=== Data Loading Results ===")
    print(f"CMJ shape: {df_cmj.shape}")
    print(f"PPU shape: {df_ppu.shape}")
    print(f"I shape: {df_i.shape}")
    print(f"Y shape: {df_y.shape}")
    print(f"T shape: {df_t.shape}")
    print(f"IR90 shape: {df_ir.shape}")
    
    print("\n=== CMJ Sample ===")
    print(df_cmj.head())
    
    print("\n=== PPU Sample ===")
    print(df_ppu.head())
    
    # Test the merge
    df_merged = (
        df_cmj.merge(df_ppu, on=["Name", "Creation_Date"], how="outer")
              .merge(df_i , on=["Name", "Creation_Date"], how="outer")
              .merge(df_y , on=["Name", "Creation_Date"], how="outer")
              .merge(df_t , on=["Name", "Creation_Date"], how="outer")
              .merge(df_ir, on=["Name", "Creation_Date"], how="outer")
    )
    
    print(f"\n=== Merged Data Shape: {df_merged.shape} ===")
    print("Merged sample:")
    print(df_merged.head())
    
    # Check participants
    participants = sorted(df_merged["Name"].dropna().unique())
    print(f"\n=== Participants ({len(participants)}): ===")
    print(participants[:10])  # Show first 10
    
    # Test filtering for a specific participant
    if participants:
        test_name = participants[0]
        print(f"\n=== Testing filter for '{test_name}' ===")
        dff = df_merged[df_merged["Name"] == test_name].sort_values("Creation_Date")
        print(f"Filtered data shape: {dff.shape}")
        print("Filtered data:")
        print(dff)
        
        # Check if any data exists
        print(f"\n=== Data availability for '{test_name}': ===")
        print(f"Jump_Height_CMJ: {dff['Jump_Height_CMJ'].notna().sum()} non-null values")
        print(f"Jump_Height_PPU: {dff['Jump_Height_PPU'].notna().sum()} non-null values")
        print(f"Avg_Force_I: {dff['Avg_Force_I'].notna().sum()} non-null values")
        print(f"Avg_Force_Y: {dff['Avg_Force_Y'].notna().sum()} non-null values")
        print(f"Avg_Force_T: {dff['Avg_Force_T'].notna().sum()} non-null values")
        print(f"Avg_Force_IR90: {dff['Avg_Force_IR90'].notna().sum()} non-null values")
    
    conn.close()
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

