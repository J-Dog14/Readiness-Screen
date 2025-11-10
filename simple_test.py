print("Testing basic functionality...")

try:
    import sqlite3
    print("✓ sqlite3 imported")
    
    import pandas as pd
    print("✓ pandas imported")
    
    import plotly.graph_objects as go
    print("✓ plotly imported")
    
    # Test database connection
    db_path = r'D:/Readiness Screen 3/Readiness_Screen_Data_v2.db'
    conn = sqlite3.connect(db_path)
    print("✓ Database connected")
    
    # Test simple query
    df = pd.read_sql("SELECT COUNT(*) as count FROM Participant", conn)
    print(f"✓ Query successful: {df['count'].iloc[0]} participants")
    
    conn.close()
    print("✓ Database closed")
    
    print("\nAll tests passed!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

