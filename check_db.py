import sqlite3
import pandas as pd

# Check if database exists and what data is in it
db_path = r'D:/Readiness Screen 3/Readiness_Screen_Data_v2.db'

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print("Tables:", [t[0] for t in tables])
    
    # Check Participant table
    if any('Participant' in t[0] for t in tables):
        df_participants = pd.read_sql("SELECT * FROM Participant", conn)
        print(f"\nParticipant table shape: {df_participants.shape}")
        print("Participants:")
        print(df_participants)
    
    # Check CMJ table
    if any('CMJ' in t[0] for t in tables):
        df_cmj = pd.read_sql("SELECT * FROM CMJ LIMIT 5", conn)
        print(f"\nCMJ table shape: {df_cmj.shape}")
        print("CMJ sample data:")
        print(df_cmj)
    
    # Check PPU table
    if any('PPU' in t[0] for t in tables):
        df_ppu = pd.read_sql("SELECT * FROM PPU LIMIT 5", conn)
        print(f"\nPPU table shape: {df_ppu.shape}")
        print("PPU sample data:")
        print(df_ppu)
    
    conn.close()
    
except Exception as e:
    print(f"Error: {e}")

