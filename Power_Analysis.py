# Power Analysis Data Parser
# Prompts you to select folder and adds baseball measurement data into Power_Analysis.db

import os
import sqlite3
import tkinter as tk
from tkinter import filedialog
import xml.etree.ElementTree as ET
import pandas as pd

# Database file path
db_path = 'Power_Analysis.db'

# Establish connection and create tables
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# -------------------------------------------------
# 1. Create/Update Tables with Correct Column Order
# -------------------------------------------------
cursor.executescript("""
CREATE TABLE IF NOT EXISTS Power_Analysis (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT,
    Date_of_birth TEXT,
    Height REAL,
    Weight REAL,
    Filename TEXT,
    Comments TEXT,
    Creation_Date TEXT
);
""")
conn.commit()

# -------------------------------------------------
# 2. Prompt user to select a folder
# -------------------------------------------------
root = tk.Tk()
root.withdraw()
selected_folder = filedialog.askdirectory(initialdir='./')

if not selected_folder:
    print("No folder selected. Exiting...")
    exit()

# -------------------------------------------------
# 3. Locate the XML file (assuming 'session.xml')
# -------------------------------------------------
xml_file_path = ''
for root_dir, _, files in os.walk(selected_folder):
    for file in files:
        if file.lower() == 'session.xml':
            xml_file_path = os.path.join(root_dir, file)
            break
    if xml_file_path:
        break

if not xml_file_path:
    print("No session.xml file found. Exiting...")
    exit()

print(f"Found XML file: {xml_file_path}")

# -------------------------------------------------
# 4. Parse the XML file
# -------------------------------------------------
tree = ET.parse(xml_file_path)
xml_root = tree.getroot()

def find_text(element, tag):
    found = element.find(tag)
    return found.text if found is not None else None

# Find Subject information - Subject is the root element
subject = xml_root
if subject.tag != "Subject":
    print("Root element is not Subject. Exiting...")
    exit()

subject_fields = subject.find("Fields")
if subject_fields is None:
    print("No Subject Fields found in XML. Exiting...")
    exit()

name = find_text(subject_fields, "Name")
date_of_birth = find_text(subject_fields, "Date_of_birth")
height = find_text(subject_fields, "Height")
weight = find_text(subject_fields, "Weight")
creation_date = find_text(subject_fields, "Creation_date")

if None in [name, date_of_birth, height, weight, creation_date]:
    print("Missing Subject data in XML file. Exiting...")
    exit()

print(f"Subject: {name}")
print(f"Date of Birth: {date_of_birth}")
print(f"Height: {height}")
print(f"Weight: {weight}")
print(f"Creation Date: {creation_date}")

# -------------------------------------------------
# 5. Process Fastball measurements and insert data
# -------------------------------------------------

# Find all Measurement elements with Type="Fastball RH"
used_measurements_found = False
for measurement in xml_root.findall(".//Measurement[@Type='Fastball RH']"):
    fields = measurement.find("Fields")
    if fields is not None:
        used = find_text(fields, "Used")
        if used and used.lower() == 'true':
            used_measurements_found = True
            filename = measurement.get("Filename")
            comments = find_text(fields, "Comments")
            measurement_creation_date = find_text(fields, "Creation_date")
            
            if filename:
                # Insert data into single table
                cursor.execute("""
                INSERT INTO Power_Analysis (
                    Name, Date_of_birth, Height, Weight, Filename, Comments, Creation_Date
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    name, date_of_birth, height, weight, filename, comments or '', measurement_creation_date or creation_date
                ))
                print(f"Found used Fastball measurement: {filename} - Comments: {comments}")

# If no measurements with Used=True, look for any Fastball measurements
if not used_measurements_found:
    print("No Fastball measurements with Used=True found. Looking for any Fastball measurements...")
    for measurement in xml_root.findall(".//Measurement[@Type='Fastball RH']"):
        fields = measurement.find("Fields")
        if fields is not None:
            filename = measurement.get("Filename")
            comments = find_text(fields, "Comments")
            measurement_creation_date = find_text(fields, "Creation_date")
            
            if filename:
                # Insert data into single table
                cursor.execute("""
                INSERT INTO Power_Analysis (
                    Name, Date_of_birth, Height, Weight, Filename, Comments, Creation_Date
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    name, date_of_birth, height, weight, filename, comments or '', measurement_creation_date or creation_date
                ))
                print(f"Found Fastball measurement: {filename} - Comments: {comments}")

# -------------------------------------------------
# 6. Final Commit and Close
# -------------------------------------------------
conn.commit()
conn.close()

print(f"\nData successfully added to the database: {db_path}")
