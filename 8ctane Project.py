import sqlite3
import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import xml.etree.ElementTree as ET
from scipy import stats


class PowerAnalyzer:
    def __init__(self):
        self.folder_path = None
        self.db_path = None
        self.conn = None
        self.cursor = None

    def select_folder(self):
        """Use tkinter to select a folder containing power files"""
        root = tk.Tk()
        root.withdraw()  # Hide the main window

        folder_path = filedialog.askdirectory(
            title="Select folder containing power files"
        )

        if folder_path:
            self.folder_path = folder_path
            self.db_path = os.path.join(folder_path, "power_analysis.db")
            print(f"Selected folder: {self.folder_path}")
            return True
        else:
            messagebox.showwarning("No Selection", "No folder selected. Exiting.")
            return False

    def connect_to_existing_db(self):
        """Connect to Power_Analysis.db to make one combined Power_Analysis table"""

        if not self.folder_path:
            raise ValueError("No folder path set. Select folder before connecting to database.")

        self.db_path = os.path.join(self.folder_path, "Power_Analysis.db")
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.conn.execute("PRAGMA busy_timeout=5000;")

        # Combined table with demographics and power data
        self.cursor.execute("""
                            CREATE TABLE IF NOT EXISTS Power_Analysis
                            (
                                id
                                INTEGER
                                PRIMARY
                                KEY
                                AUTOINCREMENT,
                                Name
                                TEXT,
                                Date_of_Birth
                                TEXT,
                                Height
                                REAL,
                                Weight
                                REAL,
                                Filename
                                TEXT,
                                Comments
                                TEXT,
                                Creation_Date
                                TEXT,
                                pitch_id
                                INTEGER,
                                PowerALL_X
                                REAL,
                                PowerALL_Y
                                REAL,
                                PowerALL_Z
                                REAL,
                                PowerMAG
                                REAL,
                                Foot_Contact
                                REAL,
                                Release
                                REAL,
                                Release100ms
                                REAL,
                                Center_of_Mass_Vel2_X
                                REAL,
                                Center_of_Mass_Vel2_Y
                                REAL,
                                Center_of_Mass_Vel2_Z
                                REAL,
                                pelvis_peak_power
                                REAL,
                                pelvis_time_to_peak
                                REAL,
                                pelvis_auc
                                REAL,
                                shoulder_peak_power
                                REAL,
                                shoulder_time_to_peak
                                REAL,
                                shoulder_auc
                                REAL,
                                elbow_peak_power
                                REAL,
                                elbow_time_to_peak
                                REAL,
                                elbow_auc
                                REAL,
                                power_curve_peak_power
                                REAL,
                                power_curve_time_to_peak
                                REAL,
                                power_curve_auc
                                REAL,
                                fs1_peak_power
                                REAL,
                                fs1_time_to_peak
                                REAL,
                                fs1_auc
                                REAL
                            );
                            """)

        required_columns = [
            "pitch_id",
            "PowerALL_X", "PowerALL_Y", "PowerALL_Z",
            "Center_of_Mass_Vel2_X", "Center_of_Mass_Vel2_Y", "Center_of_Mass_Vel2_Z",
            "PowerMAG", "Foot_Contact", "Release", "Release100ms",
            "pelvis_peak_power", "pelvis_time_to_peak", "pelvis_auc",
            "shoulder_peak_power", "shoulder_time_to_peak", "shoulder_auc",
            "elbow_peak_power", "elbow_time_to_peak", "elbow_auc",
            "power_curve_peak_power", "power_curve_time_to_peak", "power_curve_auc",
            "fs1_peak_power", "fs1_time_to_peak", "fs1_auc"
        ]

        self.cursor.execute("PRAGMA table_info(Power_Analysis);")
        existing_columns = [col[1] for col in self.cursor.fetchall()]

        # Add any missing ones
        for col in required_columns:
            if col not in existing_columns:
                self.cursor.execute(f"ALTER TABLE Power_Analysis ADD COLUMN {col} REAL;")
                print(f"Added missing column: {col}")

        self.conn.commit()
        print("Connected to Power_Analysis.db (non-destructive) and ensured all columns exist.")

    def setup_database(db_path):
        """Create a single Power_Analysis table with demographics + power data."""
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"Deleted existing database at {db_path}")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        conn.execute("PRAGMA busy_timeout=5000;")

        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS Power_Analysis
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           Name
                           TEXT,
                           Date_of_Birth
                           TEXT,
                           Height
                           REAL,
                           Weight
                           REAL,
                           Filename
                           TEXT,
                           Comments
                           TEXT,
                           Creation_Date
                           TEXT,

                           -- pitch data
                           pitch_id
                           INTEGER,
                           PowerALL_X
                           REAL,
                           PowerALL_Y
                           REAL,
                           PowerALL_Z
                           REAL,
                           PowerMAG
                           REAL,
                           Center_of_Mass_Vel2_X
                           REAL,
                           Center_of_Mass_Vel2_Y
                           REAL,
                           Center_of_Mass_Vel2_Z
                           REAL,
                           Foot_Contact
                           REAL,
                           Release
                           REAL,
                           Release100ms
                           REAL,

                           -- power metrics
                           pelvis_peak_power
                           REAL,
                           pelvis_time_to_peak
                           REAL,
                           pelvis_auc
                           REAL,
                           shoulder_peak_power
                           REAL,
                           shoulder_time_to_peak
                           REAL,
                           shoulder_auc
                           REAL,
                           elbow_peak_power
                           REAL,
                           elbow_time_to_peak
                           REAL,
                           elbow_auc
                           REAL,
                           power_curve_peak_power
                           REAL,
                           power_curve_time_to_peak
                           REAL,
                           power_curve_auc
                           REAL,
                           fs1_peak_power
                           REAL,
                           fs1_time_to_peak
                           REAL,
                           fs1_auc
                           REAL
                       );
                       """)

        conn.commit()
        print("Power_Analysis table created successfully.")
        return conn, cursor

    def extract_filename(self, line):
        """Extract filename from data line"""
        filename = os.path.splitext(os.path.basename(line))[0]
        return filename

    def extract_pitch_id(self, filename):
        match = re.search(r'(?:LH|RH)\s*(\d+)', filename)
        if match:
            return int(match.group(1))
        return None

    def read_first_numeric_row_values(self, fobj):
        """Return list of floats from the first numeric line encountered."""
        for line in fobj:
            line = line.strip()
            if not line:
                continue
            if re.match(r'^[-+]?\d', line):  # numeric line
                return [float(tok) for tok in line.split()]
        return []

    def load_power_txt(self, txt_path: str):
        """Read power file and return 1-D numpy array of power values"""
        vals = []
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            # find first numeric row like: "1\t0.00000"
            in_numeric = False
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if not in_numeric and re.match(r'^\d+\s+', line):
                    in_numeric = True
                if in_numeric and re.match(r'^\d+\s+', line):
                    parts = re.split(r'\s+', line)
                    if len(parts) >= 2:
                        try:
                            vals.append(float(parts[1]))  # second column = power
                        except ValueError:
                            pass
        if not vals:
            raise ValueError(f"No numeric power values in {txt_path}")
        return np.asarray(vals, dtype=float)

    def load_multi_column_file(self, filename):
        # Load multicolumn metric file like powerAll and COMVelo2 and also single ones
        path = os.path.join(self.folder_path, filename)

        if not os.path.exists(path):
            print(f"{filename} not found in {self.folder_path}")
            return None, None, None

        try:
            data = []
            max_len = 0
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = re.split(r'\s+', line)
                    try:
                        row = [float(x) for x in parts]
                        data.append(row)
                        if len(row) > max_len:
                            max_len = len(row)
                    except ValueError:
                        continue  # skip non-numeric lines

            # Pad rows to same length with np.nan
            for i in range(len(data)):
                if len(data[i]) < max_len:
                    data[i] += [np.nan] * (max_len - len(data[i]))

            return np.array(data, dtype=float)

        except Exception as e:
            print(f"Error reading {filename}: {e}")
            return None

    def generate_power_curves_from_powerMAG(self):
        """ Generate individual powerCurve_pitchX.txt files from powerMAG.txt for each pitch. Handles inconsistent row lengths safely."""
        mag_path = os.path.join(self.folder_path, "powerMAG.txt")
        if not os.path.exists(mag_path):
            print(f"powerMAG.txt not found in {self.folder_path}")
            return []

        try:
            numeric_lines = []
            max_cols = 0

            # Read numeric rows and track max column count
            with open(mag_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line or re.match(r"[A-Za-z]", line):
                        continue
                    parts = re.split(r"\s+", line)
                    try:
                        row = [float(x) for x in parts]
                        numeric_lines.append(row)
                        if len(row) > max_cols:
                            max_cols = len(row)
                    except ValueError:
                        continue

            # Pad rows to same length
            for i in range(len(numeric_lines)):
                if len(numeric_lines[i]) < max_cols:
                    numeric_lines[i] += [np.nan] * (max_cols - len(numeric_lines[i]))

            data = np.array(numeric_lines)

            if data.ndim < 2 or data.shape[1] < 2:
                print("powerMAG.txt does not contain multiple pitch columns.")
                return []

            num_pitches = data.shape[1] - 1  # first column = item/time
            print(f"Detected {num_pitches} pitches in powerMAG.txt")

            curve_paths = []

            for pitch_idx in range(num_pitches):
                power_values = data[:, pitch_idx + 1]  # skip first column
                curve_path = os.path.join(self.folder_path, f"powerCurve_pitch{pitch_idx + 1}.txt")
                with open(curve_path, "w") as out:
                    for i, val in enumerate(power_values, start=1):
                        if np.isnan(val):
                            continue
                        out.write(f"{i}\t{val:.6f}\n")
                curve_paths.append(curve_path)
                print(f"Created {os.path.basename(curve_path)}")

            return curve_paths

        except Exception as e:
            print(f"Error generating power curves: {e}")
            return []

    def analyze_power_curve(self, power, fs_hz: float = 1000.0):
        """Base metrics for power curve analysis"""
        p = np.asarray(power, dtype=float)
        n = p.size
        t = np.arange(n) / fs_hz

        # Find peak power and time to peak
        pk_idx = int(np.nanargmax(p))
        peak_power = float(p[pk_idx])
        time_to_peak = float(t[pk_idx])

        # Calculate area under curve (AUC)
        auc = float(np.trapezoid(p, t))

        return {
            "peak_power": peak_power,
            "time_to_peak": time_to_peak,
            "auc": auc
        }

    def compute_metrics_per_axis(self, data):
        """Compute metrics (peak, time_to_peak, auc) for each axis of a pitch"""
        data = np.asarray(data)
        metrics = {}

        if data.ndim == 1:
            # single-column
            metrics['X'] = {
                'peak': np.max(data),
                'time_to_peak': np.argmax(data) / 1000,
                'auc': np.trapezoid(data)
            }
        else:
            # multi-column
            for i, axis in enumerate(['X', 'Y', 'Z']):
                col = data[:, i]
                metrics[axis] = {
                    'peak': np.max(col),
                    'time_to_peak': np.argmax(col) / 1000,
                    'auc': np.trapezoid(col)
                }
        return metrics

    def process_power_files(self):
        """Process all power files and calculate metrics"""
        power_files = {
            "pelvis": "pelvisPower.txt",
            "shoulder": "shoulderPower.txt",
            "elbow": "elbowPower.txt",
            "fs1": "FS1.txt"
        }

        power_metrics = {}

        for file_type, filename in power_files.items():
            file_path = os.path.join(self.folder_path, filename)

            if not os.path.exists(file_path):
                print(f"{filename} not found in {self.folder_path}")
                power_metrics[file_type] = {
                    "peak_power": None,
                    "time_to_peak": None,
                    "auc": None
                }
                continue

            try:
                power_data = self.load_power_txt(file_path)
                metrics = self.analyze_power_curve(power_data)
                power_metrics[file_type] = metrics
                print(f"Processed {filename}: Peak Power = {metrics['peak_power']:.2f}W")

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                power_metrics[file_type] = {
                    "peak_power": None,
                    "time_to_peak": None,
                    "auc": None
                }

        return power_metrics

    def process_power_files_for_pitch(self, pitch_idx):
        """Process power files for a specific pitch from multi-column files"""
        power_files = {
            "pelvis": "pelvisPower.txt",
            "shoulder": "shoulderPower.txt",
            "elbow": "elbowPower.txt",
            "fs1": "FS1.txt"
        }

        power_metrics = {}

        for file_type, filename in power_files.items():
            file_path = os.path.join(self.folder_path, filename)

            if not os.path.exists(file_path):
                print(f"{filename} not found in {self.folder_path}")
                power_metrics[file_type] = {
                    "peak_power": None,
                    "time_to_peak": None,
                    "auc": None
                }
                continue

            try:
                # Load multi-column data
                data = self.load_multi_column_file(filename)
                if data is None:
                    power_metrics[file_type] = {
                        "peak_power": None,
                        "time_to_peak": None,
                        "auc": None
                    }
                    continue

                # Extract data for this specific pitch (pitch_idx + 1 because first column is ITEM)
                if len(data[0]) > pitch_idx + 1:
                    pitch_data = data[:, pitch_idx + 1]  # Get the column for this pitch
                    # Remove NaN values
                    pitch_data = pitch_data[~np.isnan(pitch_data)]
                    
                    if len(pitch_data) > 0:
                        metrics = self.analyze_power_curve(pitch_data)
                        power_metrics[file_type] = metrics
                        print(f"Processed {filename} for pitch {pitch_idx + 1}: Peak Power = {metrics['peak_power']:.2f}W")
                    else:
                        power_metrics[file_type] = {
                            "peak_power": None,
                            "time_to_peak": None,
                            "auc": None
                        }
                else:
                    power_metrics[file_type] = {
                        "peak_power": None,
                        "time_to_peak": None,
                        "auc": None
                    }

            except Exception as e:
                print(f"Error processing {filename} for pitch {pitch_idx + 1}: {e}")
                power_metrics[file_type] = {
                    "peak_power": None,
                    "time_to_peak": None,
                    "auc": None
                }

        return power_metrics

    def insert_power_data(self, pitch_id, power_exports_data, power_metrics):
        """Insert one row of pitch data into the database"""
        cols = [
            "pitch_id",
            "PowerALL_X", "PowerALL_Y", "PowerALL_Z",
            "Center_of_Mass_Vel2_X", "Center_of_Mass_Vel2_Y", "Center_of_Mass_Vel2_Z",
            "PowerMAG", "Foot_Contact", "Release", "Release100ms",
            "pelvis_peak_power", "pelvis_time_to_peak", "pelvis_auc",
            "shoulder_peak_power", "shoulder_time_to_peak", "shoulder_auc",
            "elbow_peak_power", "elbow_time_to_peak", "elbow_auc",
            "power_curve_peak_power", "power_curve_time_to_peak", "power_curve_auc",
            "fs1_peak_power", "fs1_time_to_peak", "fs1_auc"
        ]

        vals = [
            pitch_id,
            power_exports_data.get("PowerALL_X"),
            power_exports_data.get("PowerALL_Y"),
            power_exports_data.get("PowerALL_Z"),
            power_exports_data.get("COMVelo2_X"),
            power_exports_data.get("COMVelo2_Y"),
            power_exports_data.get("COMVelo2_Z"),
            power_exports_data.get("PowerMAG"),
            power_exports_data.get("footContact"),
            power_exports_data.get("release"),
            power_exports_data.get("releaseAfter"),
            power_metrics["pelvis"]["peak_power"],
            power_metrics["pelvis"]["time_to_peak"],
            power_metrics["pelvis"]["auc"],
            power_metrics["shoulder"]["peak_power"],
            power_metrics["shoulder"]["time_to_peak"],
            power_metrics["shoulder"]["auc"],
            power_metrics["elbow"]["peak_power"],
            power_metrics["elbow"]["time_to_peak"],
            power_metrics["elbow"]["auc"],
            power_metrics["power_curve"]["peak_power"],
            power_metrics["power_curve"]["time_to_peak"],
            power_metrics["power_curve"]["auc"],
            power_metrics["fs1"]["peak_power"],
            power_metrics["fs1"]["time_to_peak"],
            power_metrics["fs1"]["auc"]
        ]

        placeholders = ",".join(["?"] * len(vals))
        self.cursor.execute(f"INSERT INTO Power_Analysis ({','.join(cols)}) VALUES ({placeholders})", vals)
        self.conn.commit()

    def run_analysis(self):
        """Run full power analysis for all pitches in the folder"""
        print("Starting Power Analysis...")

        # select folder
        if not self.select_folder():
            return False

        # connect to existing database
        self.connect_to_existing_db()

        # 3️⃣ Parse XML for demographics and measurement info
        xml_file_path = ''
        for root_dir, _, files in os.walk(self.folder_path):
            for file in files:
                if file.lower() == 'session.xml':
                    xml_file_path = os.path.join(root_dir, file)
                    break
            if xml_file_path:
                break
        if not xml_file_path:
            print("No session.xml found in folder.")
            return False

        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        def find_text(element, tag):
            found = element.find(tag)
            return found.text if found is not None else None

        # Subject info
        subject_fields = root.find("Fields")
        if subject_fields is None:
            print("No Subject Fields in XML")
            return False

        subject_info = {
            "name": find_text(subject_fields, "Name"),
            "dob": (find_text(subject_fields, "Date_of_Birth") or find_text(subject_fields, "Date_of_birth") or find_text(subject_fields, "date_of_birth")),
            "height": find_text(subject_fields, "Height"),
            "weight": find_text(subject_fields, "Weight"),
            "creation_date": find_text(subject_fields, "Creation_date")
        }

        # --- Normalize / clean subject info before inserting
        subject_info["name"] = subject_info["name"].strip() if subject_info["name"] else None
        subject_info["dob"] = subject_info["dob"].strip() if subject_info["dob"] else None
        subject_info["height"] = float(subject_info["height"]) if subject_info["height"] else None
        subject_info["weight"] = float(subject_info["weight"]) if subject_info["weight"] else None
        subject_info["creation_date"] = subject_info["creation_date"].strip() if subject_info["creation_date"] else None

        # Measurement info (used Fastball RH measurements)
        measurement_info = []
        for measurement in root.findall(".//Measurement[@Type='Fastball RH']"):
            fields = measurement.find("Fields")
            if fields is None:
                continue
            used = find_text(fields, "Used")
            if used and used.lower() == 'true':
                filename = measurement.get("Filename")
                comments = find_text(fields, "Comments") or ''
                m_creation_date = find_text(fields, "Creation_date") or subject_info["creation_date"]
                if filename:
                    measurement_info.append((filename, comments, m_creation_date))

        if not measurement_info:
            print("No Fastball measurements found. Exiting.")
            return False

        # Create a mapping of pitch numbers to measurements
        # For now, we'll use the first measurement for all pitches
        # You can enhance this to match specific pitches to specific measurements
        if measurement_info:
            print(f"Found {len(measurement_info)} used measurements")
            for i, (filename, comments, creation_date) in enumerate(measurement_info):
                print(f"Measurement {i+1}: {filename}")
        else:
            print("No used measurements found!")
            return False

        # load all multi-column files
        powerAll_data = self.load_multi_column_file("powerAll.txt")
        COMVelo2_data = self.load_multi_column_file("COMVelo2.txt")
        powerMAG_data = self.load_multi_column_file("powerMAG.txt")
        footContact_data = self.load_multi_column_file("footContact.txt")
        release_data = self.load_multi_column_file("release.txt")
        releaseAfter_data = self.load_multi_column_file("releaseAfter.txt")

        if powerAll_data is None:
            print("Failed to load required files")
            return False

        # determine number of pitches (3 columns per pitch + 1 item column)
        num_columns = len(powerAll_data[0])
        num_pitches = (num_columns - 1) // 3
        print(f"Detected {num_pitches} pitches in powerAll.txt")

        # generate power curves and compute metrics
        curve_paths = self.generate_power_curves_from_powerMAG()

        # loop over each pitch
        for pitch_idx in range(num_pitches):
            pitch_id = pitch_idx + 1

            # Get the measurement info for this pitch (use modulo to cycle through measurements)
            measurement_idx = pitch_idx % len(measurement_info)
            measurement_filename, measurement_comments, measurement_creation_date = measurement_info[measurement_idx]
            print(f"Pitch {pitch_id} using measurement: {measurement_filename}")

            # Calculate power metrics for this specific pitch
            power_metrics = self.process_power_files_for_pitch(pitch_idx)
            
            # Compute metrics for this pitch's power curve
            power_metrics["power_curve"] = {}
            if curve_paths and pitch_idx < len(curve_paths):
                curve_vals = self.load_power_txt(curve_paths[pitch_idx])
                power_metrics["power_curve"] = self.analyze_power_curve(curve_vals)
            else:
                power_metrics["power_curve"] = {"peak_power": None, "time_to_peak": None, "auc": None}

            # Foot contact / release values (use first row if variable length)
            foot_contact = None
            release = None
            release100ms = None
            
            if footContact_data is not None and len(footContact_data) > 0 and len(footContact_data[0]) > pitch_idx + 1:
                foot_contact = footContact_data[0][pitch_idx + 1]
            
            if release_data is not None and len(release_data) > 0 and len(release_data[0]) > pitch_idx + 1:
                release = release_data[0][pitch_idx + 1]
                
            if releaseAfter_data is not None and len(releaseAfter_data) > 0 and len(releaseAfter_data[0]) > pitch_idx + 1:
                release100ms = releaseAfter_data[0][pitch_idx + 1]

            # --- Iterate through every row (sample) in the pitch ---
            for row_idx, row in enumerate(powerAll_data):
                if len(row) < (pitch_idx * 3 + 4):  # include item + 3 columns
                    continue

                # item value = first column in the file
                item_val = row[0]

                # PowerALL columns (X, Y, Z)
                power_vals = row[pitch_idx * 3 + 1: pitch_idx * 3 + 4]

                # COMVelo2 columns (X, Y, Z)
                if COMVelo2_data is not None and row_idx < len(COMVelo2_data):
                    com_vals = COMVelo2_data[row_idx][pitch_idx * 3 + 1: pitch_idx * 3 + 4]
                else:
                    com_vals = [None, None, None]

                # PowerMAG value (single column per pitch)
                if powerMAG_data is not None and row_idx < len(powerMAG_data):
                    if len(powerMAG_data[row_idx]) > pitch_idx + 1:
                        powerMAG_val = powerMAG_data[row_idx][pitch_idx + 1]
                    else:
                        powerMAG_val = None
                else:
                    powerMAG_val = None

                # --- Insert this sample row into database ---
                vals = (
                    subject_info["name"],
                    subject_info["dob"],
                    subject_info["height"],
                    subject_info["weight"],
                    measurement_filename,  # Use measurement filename
                    measurement_comments,  # Use measurement comments
                    measurement_creation_date,  # Use measurement creation date
                    pitch_id,
                    power_vals[0],
                    power_vals[1],
                    power_vals[2],
                    powerMAG_val,
                    foot_contact,
                    release,
                    release100ms,
                    com_vals[0],
                    com_vals[1],
                    com_vals[2],
                    # Add calculated power metrics
                    power_metrics["pelvis"]["peak_power"],
                    power_metrics["pelvis"]["time_to_peak"],
                    power_metrics["pelvis"]["auc"],
                    power_metrics["shoulder"]["peak_power"],
                    power_metrics["shoulder"]["time_to_peak"],
                    power_metrics["shoulder"]["auc"],
                    power_metrics["elbow"]["peak_power"],
                    power_metrics["elbow"]["time_to_peak"],
                    power_metrics["elbow"]["auc"],
                    power_metrics["power_curve"]["peak_power"],
                    power_metrics["power_curve"]["time_to_peak"],
                    power_metrics["power_curve"]["auc"],
                    power_metrics["fs1"]["peak_power"],
                    power_metrics["fs1"]["time_to_peak"],
                    power_metrics["fs1"]["auc"]
                )
                
                # Debug: Print the number of values
                if row_idx == 0:  # Only print for first row of each pitch
                    print(f"DEBUG: Pitch {pitch_id} - Number of values: {len(vals)}")
                    print(f"DEBUG: First few values: {vals[:5]}")
                    print(f"DEBUG: Last few values: {vals[-5:]}")

                self.cursor.execute("""
                    INSERT INTO Power_Analysis (
                        Name, Date_of_Birth, Height, Weight,
                        Filename, Comments, Creation_Date, pitch_id,
                        PowerALL_X, PowerALL_Y, PowerALL_Z,
                        PowerMAG, Foot_Contact, Release, Release100ms,
                        Center_of_Mass_Vel2_X, Center_of_Mass_Vel2_Y, Center_of_Mass_Vel2_Z,
                        pelvis_peak_power, pelvis_time_to_peak, pelvis_auc,
                        shoulder_peak_power, shoulder_time_to_peak, shoulder_auc,
                        elbow_peak_power, elbow_time_to_peak, elbow_auc,
                        power_curve_peak_power, power_curve_time_to_peak, power_curve_auc,
                        fs1_peak_power, fs1_time_to_peak, fs1_auc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, vals)

            # Commit after each pitch
            self.conn.commit()
            print(f"Inserted pitch {pitch_id} data ({len(powerAll_data)} samples).")

        self.conn.close()
        print("Power analysis completed")
        print(f"Database saved at: {self.db_path}")
        return True


def main():
    """Main entry point"""
    analyzer = PowerAnalyzer()
    success = analyzer.run_analysis()

    if success:
        print("\nAnalysis completed")
    else:
        print("\nAnalysis failed")


if __name__ == "__main__":
    main()