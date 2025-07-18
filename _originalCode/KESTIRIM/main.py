import json
import os
import platform
import subprocess
import sys
import random
import csv
import cantera as ct
import numpy as np
import colorama
from colorama import init, Fore

# Initialize colorama for colored output
init(autoreset=True)

class HTSetup:
    def __init__(self):
        self.std_atm_data = None
        self.flight_data = None
        self.network_config = None
        self.script_dir = os.path.dirname(os.path.realpath(__file__))
        self.std_atm_path = os.path.join(self.script_dir, "std_atm.txt")
        self.network_file = os.path.join(self.script_dir, "network.json")

    def initialize(self):
        print('High Temperature 1D Steady Gas Dynamics')
        try:
            print(' > Running Cantera Version: ' + colorama.Fore.GREEN + ct.__version__)
        except:
            print(' > ' + Fore.RED + "Cantera could not be located.")
        print(' > Last update:'  + Fore.GREEN + ' 15.12.2023', end = "\n\n")
        self.import_modules()
        self.load_std_atm()
        self.generate_default_network()  # Ensure network.json exists with title
        input('This script is used to import/export with ease. Press to continue with setup.')
        self.clear_screen()

    def load_std_atm(self):
        print('Importing standart atmosphere...')
        try:
            with open(self.std_atm_path, 'r') as f:
                self.std_atm_data = list(csv.reader(f, delimiter='\t'))
            self.std_atm = np.array(self.std_atm_data, dtype = float)
            self.std_atm[:, 1], self.std_atm[:, 3] = self.std_atm[:, 1] * 101325, self.std_atm[:, 3] * 1.225
            self.std_atm = np.delete(self.std_atm, 4, 1)
            print(' > ' + Fore.GREEN + 'Imported std_atm')
        except:
            print(' > ' + Fore.RED + "Couldn't import standart atmosphere. Make sure std_atm.txt is in the directory.")
            sys.exit(1)
            
    def import_modules(self):
        print('Importing modules...')
        try:
            import htshock2
            print(' > ' + Fore.GREEN + 'Imported htshock2')
            from htshock2 import nshock, oshock, cshock
        except:
            print(' > ' + Fore.RED + "Couldn't import hsthock module. Make sure it is in the directory.")
            sys.exit(1)
        
        try:
            import htransport
            print(' > ' + Fore.GREEN + 'Imported htransport')
        except:
            print(' > ' + Fore.RED + "Couldn't import htransport module. Make sure it is in the directory.")
            sys.exit(1)
            
        try:
            import htcorrelate
            print(' > ' + Fore.GREEN + 'Imported htcorrelate')
        except:
            print(' > ' + Fore.RED + "Couldn't import htcorrelate module. Make sure it is in the directory.")
            sys.exit(1)

    def display_menu(self):
        self.clear_screen()
        print("HT Shock Setup")
        print("==================")
        if self.flight_data is not None:
            file_name = os.path.basename(self.loaded_flight_file) if hasattr(self, 'loaded_flight_file') else "Not loaded"
            print(f"1. Flight data: {file_name} (Timestep: {self.time_step:.6f}, Duration: {self.total_time})")
        else:
            print("1. Flight data: Not loaded")

        # Display network configuration details with Downsample
        print(self.load_and_display_network_config())  # Ensure this includes Downsample as before

        print("S. Stagnation | N. Network | B. Both")
        print("Q. Quit")

    def load_flight_data(self):
        # If data is already loaded, offer to show the current sample or load new data
        if self.flight_data is not None:
            print("Current Flight Data Sample:")
            self.display_flight_data_sample()
            choice = input("Load new flight data file? (y/n): ").lower()
            if choice != 'y':
                return  # If the user decides not to load new data, return immediately

        file_path = input("Enter path to flight data file or 'q' to return: ").strip()
        if file_path.lower() == 'q':
            return  # If the user decides to return, do so immediately

        try:
            # Attempt to load and process the new file
            with open(file_path, 'r') as file:
                reader = csv.reader(file, delimiter=';')
                data = []
                for row in reader:
                    try:
                        float_row = [float(cell) for cell in row]
                        data.append(float_row)
                    except ValueError:
                        print(Fore.RED + "Error loading flight data: Non-numeric value encountered.")
                        return  # Return immediately if a non-numeric value is encountered

            # Update flight data and display the sample only after successful loading
            self.flight_data = np.array(data)
            self.loaded_flight_file = file_path  # Store the new file path
            time_data = self.flight_data[:, 0]
            self.time_step = time_data[1] - time_data[0]
            self.total_time = time_data[-1]

            print(Fore.GREEN + f"Flight data file '{file_path}' loaded successfully.")
            print(f"Timestep: {self.time_step}, Total Time: {self.total_time}")
        except Exception as e:
            print(Fore.RED + f"Error loading flight data: {e}")
            return  # Return if there was an error opening the file

        # Display the data sample here to ensure it's only done once after a successful load
        self.display_flight_data_sample()

    def display_flight_data_sample(self):
        if self.flight_data is not None:
            print("\nFlight Data Sample:")
            num_rows = len(self.flight_data)
            sample_size = 3  # Number of rows to show from the start and end

            # Define the number of decimal places to display
            decimal_places = 6

            # Function to format a single row with consistent spacing
            def format_row(row):
                return " ".join(f"{item:20.{decimal_places}f}" for item in row)

            # Display the first few rows
            for row in self.flight_data[:sample_size]:
                print(format_row(row))

            # Indicate truncation with ellipses if the dataset is large
            if num_rows > 2 * sample_size:
                print(" " * 10 + "...")

            # Display the last few rows for large datasets
            if num_rows > sample_size:
                for row in self.flight_data[-sample_size:]:
                    print(format_row(row))
        else:
            print(Fore.RED + "No flight data loaded.")

    def load_and_display_network_config(self):
        network_details = "2. Gas network: (network.json)\n"
        try:
            with open(self.network_file, 'r') as file:
                network_config = json.load(file)
            
            # Display the title from the network configuration
            config_title = network_config.get("title", "N/A")
            network_details += f"      - Title: {config_title}\n"
            
            # Display Downsample and other details
            downsample = network_config.get("downsample", "N/A")
            network_details += f"      - Downsample: {downsample}\n"
            
            for n_type in network_config.get("types", []):
                if n_type["type"] == "Stagnation":
                    details = f"Rnose: {n_type.get('Rnose', 'N/A')}, Angles: {', '.join(map(str, n_type.get('Angles', [])))}"
                else:
                    details = f"Angle: {n_type.get('Angle', 'N/A')}, Pos: {', '.join(map(str, n_type.get('Pos', [])))}"
                network_details += f"      - Type: {n_type['type']}, {details}\n"
        except FileNotFoundError:
            network_details += "      - Configuration not loaded"

        return network_details

    def update_network(self):
        if not os.path.exists(self.network_file):
            print(Fore.RED + "Network configuration file not found. Generating default.")
            self.generate_default_network()
        subprocess.run(["notepad", self.network_file])
        self.load_network()

    def load_network(self):
        try:
            with open(self.network_file, 'r') as file:
                self.network_config = json.load(file)
                if not self.validate_network_config():
                    print(Fore.RED + "Invalid network configuration. Please correct the network.json file.")
                    # Optional: Load a default configuration or handle the error as needed
        except FileNotFoundError:
            print(Fore.RED + "Error: Network configuration file not found.")
            # Optional: Generate default network configuration here

    def validate_network_config(self):
        valid = True
        error_messages = []

        if not self.network_config.get('title'):
            valid = False
            error_messages.append("Title cannot be empty.")

        downsample = self.network_config.get('downsample', -1)
        if downsample < 0:
            valid = False
            error_messages.append("Downsample must be 0 or positive.")

        types_encountered = set()
        for i, n_type in enumerate(self.network_config.get("types", [])):
            type_name = n_type.get("type")
            if type_name not in ["Stagnation", "Conic", "PM"]:
                valid = False
                error_messages.append(f"Invalid type: {type_name}. Only 'Stagnation', 'Conic', 'PM' are allowed.")
                continue

            if type_name == "Stagnation":
                if types_encountered:
                    valid = False
                    error_messages.append("Stagnation type must be the first entry if present.")
                rnose = n_type.get("Rnose", -1)
                if rnose <= 0:
                    valid = False
                    error_messages.append("Stagnation Rnose must be positive.")
                angles = n_type.get("Angles", [])
                if not all(0 <= angle <= 90 for angle in angles):
                    valid = False
                    error_messages.append("Stagnation angles must be between 0 and 90.")

            elif type_name in ["Conic", "PM"]:
                angle = n_type.get("Angle", -1)
                if not (0 <= angle <= 90):
                    valid = False
                    error_messages.append(f"{type_name} angle must be between 0 and 90.")
                pos = n_type.get("Pos", [])
                if not all(p > 0 for p in pos):
                    valid = False
                    error_messages.append(f"{type_name} Pos must be positive.")

            types_encountered.add(type_name)

        if "Stagnation" in types_encountered and list(types_encountered)[0] != "Stagnation":
            valid = False
            error_messages.append("Stagnation, if present, must be the first type.")

        if not valid:
            print(Fore.RED + "Network configuration validation failed with the following issues:")
            for msg in error_messages:
                print(Fore.RED + "- " + msg)
        return valid

    def generate_default_network(self):
        # Check if the network.json file already exists
        if os.path.exists(self.network_file):
            print(Fore.YELLOW + "Network configuration file already exists. Skipping generation.")
            return

        # If the file doesn't exist, generate the default network configuration
        default_network_config = {
            "title": "default_config",
            "downsample": 0.25,
            "types": [
                {"type": "Stagnation", "Rnose": 0.1, "Angles": [15, 30, 45, 60, 70]},
                {"type": "Conic", "Angle": 20, "Pos": [0.4, 0.5, 0.6, 0.7]},
                {"type": "Conic", "Angle": 10, "Pos": [0.7, 0.85, 1.0]},
                {"type": "PM", "Angle": 5, "Pos": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]}
            ]
        }
        with open(self.network_file, 'w') as file:
            json.dump(default_network_config, file, indent=4)
        print(Fore.GREEN + "Default network configuration file generated successfully.")

    def process_network_selection(self, selection):
        # Check if flight_data has been loaded by checking if it's not None
        if self.flight_data is None:
            print(Fore.RED + "Flight data not loaded.")
            return

        # Check if network_config has been loaded by checking if it's not None
        if self.network_config is None:
            print(Fore.RED + "Network configuration not loaded.")
            return

        # Initial flight data setup
        current_data = self.flight_data  # Assumes self.flight_data is ready for processing

        # Check and process "Stagnation" if selected and defined in the network config
        if selection == 's':
            stagnation_found = False
            for n_type in self.network_config.get("types", []):
                if n_type["type"] == "Stagnation":
                    rnose = n_type.get("Rnose")
                    angles = n_type.get("Angles")
                    # Apply the Stagnation function
                    # current_data = stagnation(current_data, rnose, angles)
                    print("Processed Stagnation.")
                    stagnation_found = True
                    break  # Stop after processing the first Stagnation type found

            if not stagnation_found:
                print(Fore.YELLOW + "No Stagnation type defined in the network configuration.")

        # Sequential processing for "Conic" and "PM" types if "Network" is selected
        elif selection == 'n':
            for n_type in self.network_config.get("types", []):
                if n_type["type"] == "Conic":
                    angle = n_type.get("Angle")
                    pos = n_type.get("Pos")
                    # Apply Conic function and update current_data for the next iteration
                    # current_data = cshock(current_data, angle, pos)
                    print("Processed Conic.")

                elif n_type["type"] == "PM":
                    angle = n_type.get("Angle")
                    pos = n_type.get("Pos")
                    # Apply PM function and update current_data for the next iteration
                    # current_data = pm(current_data, angle, pos)
                    print("Processed PM.")

        # Update self.flight_data with the final result
        self.flight_data = current_data


    def process_stagnation(self, flight_data, rnose, angles):
        # Placeholder for Stagnation processing logic
        print("Processing Stagnation with Rnose:", rnose, "and Angles:", angles)
        # Return modified flight data as a result
        return flight_data

    def process_conic(self, flight_data, angle, pos):
        # Placeholder for Conic processing logic
        print("Processing Conic with Angle:", angle, "and Pos:", pos)
        # Return modified flight data as a result
        return flight_data

    def process_pm(self, flight_data, angle, pos):
        # Placeholder for PM processing logic
        print("Processing PM with Angle:", angle, "and Pos:", pos)
        # Return modified flight data as a result
        return flight_data

    def process_data(self, choice):
        # Placeholder for data processing logic
        print(Fore.YELLOW + f"Processing {choice} data... (not implemented)")

    def clear_screen(self):
        if platform.system() == "Windows":
            os.system('cls')
        else:
            os.system('clear')

    def run(self):
        self.initialize()
        while True:
            self.display_menu()
            choice = input("Enter your choice: ").lower()

            if choice == '1':
                self.load_flight_data()
                # Remove the direct call to self.display_flight_data_sample() here since it's already called within load_flight_data
                input("\nPress Enter to continue...")
            elif choice == '2':
                self.update_network()
                input("\nPress Enter to continue...")
            elif choice in ['s', 'n']:
                self.process_network_selection(choice)
                input("\nPress Enter to continue...")
            elif choice == 'q':
                break
            else:
                print(Fore.RED + "Invalid choice. Please try again.")   

if __name__ == "__main__":
    system = HTSetup()
    system.run()
