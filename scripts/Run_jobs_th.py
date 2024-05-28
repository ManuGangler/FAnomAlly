#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 11:10:32 2024

@author: mohamadjouni
"""

from concurrent.futures import ThreadPoolExecutor
import runpy
import io
import requests
import sys
import time
import pandas as pd
import subprocess
import os

# Set the correct path to your script
script_path = '/Users/mohamadjouni/work/FAnomAlly/scripts/classificcation_transfer.py'

# Check if the script exists at the specified path
if not os.path.exists(script_path):
    raise FileNotFoundError(f"The script at {script_path} was not found.")

def read_unique_ids_from_file(file_path):
    # Read the file and store each line as an element in a list
    with open(file_path, 'r') as file:
        unique_ids = file.read().splitlines()
    
    return unique_ids



unique_ids = read_unique_ids_from_file('unique_ids.txt')


nb_Ids = len(unique_ids)

# Initialize the list to store dictionaries
dict_list = []

# Split nb_Ids into chunks of 100
start_time = time.time()

def run_script(Id_first, Id_last, folder_name):
    # Construct the command to run the script with arguments
    command = [
        'python', script_path, 
        str(Id_first), str(Id_last), folder_name
    ]
    
    # Run the script with the specified arguments
    result = subprocess.run(command, capture_output=True, text=True)
    
    # Print the output or check for errors
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")

with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust the number of workers as needed
    futures = []
    for i in range(0, nb_Ids, 100):
        Id_first = i + 1
        Id_last = min(i + 100, nb_Ids)
        folder_name = 'jobs_th2'
        
        futures.append(executor.submit(run_script, Id_first, Id_last, folder_name))
    
    # Wait for all tasks to complete
    for future in futures:
        future.result()

end_time = time.time()
elapsed_time = end_time - start_time

print("Temps écoulé:", elapsed_time, "secondes")
