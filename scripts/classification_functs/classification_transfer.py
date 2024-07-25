#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:25:44 2024

@author: mohamadjouni
"""
import io
import requests
import sys
import time
import pandas as pd
import os 
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def save_unique_ids_to_file(parquet_file_path, output_file_path):
    # Read the parquet file
    pdf = pd.read_parquet(parquet_file_path)
    
    # Get unique IDs
    unique_ids = pdf['objectId'].unique().tolist()
    
    # Write unique IDs to a file
    with open(output_file_path, 'w') as file:
        for Id in unique_ids:
            file.write(f"{Id}\n")
    
    print(f"Unique IDs have been saved to {output_file_path}")




def read_unique_ids_from_file(file_path):
    # Read the file and store each line as an element in a list
    with open(file_path, 'r') as file:
        unique_ids = file.read().splitlines()
    
    return unique_ids






def get_classification3(Id):
    #print(Id)
    r = requests.post(
        'https://fink-portal.org/api/v1/objects',
        json={
            'objectId': Id,
        }
    )

    # Format output in a DataFrame
    pdf = pd.read_json(io.BytesIO(r.content))

    # Get value counts for each classification
    class_counts = pdf['v:classification'].value_counts()

    # Calculate total count
    total_count = class_counts.sum()

    # Calculate percentage for each classification
    class_percentages = (class_counts / total_count * 100).round(1)
    error_bar = np.sqrt(class_counts*(total_count-class_counts)) * (total_count)**(-3/2)

    # Create a DataFrame with the results
    result_df = pd.DataFrame({
        'classification': class_percentages.index,
        'percentage': class_percentages.values, 
        'error_bar' : error_bar
    })

    return result_df


    # Convert to dictionary
    #class_percentages_dict = class_percentages.to_dict()
    #return class_percentages_dict




def process_id(Id, folder_name):
    file_name = f'{Id}.parquet'
    file_path = os.path.join(folder_name, file_name)
    
    if not os.path.exists(file_path):
        dataframe = get_classification3(Id)
        
        dataframe.to_parquet(file_path, index=False)
        #print(f"File '{file_path}' created and DataFrame saved.")
    #else:
    #   print(f"File '{file_path}' already exists.")
        
        
        
def main():
    num_cores =  4#os.cpu_count()

    # Check if there are enough arguments
    if len(sys.argv) < 4:
        print("Usage: python script.py <number of Ids> <Folder_name> ... <argN>")
        sys.exit(1)
    
    # Access the arguments
    args = sys.argv[1:]  # Exclude the script name
    #print("Arguments received:", args)

    folder_name = args[2]
        
    # Check if the folder exists
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created.")

    unique_ids = read_unique_ids_from_file('unique_ids.txt')
    
    first = int(args[0])
    last = int(args[1])
    #if nb_Ids == 0:
    #   nb_Ids = len(unique_ids)
        
    # Process IDs concurrently
    with ThreadPoolExecutor(max_workers=num_cores) as executor:  # Adjust the number of workers as needed
        futures = [executor.submit(process_id, Id, folder_name) for Id in unique_ids[first:last]]
        for future in futures:
            future.result()  # Wait for all tasks to complete
            
    print(first, last)


if __name__ == "__main__":
    #parquet_file_path = '../../ftransfer_ztf_2024-02-01_689626'
    #output_file_path = 'unique_ids.txt'
    #save_unique_ids_to_file(parquet_file_path, output_file_path)
    
    start_time = time.time()
    
    main()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Temps écoulé:", elapsed_time/60, "minutes")



