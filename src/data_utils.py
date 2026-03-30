import pandas as pd
import os
path = os.chdir(r'D:\Machine_Learning\Telco_churn\Data\processed')

def load_data(file_name = 'cleaned.csv'):
    try:
        df = pd.read(file_name)
        print("Data loaded successfully")
        return df
    except FileNotFoundError:
        print('File does not exit')
    except Exception as e:
        print(f"error occured {e}")
        return None

    