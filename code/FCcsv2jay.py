
# Convert FC csv to jay

import pandas as pd
import datatable as dt

csv_data = pd.read_csv("/data/project/impulsivity/prediction_simulations/input/Schaefer400x17_WM+CSF+GS_hcpaging_695.csv")
# check some stuff
csv_data.keys()
csv_data.head()

DT = dt.Frame(csv_data)

DT.to_jay("/data/project/impulsivity/prediction_simulations/input/Schaefer400x17_WM+CSF+GS_hcpaging_695.jay")


# Load again to see diff in speed and check df looks the same after converting to pandas
DT_new = dt.fread("/data/project/impulsivity/prediction_simulations/input/Schaefer400x17_WM+CSF+GS_hcpaging_695.jay")

jay_data = DT_new.to_pandas()
jay_data.keys()
jay_data.head()

