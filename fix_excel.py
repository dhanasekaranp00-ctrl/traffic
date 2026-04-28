import pandas as pd

df = pd.DataFrame({
    'Number Plate': ['TN47AX1433', 'KA01CD5678', 'MH12EF9012', 'TN01XY4321', 'AP09ZZ9999'],
    'Owner Name':   ['ANU', 'Priya Nair', 'Rahul Mehta', 'Lakshmi D', 'Suresh Bab'],
    'Phone Number': ['2323543210', '8765432109', '7654321098', '9988776655', '9123456780'],
    'Email ID':     ['anugracyp@gmail.com', 'priya@example.com', 'rahul@example.com', 'lakshmi@example.com', 'suresh@example.com'],
})

df.to_excel('vehicle_data.xlsx', index=False)
print("Done! vehicle_data.xlsx updated.")
print(df)