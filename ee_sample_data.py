import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Create timestamp range for 6 months of hourly data
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 6, 30)
dates = pd.date_range(start=start_date, end=end_date, freq='H')

# Function to generate realistic power system data with daily and seasonal patterns
def generate_substation_data(station_id, base_load, variation):
    """
    Generate power system data for a single substation
    
    Parameters:
    station_id (str): Identifier for the substation
    base_load (float): Base load in MW
    variation (float): Standard deviation for random variations
    
    Returns:
    pandas.DataFrame: DataFrame containing power system measurements
    """
    data = []
    
    for timestamp in dates:
        # Add daily pattern (peak during day, low at night)
        hour = timestamp.hour
        daily_factor = np.sin(np.pi * (hour - 4) / 24) * 0.3 + 1
        
        # Add seasonal pattern (higher in summer)
        day_of_year = timestamp.dayofyear
        seasonal_factor = np.sin(np.pi * (day_of_year - 15) / 365) * 0.2 + 1
        
        # Base load with random variation
        load = base_load * daily_factor * seasonal_factor
        load += np.random.normal(0, variation)
        
        # Generate voltage (normally distributed around 1.0 pu)
        voltage = np.random.normal(1.0, 0.02)
        
        # Calculate current based on power and voltage
        current = load / voltage
        
        # Power factor (typically between 0.8 and 0.95)
        power_factor = np.random.uniform(0.8, 0.95)
        
        # Calculate real and reactive power
        real_power = load
        apparent_power = real_power / power_factor
        reactive_power = np.sqrt(apparent_power**2 - real_power**2)
        
        data.append({
            'timestamp': timestamp,
            'station_id': station_id,
            'voltage_pu': voltage,
            'current_pu': current,
            'real_power_mw': real_power,
            'reactive_power_mvar': reactive_power,
            'power_factor': power_factor
        })
    
    return pd.DataFrame(data)

# Generate data for three substations with different base loads
substation1 = generate_substation_data('SUB_001', base_load=80, variation=5)
substation2 = generate_substation_data('SUB_002', base_load=120, variation=8)
substation3 = generate_substation_data('SUB_003', base_load=60, variation=4)

# Combine all substation data
all_data = pd.concat([substation1, substation2, substation3], ignore_index=True)

# Add some anomalies for fault analysis
# Voltage sag in substation 1 on March 15
sag_mask = (all_data['station_id'] == 'SUB_001') & (all_data['timestamp'].dt.date == datetime(2024, 3, 15).date())
all_data.loc[sag_mask, 'voltage_pu'] *= 0.8

# Power factor issues in substation 2 during May
pf_mask = (all_data['station_id'] == 'SUB_002') & (all_data['timestamp'].dt.month == 5)
all_data.loc[pf_mask, 'power_factor'] *= 0.9

# Save to CSV
all_data.to_csv('power_system_data.csv', index=False)

print("Data generation complete!")
print(f"Generated {len(all_data)} records")
print("\nSample of the data:")
print(all_data.head())