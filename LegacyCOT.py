#!/usr/bin/env python
# coding: utf-8

# # LEGACY COT REPORT

# In[27]:


import pandas as pd
from sodapy import Socrata
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def fetch_cftc_data(x, start = None, end = None):
    # Unauthenticated client only works with public data sets.
    # Note the domain and resource ID are separated in the URL.
    client = Socrata("publicreporting.cftc.gov", None)

    # Specify the resource ID or path in the get method.
    # Add any query parameters as needed, including the date range.
    results = client.get(
        "6dca-aqww",
        limit=50000,
        commodity=x
    )

    # Convert to pandas DataFrame
    results_df = pd.DataFrame.from_records(results)

    # Filter data based on the date range
    results_df['report_date'] = pd.to_datetime(results_df['report_date_as_yyyy_mm_dd'])
    inp = input("Do you want to select a market? Yes or No ")

    if inp.lower() == 'yes':
        mercati = set(results_df['contract_market_name'])
        print(mercati)
        market = input('Which market?: ')
        if not start and not end:
            mask = (results_df['contract_market_name'] == market)
            filtered_df = results_df[mask]
        elif start and end:
            date_mask = (results_df['report_date'] >= start) & (results_df['report_date'] <= end) & (results_df['contract_market_name'] == market)
            filtered_df = results_df[date_mask]
        elif start and not end:
            date_mask = (results_df['report_date'] >= start) & (results_df['contract_market_name'] == market)
            filtered_df = results_df[date_mask]
        elif not start and end:
            date_mask = (results_df['report_date'] <= end) & (results_df['contract_market_name'] == market)
            filtered_df = results_df[date_mask]
        
    elif inp.lower() == 'no':
        if not start and not end:
            filtered_df = results_df
        elif start and end:
            date_mask = (results_df['report_date'] >= start) & (results_df['report_date'] <= end)
            filtered_df = results_df[date_mask]
        elif start and not end:
            date_mask = (results_df['report_date'] >= start)
            filtered_df = results_df[date_mask]
        elif not start and end:
            date_mask = (results_df['report_date'] <= end)
            filtered_df = results_df[date_mask]
        
    else:
        print("Invalid input. Please enter 'yes' or 'no'.")


    # Select only the desired columns
    selected_columns = [
        'report_date',
        'commodity_name',
        'contract_market_name',
        'open_interest_all',
        'noncomm_positions_long_all',
        'noncomm_positions_short_all',
        'comm_positions_long_all',
        'comm_positions_short_all'
    ]
    filtered_df = filtered_df[selected_columns]

    # Sort DataFrame based on the "report_date" column
    filtered_df = filtered_df.sort_values(by='report_date')
    filtered_df = filtered_df.set_index('report_date')

    # Convert columns to numeric types
    filtered_df['noncomm_positions_long_all'] = pd.to_numeric(filtered_df['noncomm_positions_long_all'], errors='coerce')
    filtered_df['noncomm_positions_short_all'] = pd.to_numeric(filtered_df['noncomm_positions_short_all'], errors='coerce')
    filtered_df['comm_positions_long_all'] = pd.to_numeric(filtered_df['comm_positions_long_all'], errors='coerce')
    filtered_df['comm_positions_short_all'] = pd.to_numeric(filtered_df['comm_positions_short_all'], errors='coerce')
    filtered_df['open_interest_all'] = pd.to_numeric(filtered_df['open_interest_all'], errors = 'coerce')

    # Create the new columns
    filtered_df['noncomm_net'] = filtered_df['noncomm_positions_long_all'] - filtered_df['noncomm_positions_short_all']
    filtered_df['comm_net'] = filtered_df['comm_positions_long_all'] - filtered_df['comm_positions_short_all']

    # Calculate z-scores
    calculate_zscore(filtered_df, 'open_interest_all')
    calculate_zscore(filtered_df, 'noncomm_net')
    calculate_zscore(filtered_df, 'comm_net')

    return filtered_df


def calculate_zscore(df, column_name, window=52):
    column = pd.to_numeric(df[column_name], errors='coerce')
    rolling_mean = column.rolling(window=window).mean()
    rolling_std = column.rolling(window=window).std()
    zscore_column_name = f'zscore_{column_name}'
    df[zscore_column_name] = (column - rolling_mean) / rolling_std




# # SINGLE PLOT

# In[28]:


def plot_data(df, columns=['noncomm_net', 'comm_net'], open_interest=False, zscore=False):
    # Create a figure with two y-axes
    if not open_interest:
        plt.figure(figsize=(10, 6))
    
        # Calculate the number of columns and the total width needed
        num_columns = len(columns)
        bar_width = 1
        total_width = bar_width * num_columns

        # Adjust the x-position for each set of bars
        x_positions = range(len(df.index))

        for i, column in enumerate(columns):
            x_positions_i = [pos + i * bar_width - total_width / 2 for pos in x_positions]
            plt.bar(x_positions_i, df[column], width=bar_width, label=column)

        # Add a dashed line at 0
        plt.axhline(0, color='black', linestyle='--', linewidth=1, label='Zero')

        plt.title('CFTC Data Visualization')
        plt.xlabel('Date')
        plt.ylabel('Values')
        plt.legend(loc='upper left')
        plt.show()
       
    
    if open_interest:
        fig, ax1 = plt.subplots(figsize=(15, 6))

        # Plot the specified columns on the primary y-axis
        for column in columns:
            ax1.plot(df.index, df[column], label=column)

        # Set labels and legend for the primary y-axis
        ax1.set_title('CFTC Data Visualization')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Net position')
        ax1.legend(loc='upper left')

        # Add a dashed line at 0 on the primary y-axis
        ax1.axhline(0, color='black', linestyle='--', linewidth=1, label='Zero')
        # Create a secondary y-axis for 'open_interest_all'
        ax2 = ax1.twinx()
        ax2.plot(df.index, df['open_interest_all'], color='grey', linestyle='--', label='Open Interest')

        # Set labels and legend for the secondary y-axis
        ax2.set_ylabel('Open Interest', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.legend(loc='upper right')


    plt.show()
    if zscore:
        fig = go.Figure(data=[go.Table(header=dict(values=['Non Commercial Net', 'ZScore Non Commercial', 'Commercial Net', 'ZScore Commercial','Open Interest', 'ZScore Open Interest']),
                         cells=dict(values=([round(df['noncomm_net'][-1],3)],[round(df['zscore_noncomm_net'][-1],3)], [round(df['comm_net'][-1],3)],
                                            [round(df['zscore_comm_net'][-1],3)], round(df['open_interest_all'][-1],3), round(df['zscore_open_interest_all'][-1],3)))
                                      )])
        fig.update_layout(height=100, margin=dict(t=20,b=10))
        fig.add_annotation(x=0.5, y=-0.2,  # Adjust the y coordinate for the annotation
                   text="The Z-Score is computed as the distance, in terms of standard deviation, of the last available data from the mean of the last 52 weeks",
                   showarrow=False,
                   font=dict(size=10))
        fig.show()


# # SUBPLOTS

# In[29]:


def plot_subplots(df, columns=['noncomm_net', 'comm_net'], open_interest=False, zscore=False):
    if not open_interest:
        plt.figure(figsize=(10, 6))
    
        # Calculate the number of columns and the total width needed
        num_columns = len(columns)
        bar_width = 0.4
        total_width = bar_width * num_columns

        # Adjust the x-position for each set of bars
        x_positions = range(len(df.index))

        for i, column in enumerate(columns):
            x_positions_i = [pos + i * bar_width - total_width / 2 for pos in x_positions]
            plt.bar(x_positions_i, df[column], width=bar_width, label=column)

        # Add a dashed line at 0
        plt.axhline(0, color='black', linestyle='--', linewidth=1, label='Zero')

        plt.title('CFTC Data Visualization')
        plt.xlabel('Date')
        plt.ylabel('Values')
        plt.legend(loc='upper left')
        plt.show()


    elif open_interest:
    # Create a figure with multiple subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

        # Plot the specified columns on the first subplot
        for column in columns:
            ax1.plot(df.index, df[column], label=column)

        # Set labels and legend for the first subplot
        ax1.set_ylabel('Values')
        ax1.legend(loc='upper left')

        # Add a dashed line at 0 on the first subplot
        ax1.axhline(0, color='black', linestyle='--', linewidth=1, label='Zero')

        # Plot 'open_interest_all' on the second subplot
        ax2.plot(df.index, df['open_interest_all'], color='grey', linestyle='--', label='Open Interest')

        # Set labels and legend for the second subplot
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Open Interest', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.legend(loc='upper right')

        plt.show()
    if zscore:
    
        fig = go.Figure(data=[go.Table(header=dict(values=['Non Commercial Net', 'ZScore Non Commercial', 'Commercial Net', 'ZScore Commercial','Open Interest', 'ZScore Open Interest']),
                         cells=dict(values=([round(df['noncomm_net'][-1],3)],[round(df['zscore_noncomm_net'][-1],3)], [round(df['comm_net'][-1],3)],
                                            [round(df['zscore_comm_net'][-1],3)], round(df['open_interest_all'][-1],3), round(df['zscore_open_interest_all'][-1],3)))
                                      )])
        fig.update_layout(height=100, margin=dict(t=20,b=10))
        fig.add_annotation(x=0.5, y=-0.2,  # Adjust the y coordinate for the annotation
                   text="The Z-Score is computed as the distance, in terms of standard deviation, of the last available data from the mean of the last 52 weeks",
                   showarrow=False,
                   font=dict(size=10))
        fig.show()


# In[ ]:




