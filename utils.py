def burn_down_forecast(df, total_scope, scope_type):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # Calculate average throughput per month
    if scope_type == 'throughput':
      throughput = df.set_index('ResolutionDate').resample('M').agg({'IssueKey': 'count'}).mean().values[0] 
    elif scope_type == 'story_points': 
      throughput = df.set_index('ResolutionDate').resample('M').agg({'StoryPoints': 'sum'}).mean().values[0]
    else:
      return("scope_type must be throughput' or 'story_points'")

    # Calculate the time to completion
    time_to_completion = total_scope / throughput

    # Calculate intermediate points
    intervals = 10
    time_steps = np.linspace(0, time_to_completion, intervals)
    work_remaining = total_scope - (np.array(time_steps) * throughput)

    # Plotting
    fig, ax = plt.subplots()
    _ = ax.plot(time_steps, work_remaining, label='Forecast', linewidth=3)
    _ = ax.scatter(time_steps, work_remaining, color='red')  # Plot intermediate points
    _ = ax.set_title(f'Burn-Down Chart for {scope_type.title()}')
    _ = ax.set_xlabel('Months from Start')
    _ = ax.set_ylabel('Remaining Work Left')
    _ = ax.set_xticks(time_steps)
    _ = ax.set_xticklabels([f'{round(step)}' for step in time_steps])
    _ = ax.grid(True)
    _ = ax.legend()
    

def monte_carlo_simulation(df, total_scope=None, scope_type='story_points', scope_range=None, delivery_pace_factor=1.0, trials=1000, percentile=85):
    """
    Runs a Monte Carlo Simulation to estimate completion times for a given project based on historical data.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing historical project data.
    - total_scope (float): Total scope of the project. If None, the range is used.
    - scope_type (str): Type of scope, either 'story_points' or 'throughput'.
    - scope_range (tuple): Range of possible total scope as (min, max). Default is None.
    - delivery_pace_factor (float): Factor to adjust the average delivery pace and its variation.
    - trials (int): Number of Monte Carlo simulation trials to run.
    - sample_size (int): Size of the sample for each trial.
    - percentile (float): Percentile value for completion time estimation.

    Returns:
    - simulations (list): List of tuples, each containing completion time and corresponding work remaining at each step.
    """

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import random
    
    if total_scope is not None and scope_range is None:
        # Assuming a range of 120% to 140% of the original scope
        scope_range = (total_scope * 1.2, total_scope * 1.4)

    # Convert ResolutionDate to datetime and group by time unit (e.g., month)
    df['ResolutionDate'] = pd.to_datetime(df['ResolutionDate'])
    df['TimeUnit'] = df['ResolutionDate'].dt.to_period('M')  # Grouping by month, can be changed as needed

    # Calculate historical delivery pace
    if scope_type == 'story_points':
        delivery_pace_data = df.groupby('TimeUnit')['StoryPoints'].sum()
    else:  # scope_type == 'throughput'
        delivery_pace_data = df.groupby('TimeUnit').size()

    avg_delivery_pace = delivery_pace_data.mean() * delivery_pace_factor
    std_dev_delivery_pace = delivery_pace_data.std() * delivery_pace_factor

    # Define delivery pace range
    delivery_pace_range = (max(avg_delivery_pace - std_dev_delivery_pace, 0),
                           avg_delivery_pace + std_dev_delivery_pace)

    # Monte Carlo Simulation
    simulations = []
    for _ in range(trials):
        total_work = random.uniform(*scope_range) if scope_range else total_scope
        time_passed = 0
        simulation = []
        work_remaining = total_work
        while work_remaining > 0:
            delivery_pace = random.uniform(*delivery_pace_range)
            work_remaining -= delivery_pace
            time_passed += 1
            simulation.append(round(max(work_remaining, 0),1))
        simulations.append((time_passed-1, simulation))

    return simulations


def monte_carlo_burndown(simulation_data, scope_type='story_points', sample_size=100):
    """
    Plots a burn-down forecast chart with variability from Monte Carlo Simulation data with shades of blue.

    Parameters:
    - simulation_data (list): List of tuples, each containing completion time and corresponding work remaining at each step.
    - scope_type (str): Type of scope ('story_points' or 'issues').
    - sample_size (int): Number of sample lines to plot.
    - percentile (int): Percentile for confidence interval calculation. Used to determine the completion time for the burn-down chart.

    Returns:
    - Matplotlib figure and axes of the plotted burn-down chart.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import random

    # Plotting
    fig, ax = plt.subplots();
    for _, sim in random.sample(simulation_data, min(sample_size, len(simulation_data))):
        _ = ax.plot(range(len(sim)), sim, alpha=0.5, color='skyblue')
    _ = ax.set_title(f'Monte Carlo Burn-Down Forecasts')
    _ = ax.set_xlabel('Months')
    _ = ax.set_ylabel(f'Work Remaining ({scope_type})')


def monte_carlo_histogram(simulation_data, percentile=85):
    """
    Plots a histogram with probability for completion times using a Monte Carlo Simulation with shades of blue.

    Parameters:
    - simulation_data (list): List of tuples, each containing completion time and corresponding work remaining at each step.
    - percentile (int): Percentile for confidence interval calculation.

    Returns:
    - Matplotlib figure and axes of the plotted histogram.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import random

    # Calculate Completion Times and Percentile
    completion_times = [sim[0] for sim in simulation_data]
    completion_time_percentile = np.percentile(completion_times, percentile)
    completion_times = pd.Series(completion_times).value_counts()


    # Histogram of Completion Times
    fig2, ax2 = plt.subplots()
    _ = ax2.bar(completion_times.index, completion_times.values, color='skyblue')
    _ = ax2.axvline(completion_time_percentile, color='red', linestyle='dashed', label=f'{percentile}th Percentile')
    _ = ax2.set_title('Distribution of Completion Times')
    _ = ax2.set_xlabel('Time to Complete (Months)')
    _ = ax2.set_ylabel('Frequency')
    _ = ax2.legend()



def calculate_completion_time(simulation_data, percentile=85, start_date = None):
  import numpy as np
  import pandas as pd

  # Calculate Completion Times and Percentile
  completion_times = [sim[0] for sim in simulation_data]
  completion_time_percentile = np.percentile(completion_times, percentile)

  if start_date:
    completion_time = pd.to_datetime(start_date) + pd.DateOffset(months=completion_time_percentile)
    return completion_time.strftime("%Y-%m-%d")
  else:
    return completion_time_percentile
