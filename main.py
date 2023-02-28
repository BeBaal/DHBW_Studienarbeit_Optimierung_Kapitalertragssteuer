"""This module is part of my science project at the DHBW CAS. Primary
    goal is to calculate a timeseries of investments into exchange
    traded index funds. Each december the program decides to sell stocks
    according to the yearly tax free capital gain. Results are compared
    to all time intervals in the dataset for the amount of the
    investment period.
    """

# Import Statements
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


# Declaration of class variables
DEBUGGING = False
CUT_INDEX_FOR_SAME_INTERVAL = True
LIST_OF_MONTHLY_CONTRIBUTIONS = [120, 180, 240, 300, 360]
LIST_OF_INVESTMENT_DURATIONS = [120, 180, 240, 300, 360]
LIST_OF_TRANSACTION_COST = [0.01, 0.02]
TAX_FREE_CAPITAL_GAIN = 801
FILE_PATH_FOR_IMAGES = r'./Results/Figures/'
CM = 1/2.54                  # conversion rate centimeters to inches


def main():
    """
    This is the main method of the program.

    Here most of the function calls are made as well as the start of the
    iteration.
    """
    # Reset result folders
    delete_results()

    # Sets some class parameters differently for debugging
    if DEBUGGING:
        set_debugging()

    # Sets some general matplotlib settings
    matplotlib_settings()

    # Create result df for summary data
    results = pd.DataFrame()

    # get the start time for run time calculation
    start_time = time.time()

    # Create array of index dataframes for looping
    initial_index_time_series = load_data()

    # Plot the initial index performance
    plot_index_funds(initial_index_time_series)

    # Deep iteration over a few complicated functions
    results = iteration_over_monthly_investment_sum(
        initial_index_time_series, results)

    # Feature Engineering with the result table
    results = calc_results_on_results(results)

    export_results_df_to_csv(results)

    # Export some descriptive information regarding the result
    generate_descriptive_info_results(results)

    # plot the result table
    plot_distribution_of_all_results(results)
    plot_distribution_of_specific_result(results)
    plot_scatter(results)

    # get end time
    end_time = time.time()

    # calculate calculation time and print to console
    elapsed_time = end_time - start_time
    print('Execution time:', elapsed_time, 'seconds')


def delete_results():
    """For convenience and error pruning this function deletes the old
    results files from the export folders.
    """

    # declare some paths for deletion
    paths = [r'./Results/Figures/',
             r'./Results/tables/',
             r'./Calculation_Files/',
             r'./Logfiles/'
             ]

    # loop over paths and delete files if possible
    for path in paths:
        for file_name in os.listdir(path):
            # construct full file path
            file = path + file_name
            if os.path.isfile(file):
                os.remove(file)


def matplotlib_settings():
    """This function sets the matplotlib global export settings for Word.
    """
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "12"
    plt.rcParams["figure.figsize"] = (20*CM, 5*CM)


def iteration_over_monthly_investment_sum(
        initial_index_time_series,
        results):
    """This function iterates over the list of monthly investment sums
        and calls the next iteration

    Args:
        initial_index_time_series (array of pandas dataframes):
        dataframe with the index time series
        results (pandas dataframe): Empty dataframe

    Returns:
        pandas dataframe: Result table with observations
    """

    for monthly_investment_sum in LIST_OF_MONTHLY_CONTRIBUTIONS:

        results = iteration_over_transaction_cost(
            initial_index_time_series,
            results,
            monthly_investment_sum)

    return results


def iteration_over_transaction_cost(
        initial_index_time_series,
        results,
        monthly_investment_sum):
    """This function iterates over the list of transaction costs and
        calls the next iteration.

    Args:
        initial_index_time_series (array of pandas dataframes):
            dataframe with the index time series
        results (pandas dataframe): Dataframe of simulation results
        monthly_investment_sum (int): How much the monthly investment is

    Returns:
        pandas dataframe: Result table with observations
    """

    for transaction_cost in LIST_OF_TRANSACTION_COST:

        results = iteration_over_investment_duration(
            initial_index_time_series,
            results,
            monthly_investment_sum,
            transaction_cost)

    return results


def iteration_over_investment_duration(
        initial_index_time_series,
        results,
        monthly_investment_sum,
        transaction_cost):
    """This function iterates over the list of investment durations and
        calls the next iteration.

    Args:
        initial_index_time_series (array of pandas dataframes):
            dataframe with the index time series
        results (pandas dataframe): Dataframe of simulation results
        monthly_investment_sum (int): How much the monthly investment is
        transaction_cost (float): How much the transaction cost is
            ex. 0.01

    Returns:
        pandas dataframe: Result table with observations
    """
    for duration in LIST_OF_INVESTMENT_DURATIONS:

        results = iteration_over_index_funds(
            initial_index_time_series,
            results,
            monthly_investment_sum,
            transaction_cost,
            duration)

    return results


def iteration_over_index_funds(
        initial_index_time_series,
        results,
        monthly_investment_sum,
        transaction_cost,
        duration):
    """This function iterates over the array of index funds and
        calls the next iteration.

    Args:
        initial_index_time_series (array of pandas dataframes):
            dataframe with the index time series
        results (pandas dataframe): Dataframe of simulation results
        monthly_investment_sum (int): How much the monthly investment is
        transaction_cost (float): How much the transaction cost is
            ex. 0.01
        duration (int): How long to invest for

    Returns:
        pandas dataframe: Dataframe of simulation results
    """

    # loop over all index time series
    for index_counter, _ in enumerate(initial_index_time_series):

        results = iteration_over_time_slices(
            initial_index_time_series,
            results,
            index_counter,
            monthly_investment_sum,
            transaction_cost,
            duration)

    return results


def iteration_over_time_slices(
        initial_index_time_series,
        results,
        index_counter,
        monthly_investment_sum,
        transaction_cost,
        duration):
    """This function iterates over all possible time slices in the data
        and calls the preparation function as well as the calculation
        function. Also the concatenation of the results is done here.

    Args:
        initial_index_time_series (array of pandas dataframes):
            dataframe with the index time series
        results (pandas dataframe): Dataframe of simulation results
        index_counter (int): Number of index in the dataframe array
        monthly_investment_sum (int): How much the monthly investment is
        transaction_cost (float): How much the transaction cost is
            ex. 0.01
        duration (int): How long to invest for

    Returns:
        pandas dataframe: Dataframe of simulation results
    """

    # Initialize dataframe with copy
    df_calc = initial_index_time_series[index_counter].copy()

    # set some initial values for the first time slice
    upper_bound_month = duration  # How many months are in dataset
    lower_bound_month = 0        # Start with first period

    # The maximum lower bound is the length of time series minus
    # investment duration
    max_lower_bound_month = (
        len(df_calc.index)
        - duration)

    # Iterate over all time intervals in the dataset
    while lower_bound_month <= max_lower_bound_month:

        # Reset dataframe for each time slice
        df_calc = initial_index_time_series[index_counter].copy()

        # Do some preparation for the calculation
        df_calc = setup_calculation(
            df_calc,
            lower_bound_month,
            upper_bound_month,
            monthly_investment_sum
        )

        # Do the calculation and return new observations
        new_results = table_calculation(
            df_calc,
            duration,
            index_counter,
            monthly_investment_sum,
            transaction_cost)

        # Concat old results with new results
        results = pd.concat([results, new_results])

        # iterate over all available time slices in the data
        upper_bound_month += 1
        lower_bound_month += 1

    return results


def set_debugging():
    """This function sets some class variables for debugging.
    """

    # Using the global keyword is not regarded as best practice but in
    # this case it seems fine for setting class parameters for debugging.
    global DEBUGGING
    global LIST_OF_MONTHLY_CONTRIBUTIONS
    global LIST_OF_INVESTMENT_DURATIONS
    global LIST_OF_TRANSACTION_COST
    global TAX_FREE_CAPITAL_GAIN
    DEBUGGING = True
    LIST_OF_MONTHLY_CONTRIBUTIONS = [200]
    LIST_OF_INVESTMENT_DURATIONS = [12]
    LIST_OF_TRANSACTION_COST = [0.01]
    TAX_FREE_CAPITAL_GAIN = 801


def load_data():
    """This function loads the index fund timeseries and gives the result
    back in a dataframe. There are four index timeseries. One for
    debugging as well as three different indexes. In the finished study
    MSCI ACWI was not used due to the long runtime of the program.

    Returns:
        pandas dataframe:  Initial index fund time series
    """
    # Load Excel file
    xls = pd.ExcelFile(
        '../daten/daten_Studienarbeit_Optimierung_Kapitalertragssteuer.xlsx')

    sheets = []

    # Load Excel sheets
    if DEBUGGING is True:
        sheets.append(pd.read_excel(xls, 'Testdata', header=0))
    else:
        sheets.append(pd.read_excel(xls, 'MSCI_World', header=0))
        sheets.append(pd.read_excel(xls, 'MSCI_EM', header=0))
        # sheets.append(pd.read_excel(xls, 'MSCI_ACWI', header=0))

    # Create array of dataframes
    initial_df_array = []

    # List comprehension for loading the dataframes
    initial_df_array = [pd.DataFrame(data=sheet, columns=['date', 'value_in_usd'])
                        for sheet in sheets]

    # Change date format to datetime
    initial_df_array = [df.assign(
        date=pd.to_datetime(df['date'],
                            format='%b %d, %Y'))
                        for df in initial_df_array]

    # Calculate additional features for further use
    initial_df_array = [
        df.assign(
            year=df['date'].dt.year,
            month=df['date'].dt.month,
            day=df['date'].dt.day)
        for df in initial_df_array]

    # The indexes have different start dates. If the option is set this
    # code cuts them down to the same length.
    if CUT_INDEX_FOR_SAME_INTERVAL:
        # Filter data to the smallest denominator of the time series
        initial_df_array = [
            df[~(df['date'] < '1987-12-01')]
            for df in initial_df_array]

    return initial_df_array


def end_of_capital_year(
        df_calc,
        current_period_index,
        transaction_cost):
    """This is the most complex function in the program where most of the
        logic is handled. The function is called in each december and
        calculates the appropriate calculation table.

    Args:
        df_calc (pandas dataframe): Dataframe in which the calculation
            is done
        current_period_index (int): The current period in the dataframe
        transaction_cost (float): How much is the transaction cost
            ex. 0.01
    """

    # setting some initial values for the calculation
    tax_free_capital_gain = TAX_FREE_CAPITAL_GAIN
    transaction_sum = 0.0
    former_period_index = 0

    # Gets the current value of the index
    current_value_of_index = df_calc.value_in_usd.values[current_period_index]

    # Look back in table for capital gains
    while former_period_index < current_period_index:

        # Calculate still to be taxed_profit on a row basis
        # Current Value of fund in relation to buy in value
        df_calc.not_taxed_change.values[former_period_index] = (
            df_calc.not_taxed_investment.values[former_period_index]
            * current_value_of_index
            / df_calc.value_in_usd.values[former_period_index]
            - df_calc.not_taxed_investment.values[former_period_index]
        )

        # Skip zero profit and zero allowance
        if (df_calc.not_taxed_change.values[former_period_index] == 0
                or tax_free_capital_gain == 0):
            former_period_index += 1
            continue

        # If the profit is negative do not sell
        if df_calc.not_taxed_change.values[former_period_index] < 0:
            # jump out of while loop
            break

        # Profit must be positive
        # Does the tax free allowance allow selling of all profits?
        if (tax_free_capital_gain
                > df_calc.not_taxed_change.values[former_period_index]):

            # 1 Save profit for statistics
            df_calc.taxed_profit.values[former_period_index] = (
                df_calc.taxed_profit.values[former_period_index]
                + df_calc.not_taxed_change.values[former_period_index]
            )

            # 2 Add order value to transaction sum
            transaction_sum = (
                transaction_sum
                + df_calc.not_taxed_change.values[former_period_index]
                + df_calc.not_taxed_investment.values[former_period_index])

            # 3 Detract the profit from tax free allowance
            tax_free_capital_gain = (
                tax_free_capital_gain
                - df_calc.not_taxed_change.values[former_period_index])

            # 4 All profits of the period were sold
            df_calc.not_taxed_change.values[former_period_index] = 0

            # 5 The investment in this period is sold
            df_calc.not_taxed_investment.values[former_period_index] = 0

            former_period_index += 1
            continue

        else:
            # If the rest allowance is smaller then the profit, the
            # buy order needs to be split into two parts because it can
            # not be realized completely .

            # 1 Add order value to transaction sum. Both the investment
            # as well as the change need to be added to transaction sum.
            transaction_sum = (
                transaction_sum
                + df_calc.not_taxed_investment.values[former_period_index]
                * (1 - tax_free_capital_gain
                   / df_calc.not_taxed_change.values[former_period_index])
                + df_calc.not_taxed_change.values[former_period_index]
                * (1 - tax_free_capital_gain
                   / df_calc.not_taxed_change.values[former_period_index])
            )

            # 2 Calculate the rest buy order for future taxation. The
            # calculation is limited to the rest tax free allowance.
            # This value needs to be calculated first
            df_calc.not_taxed_investment.values[former_period_index] = (
                df_calc.not_taxed_investment.values[former_period_index]
                * tax_free_capital_gain
                / df_calc.not_taxed_change.values[former_period_index]
            )

            # 3 Calculate for debugging not needed for calculation
            df_calc.not_taxed_change.values[former_period_index] = (
                df_calc.not_taxed_change.values[former_period_index]
                - tax_free_capital_gain)

            # 4 Keep track of the partial taxed profits for statistics
            df_calc.taxed_profit.values[former_period_index] = (
                df_calc.taxed_profit.values[former_period_index]
                + tax_free_capital_gain)

            # 5 Code only runs when allowance is smaller then profits.
            # Therefore the allowance is zero
            tax_free_capital_gain = 0

            former_period_index += 1
            continue

    # Some debugging code. These conditions should not be possible.
    if transaction_sum < 0:
        print("Transaction Sum is negative")

    if tax_free_capital_gain < 0:
        print("Tax Free Capital gain is negative")

    # Deduct the transaction cost from the value of investment from
    # previous periods
    df_calc.transaction_cost.values[current_period_index] = (
        df_calc.transaction_cost.values[current_period_index]
        + (transaction_sum * transaction_cost))

    # Set tax free capital gain and remaining allowance after calculation
    df_calc.remaining_tax_free_capital_gain.values[current_period_index] = (
        tax_free_capital_gain)
    df_calc.realized_tax_free_capital_gain.values[current_period_index] = (
        TAX_FREE_CAPITAL_GAIN - tax_free_capital_gain)

    # Save the transaction sum as a reinvestment
    df_calc.reinvestment.values[current_period_index] = (
        df_calc.reinvestment.values[current_period_index]
        + transaction_sum)

    # Create a new buy order from the reinvestment so it can be taxed in
    # the future
    df_calc.not_taxed_investment.values[current_period_index] = (
        df_calc.not_taxed_investment.values[current_period_index]
        + transaction_sum
        - (transaction_sum * transaction_cost)
    )


def table_calculation(
        df_calc,
        duration,
        index_counter,
        monthly_investment_sum,
        transaction_cost):
    """
    This method loops over all periods in the dataframe and calls the
    end of capital year function

    Args:
        df_calc (dataframe): Table on which the calculation is done
        duration (integer): How long you invest for in months
        index_counter (integer): Which index is used
        tax_free_capital_gain (integer): Tax free capital gain
        monthly_investment_sum (int): How much is invested monthly

    Returns:
        pandas dataframe: Dataframe of simulation results
    """

    # method variables
    df_calc.reset_index(drop=True)

    # Loop over each december period in table with list comprehension
    # In the first period there can be no profits
    # Assignment to nothing is wanted, function works inplace
    [end_of_capital_year(df_calc,
                         index,
                         transaction_cost)
     for index, month in zip(df_calc.index.values, df_calc.month.values)
     if month == 12 and index != 0
     ]

    # Calculate results on the calculation dataframe
    results = calc_results_on_df_calc(
        df_calc,
        index_counter,
        duration,
        monthly_investment_sum,
        transaction_cost)

    # Save the calculation file when debugging
    if DEBUGGING is True:
        export_calc_df_to_csv(
            df_calc,
            index_counter)

    return results


def setup_calculation(
        df_calc,
        lower_bound_month,
        upper_bound_month,
        monthly_investment):
    """
    This method setups the used dfs.

    A few values are calculated, columns calculated and NaN values
    cleaned up. Also it cuts the dataframe to the proper period
    regarding to lower and upper bound of the current analysis.

    Args:
        df_calc (pandas dataframe):  Initial index fund time series
        lower_bound_month (int): start of evaluation time series
        upper_bound_month (int): end of evaluation time series
        monthly_investment (int): how much you invest monthly

    Returns:
        pandas dataframe: Cut down and cleaned dataframe
    """

    # delete not necessary lines from dataframes with reference to months
    df_calc = df_calc.iloc[lower_bound_month:upper_bound_month].copy()

    # Calculate percentage changes in new column
    df_calc['%-change'] = df_calc['value_in_usd'].pct_change() * 100
    # Fill NaN rows with zeroes
    df_calc['%-change'].fillna(value=0, inplace=True)
    df_calc['not_taxed_investment'] = float(monthly_investment)
    df_calc['not_taxed_change'] = float(0)
    df_calc['taxed_profit'] = float(0)
    df_calc['reinvestment'] = float(0)
    df_calc['transaction_cost'] = float(0)
    df_calc['remaining_tax_free_capital_gain'] = np.nan
    df_calc['realized_tax_free_capital_gain'] = np.nan

    # Reset index of dataframe
    df_calc.reset_index(drop=True, inplace=True)

    return df_calc


def generate_descriptive_info_results(
        inbound_results):
    """This function generates descriptive information on the result and
        saves them to csv files

    Args:
        inbound_results (pandas dataframe): All results of the simulation
    """

    # Export all simulations
    inbound_results.describe().to_csv(
        './Results/tables/Result_Descriptive_Info.CSV'
    )

    # Export only specific simulations
    results = inbound_results.loc[(
        inbound_results['duration_in_months'] == 240)
        & (inbound_results['monthly_contribution'] == 240)
        & (inbound_results['transaction_cost'] == 0.01)].copy()

    results.describe().to_csv(
        './Results/tables/Result_Descriptive_Info_Sample.CSV'
    )


def plot_index_funds(initial_index_time_series):
    """This function plots the index funds performance.
    """

    # do not change the initial_index_time_series
    index_time_series = initial_index_time_series.copy()

    # Plot indexes and set the first value to one and the rest in
    # relation to the first
    for index in index_time_series:
        index['value_in_percent'] = [element / index.value_in_usd.values[0]
                                     for element in index.value_in_usd.values]
        plt.plot(index.date, index.value_in_percent)

    # name the dataframes for plotting
    if DEBUGGING is True:
        plt.legend("Testdata")
    else:
        plt.legend([
            "MSCI_EM",
            "MSCI_World",
            # "MSCI_ACWI"
        ])

    plt.title("Index values over time")
    plt.ylabel("price")
    plt.yscale("linear")
    plt.xlabel("date")
    plt.savefig(FILE_PATH_FOR_IMAGES + "Plot_index_performance.svg")
    plt.close()


def plot_distribution_of_all_results(inbound_results):
    """This method plots the distribution of the relevant keyfigures
        for all simulations.

    Args:
        inbound_results (pandas dataframe): results of all simulations
    """

    # do not change the inbound_results
    results = inbound_results.copy()

    # drop some features which do not need to be plotted
    results.drop(['start_date',
                  'end_date',
                  'duration_in_months',
                  'end_value_index',
                  'start_value_index',
                  'stock_market_index',
                  'monthly_contribution'
                  ],
                 axis=1,
                 inplace=True)

    # plot all features
    for column in results:

        results[column].hist(bins=25,
                             color='b',
                             alpha=0.6,
                             log=False
                             )

        name_of_column = column
        name_of_column = name_of_column.replace('_', ' ')

        plt.title("Histogram " + name_of_column)
        plt.xlabel(name_of_column)
        plt.ylabel("count of observations")

        plt.savefig(FILE_PATH_FOR_IMAGES
                    + "Distribution_"
                    + column
                    + ".svg",
                    bbox_inches="tight")
        plt.close()


def plot_distribution_of_specific_result(inbound_results):
    """This method plots the distribution of the relevant keyfigures
        for specific simulations.

    Args:
        dataframe (inbound_results):  Result dataframe with observations
    """

    # filter the results to specific hyperparameter
    results = inbound_results.loc[(
        inbound_results['duration_in_months'] == 240)
        & (inbound_results['monthly_contribution'] == 240)
        & (inbound_results['transaction_cost'] == 0.01)].copy()

    # drop features which do not need to be plotted
    results.drop(['start_date',
                  'end_date',
                  'duration_in_months',
                  'end_value_index',
                  'start_value_index',
                  'stock_market_index',
                  'monthly_contribution'
                  ],
                 axis=1,
                 inplace=True)

    # create histogram of all remaining features
    for column in results:

        results[column].hist(bins=25,
                             color='b',
                             alpha=0.6,
                             log=False
                             )

        name_of_column = column
        name_of_column = name_of_column.replace('_', ' ')

        plt.title("Histogram " + name_of_column + " sample")
        plt.xlabel(name_of_column)
        plt.ylabel("count of observations")
        plt.savefig(FILE_PATH_FOR_IMAGES
                    + "Distribution_"
                    + column
                    + "_sample"
                    + ".svg",
                    bbox_inches="tight")
        plt.close()


def plot_scatter(inbound_results):
    """This method scatter plots  the relevant keyfigures for all results.

    Args:
        inbound_results (pandas dataframe): Result dataframe with observations
    """
    results = inbound_results.copy()

    # drop features which do not need to be scatter plotted
    results.drop(
        ['start_date',
         'end_date',
         'start_value_index',
         'stock_market_index'
         ],
        axis=1,
        inplace=True)

    # loop over all features and plot them
    for keyfigure_x in results:

        # comparison with itself is not necessary
        if "tax_savings_minus_cost_of_strategy_relative_to_investment_sum" == keyfigure_x:
            continue

        # do the scattering :D
        sns.scatterplot(
            data=results,
            x=results.tax_savings_minus_cost_of_strategy_relative_to_investment_sum,
            y=keyfigure_x)

        # Add x and y histograms with linear regression to plot
        sns.jointplot(
            data=results,
            x="tax_savings_minus_cost_of_strategy_relative_to_investment_sum",
            y=keyfigure_x,
            kind="reg")

        plt.title("Scatterplot relative tax difference "
                  + keyfigure_x.replace('_', ' '))
        plt.xlabel(
            "tax savings minus cost of strategy relative to investment sum")
        plt.ylabel(keyfigure_x.replace('_', ' '))
        plt.tight_layout()
        plt.savefig(
            FILE_PATH_FOR_IMAGES
            + "Scatterplot_"
            + keyfigure_x
            + ".png",)
        plt.close()


def calc_results_on_df_calc(
        df_calc,
        index_counter,
        duration,
        monthly_investment_sum,
        transaction_cost):
    """This function calculates features on the table calculation. It
        summarizes the calculation so to say.

    Args:
        df_calc (pandas dataframe): Calculation table
        index_counter (int): Which index is the analysis for
        duration (int): How long is the investment period
        monthly_investment_sum (int): What is the monthly investment
        transaction_cost (float): What is the transaction cost (ex. 0.01)

    Returns:
        pandas dataframe: One line in the result table
    """

    # create empty dataframe
    results = pd.DataFrame()

    # Some simple value setting. No comment necessary
    results.loc[0, 'stock_market_index'] = index_counter
    results['duration_in_months'] = duration

    results['start_date'] = df_calc.date.iloc[0]
    results['end_date'] = df_calc.date.iloc[-1]

    results['start_value_index'] = df_calc.value_in_usd.iloc[0]
    results['end_value_index'] = df_calc.value_in_usd.iloc[-1]

    results['monthly_contribution'] = monthly_investment_sum

    results['realized_profits'] = df_calc.taxed_profit.sum()

    results['reinvestment'] = df_calc.reinvestment.sum()

    results['transaction_cost'] = transaction_cost

    results['transaction_cost_sum'] = df_calc.transaction_cost.sum()

    results['remaining_tax_free_capital_gain'] = df_calc.remaining_tax_free_capital_gain.sum()

    results['realized_tax_free_capital_gain'] = df_calc.realized_tax_free_capital_gain.sum()

    # Count all periods with completely realized tax free allowance
    results['tax_free_capital_gain_count'] = (
        df_calc.remaining_tax_free_capital_gain.isin([0]).sum(axis=0))

    # calculate value at the end of investment with reselling
    results['investment_value_at_end'] = (
        df_calc.not_taxed_investment.values
        * df_calc.value_in_usd.values[-1]
        / df_calc.value_in_usd.values
    ).sum()

    # calculate still to be taxed profits
    results['still_to_be_taxed_value_at_end'] = (
        df_calc.not_taxed_investment.values
        * df_calc.value_in_usd.values[-1]
        / df_calc.value_in_usd.values
        - df_calc.not_taxed_investment.values
    ).sum()

    # how much would be the transaction cost worth if you reinvested them
    results['transaction_cost_reinvested_sum'] = (
        df_calc.transaction_cost.values
        * df_calc.value_in_usd.values[-1]
        / df_calc.value_in_usd.values
    ).sum()

    # calculate the worth of the monthly investments and the index
    # without any kind of reselling or strategy
    results['investment_value_at_end_without_selling'] = (
        np.full((duration), monthly_investment_sum)
        * np.full((duration), df_calc.value_in_usd.iloc[-1]
                  / df_calc.value_in_usd.values)
    ).sum()

    # calculate still to be taxed value at end without selling
    results['still_to_be_taxed_value_at_end_without_selling'] = (
        np.full((duration), monthly_investment_sum)
        * np.full((duration), df_calc.value_in_usd.iloc[-1]
                  / df_calc.value_in_usd.values)
        - np.full((duration), monthly_investment_sum)
    ).sum()  # type: ignore
    # false positive type error. Operation is allowed

    return results


def calc_results_on_results(results):
    """This function does some feature engineering on the results table.

    Args:
        results (pandas dataframe): table to feature engineer on
    """

    # how much less profit needs to be taxed
    results['still_to_be_taxed_value_at_end_difference'] = (
        results['still_to_be_taxed_value_at_end_without_selling']
        - results['still_to_be_taxed_value_at_end']
    )

    # calculate the geometric return of the index annually
    results['geometric_mean_return_of_index_in_percent_per_year'] = (
        (results['end_value_index'] - results['start_value_index'])
        / results['start_value_index']
        / results['duration_in_months']
        * 12
        * 100
    )

    # how much taxes would you need to pay for that
    results['tax_savings'] = (
        results['still_to_be_taxed_value_at_end_difference']
        * (1-0.26375)  # capital gains tax plus soli tax
    )

    # how much cost does the strategy have
    results['loss_due_to_strategy'] = (
        results['investment_value_at_end']
        - results['investment_value_at_end_without_selling'])

    # Are the tax savings enough to cover the costs?
    results['tax_savings_minus_cost_of_strategy'] = (
        results['still_to_be_taxed_value_at_end_difference']
        * (1-0.26375)
        + results['loss_due_to_strategy']
    )

    # How does that relate to the investment sum?
    results['tax_savings_minus_cost_of_strategy_relative_to_investment_sum'] = (
        results['tax_savings_minus_cost_of_strategy']
        / (results['duration_in_months']
            * results['monthly_contribution']
           )
    )

    # On how much periods the tax free capital gains allowance is
    # completely used up relative to the maximal count?
    results['tax_free_capital_gain_count_relative'] = (
        results['tax_free_capital_gain_count']
        / (results['duration_in_months']
           / 12
           - 1
           )
    )

    return results


def export_results_df_to_csv(results):
    """ This method exports the results df to a excel file.

    Args:
        df_results (pandas dataframe): Result dataframe of all calculation runs
    """

    # Export df_results to CSV file
    results.to_csv('./Results/tables/Result_Export.CSV',
                   index=True)


def export_calc_df_to_csv(df_calc, index_counter):
    """This method exports the calculation dfs to a excel file.

    Args:
        df_calc (pandas dataframe): The calculation table
        index_counter (_type_): which index is used
    """

    df_calc.sort_index(ascending=True)
    df_calc.to_csv('./Calculation_Files/Dataframe_Export_'
                   + str(index_counter)
                   + '.CSV')


# main method call
main()
