"""This module is part of my science project at the DHBW CAS. Primary goal is
   to calculate a timeseries of investments into exchange traded index funds.
   Each december the program decides to sell stocks according to the yearly
   tax free capital gain. Results are compared to all time intervals in the
   dataset for the amount of the investment period.
    """

# Import Statements
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Declaration of class variables
DEBUGGING = False
SELL_OPTION = [True, False]
CUT_INDEX_FOR_SAME_INTERVAL = True
LIST_OF_MONTHLY_CONTRIBUTIONS = [120, 240]
LIST_OF_INVESTMENT_DURATIONS = [240, 360]
LIST_OF_TRANSACTION_COST = [0, 0.01]  # transaction cost per sell order
TAX_FREE_CAPITAL_GAIN = 800  # tax free capital gain allowance
FILE_PATH_FOR_IMAGES = r'./Results/Figures/'  # Path for image export
CM = 1/2.54                  # centimeters in inches


def main():
    """
    This is the main method of the program.

    Here the loop logic over the df array is implemented as well as the
    different method calls.
    """

    # Create result df for summary data
    results = pd.DataFrame()

    # get the start time
    start_time = time.time()

    # Create array of dataframes for looping
    initial_index_time_series = load_data()

    plot_index_funds(initial_index_time_series)

    results = iteration_over_monthly_investment_sum(
        initial_index_time_series, results)

    results = calc_results_on_results(results)

    export_results_df_to_csv(results)
    generate_descriptive_info_results(results)
    plot_results(results)

    # get end time
    end_time = time.time()

    # calculate calculation time
    elapsed_time = end_time - start_time
    print('Execution time:', elapsed_time, 'seconds')


def iteration_over_sell_option(
        initial_index_time_series,
        results):
    """_summary_

    Args:
        initial_index_time_series (_type_): _description_
        results (_type_): _description_

    Returns:
        _type_: _description_
    """

    # ToDo iteration_over_sell_option needs to be implemented

    # for sell_option in SELL_OPTION:

    # results = iteration_over_monthly_investment_sum(
    #     initial_index_time_series,
    #     results,
    #     sell_option)

    return results


def iteration_over_monthly_investment_sum(
        initial_index_time_series,
        results):
    """_summary_

    Args:
        initial_index_time_series (_type_): _description_
        results (_type_): _description_

    Returns:
        _type_: _description_
    """

    for monthly_investment_sum in LIST_OF_MONTHLY_CONTRIBUTIONS:

        results = iteration_over_transaction_cost(
            initial_index_time_series,
            results,
            monthly_investment_sum)

    return results


def iteration_over_transaction_cost(initial_index_time_series,
                                    results,
                                    monthly_investment_sum):
    """_summary_

    Args:
        initial_index_time_series (_type_): _description_
        results (_type_): _description_
        monthly_investment_sum (_type_): _description_

    Returns:
        _type_: _description_
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
    """_summary_

    Args:
        initial_index_time_series (_type_): _description_
        results (_type_): _description_
        monthly_investment_sum (_type_): _description_
        transaction_cost (_type_): _description_

    Returns:
        _type_: _description_
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
        initial_index_time_series, results,
        monthly_investment_sum,
        transaction_cost,
        duration):
    """_summary_

    Args:
        initial_index_time_series (_type_): _description_
        results (_type_): _description_
        monthly_investment_sum (_type_): _description_
        transaction_cost (_type_): _description_
        duration (_type_): _description_

    Returns:
        _type_: _description_
    """

    index_time_series = initial_index_time_series.copy()

    # loop over all index time series
    for index_counter, _ in enumerate(index_time_series):

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
    """_summary_

    Args:
        initial_index_time_series (_type_): _description_
        results (_type_): _description_
        index_counter (_type_): _description_
        monthly_investment_sum (_type_): _description_
        transaction_cost (_type_): _description_
        duration (_type_): _description_

    Returns:
        _type_: _description_
    """

    df_calc = initial_index_time_series[index_counter].copy()

    # Start Value / how many months in dataset
    upper_bound_month = duration
    lower_bound_month = 0      # Start Value / end upper_bound minus months
    max_lower_bound_month = len(df_calc.index) - duration

    # Iterate over all time intervals in the dataset
    while lower_bound_month < max_lower_bound_month:

        # Reset df
        df_calc = initial_index_time_series[index_counter].copy()
        df_calc = setup_calculation(df_calc,
                                    lower_bound_month,
                                    upper_bound_month,
                                    monthly_investment_sum
                                    )

        new_results = table_calculation(
            df_calc,
            duration,
            index_counter,
            monthly_investment_sum,
            transaction_cost)

        results = pd.concat([results, new_results])

        upper_bound_month += 1
        lower_bound_month += 1

    return results


def set_debugging():
    """_summary_
    """
    global DEBUGGING
    global LIST_OF_MONTHLY_CONTRIBUTIONS
    global LIST_OF_INVESTMENT_DURATIONS
    global LIST_OF_TRANSACTION_COST
    DEBUGGING = True
    LIST_OF_MONTHLY_CONTRIBUTIONS = [100, 200, 300]
    LIST_OF_INVESTMENT_DURATIONS = [12, 24]
    LIST_OF_TRANSACTION_COST = [0, 0.01]


def load_data():
    """This function load the index fund timeseries and gives the result
    back in a dataframe. There are four index timeseries. One for
    debugging as well as three different indexes.

    Returns:
        pandas dataframe:  Initial index fund time series
    """
    # Load Excel file
    xls = pd.ExcelFile(
        '../daten/daten_Studienarbeit_Optimierung_Kapitalertragssteuer.xlsx')

    # Load Excel sheets
    sheets = []

    if DEBUGGING is True:
        sheets.append(pd.read_excel(xls, 'Testdata', header=0))
    else:
        sheets.append(pd.read_excel(xls, 'MSCI_World', header=0))
        sheets.append(pd.read_excel(xls, 'MSCI_EM', header=0))
        sheets.append(pd.read_excel(xls, 'MSCI_ACWI', header=0))

    # Create array of dataframes
    initial_df_array = []

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

    # Filter data to the smallest denominator of the time series
    initial_df_array = [
        df[~(df['date'] < '1988-01-01')]
        for df in initial_df_array]

    return initial_df_array


def end_of_capital_year(
        df_calc,
        current_period_index,
        monthly_investment_sum,
        transaction_cost):
    """_summary_

    Args:
        df_calc (_type_): _description_
        r_counter (_type_): _description_
        monthly_investment_sum (_type_): _description_
        transaction_cost (_type_): _description_
    """

    tax_free_capital_gain = TAX_FREE_CAPITAL_GAIN
    transaction_sum = 0.0
    former_period_index = current_period_index - 12
    if former_period_index < 0:
        former_period_index = 0

    current_value_of_index = df_calc.value_in_usd.values[current_period_index]

    # ToDo Extract function lookup former capital gains
    # Look back in table for capital gains and losses
    while former_period_index < current_period_index:

        # Calculate still to be taxed_profit on a row basis
        # Current Value of fund in relation to buy in value
        df_calc.not_taxed_profit.values[former_period_index] = (
            df_calc.not_taxed_investment.values[former_period_index]
            * current_value_of_index
            / df_calc.value_in_usd.values[former_period_index]
            - df_calc.not_taxed_investment.values[former_period_index]
        )

        # Skip zero lines, NaN and zero allowance
        if (df_calc.not_taxed_profit.values[former_period_index] == 0
                or df_calc.not_taxed_profit.values[former_period_index] is None
                or tax_free_capital_gain == 0):
            former_period_index += 1
            continue

        # Check if tax free capital gain allowance is available
        # greater than the profit in the period and if profit is
        # greater then rest allowance.
        if tax_free_capital_gain > df_calc.not_taxed_profit.values[former_period_index] \
                and tax_free_capital_gain >= 0:

            # Add order value to transaction sum
            transaction_sum = (
                transaction_sum
                + df_calc.not_taxed_profit.values[former_period_index]
                + df_calc.not_taxed_investment.values[former_period_index])

            # Detract the profit from tax free allowance
            tax_free_capital_gain = (
                tax_free_capital_gain
                - df_calc.not_taxed_profit.values[former_period_index])

            # Keep track of the taxed profits for statistics
            df_calc.taxed_profit.values[former_period_index] = (
                df_calc.taxed_profit.values[former_period_index]
                + df_calc.not_taxed_profit.values[former_period_index])

            # All profits of the period were sold
            df_calc.not_taxed_profit.values[former_period_index] = 0

            # Zero the buy order
            df_calc.not_taxed_investment.values[former_period_index] = 0

        else:
            # If the rest allowance is smaller then the profit, the
            # buy order needs to be split into two parts.

            # Calculate the rest buy order for future taxation. The
            # calculation is limited to the rest tax free allowance.
            df_calc.not_taxed_investment.values[former_period_index] = (
                df_calc.not_taxed_investment.values[former_period_index]
                - df_calc.not_taxed_investment.values[former_period_index]
                / (df_calc.not_taxed_profit.values[former_period_index]
                    / tax_free_capital_gain)
            )

            # Code only runs when allowance is smaller then profits.
            # Therefore the allowance is zero
            tax_free_capital_gain = 0

        former_period_index += 1

    # Save the transaction sum as a reinvestment
    df_calc.reinvestment.values[current_period_index] = transaction_sum

    # Create a new buy order so it can be taxed in the future
    df_calc.not_taxed_investment.values[current_period_index] = (
        df_calc.reinvestment.values[current_period_index]
        + monthly_investment_sum)

    # Deduct the transaction cost from the value of investment from
    # previous periods
    df_calc.transaction_cost.values[current_period_index] += [
        transaction_sum
        * transaction_cost
    ]


def table_calculation(
        df_calc,
        duration,
        t_counter,
        monthly_investment_sum,
        transaction_cost):
    """
    This method handles most of the df calculation on a row level.

    Args:
        df_calc (dataframe): table on which the calculation is done
        months (integer): How long you invest for in months?
        t_counter (integer): Which index is used. Is put into result table
        transaction_cost (float): How much percent costs a buy sell order?
        tax_free_capital_gain (integer): Tax free capital gain

    Returns:
        Two dataframes as an array: The calculation table and the result
                                    table row are returned.
    """

    # method variables
    df_calc.reset_index(drop=True)

    # Loop over each december period in table
    [end_of_capital_year(df_calc,
                         index,
                         monthly_investment_sum,
                         transaction_cost)
     for index, month in zip(df_calc.index.values, df_calc.month.values)
     if month == 12 and index != 0
     ]

    results = calc_results_on_df_calc(
        df_calc,
        t_counter,
        duration,
        monthly_investment_sum)

    # ToDo calculate value at the end

    if DEBUGGING is True:
        export_calc_df_to_csv(df_calc,
                              t_counter)

    return results


def setup_calculation(
        df_calc,
        lower_bound_month,
        upper_bound_month,
        monthly_investment_sum):
    """
    This method setups the used dfs.

    A few values are calculated, columns calculated and NaN values cleaned up.
    Also it cuts the dataframe to the proper period regarding to lower and
    upper bound of the current analysis.

    Args:
        df_calc (pandas dataframe):  Initial index fund time series
        lower_bound_month (integer): start of evaluation time series
        upper_bound_month (integer): end of evaluation time series

    Returns:
        pandas dataframe: Cut down and cleaned dataframe
    """

    # delete not necessary lines from dataframes with reference to months
    df_calc = df_calc.iloc[lower_bound_month:upper_bound_month].copy()

    # Calculate percentage changes in new column
    df_calc.loc[:, '%-change'] = float(0)
    df_calc.loc[:, '%-change'] = df_calc.loc[:,
                                             'value_in_usd'].pct_change() * 100
    # Fill NaN rows with zeroes
    df_calc.loc[:, '%-change'] = df_calc.loc[:, '%-change'].fillna(0)
    df_calc.loc[:, 'not_taxed_investment'] = float(monthly_investment_sum)
    df_calc.loc[:, 'not_taxed_profit'] = float(0)
    df_calc.loc[:, 'taxed_profit'] = float(0)
    df_calc.loc[:, 'reinvestment'] = float(0)
    df_calc.loc[:, 'transaction_cost'] = float(0)

    df_calc.reset_index(drop=True, inplace=True)

    return df_calc


def plot_results(df_results):
    """ Does the visualization of the results.

    Args:
        df_results (pandas dataframe): Result dataframe of all calculation runs
    """
    # plt.subplots(figsize=(15*CM, 5*CM))
    plt.hist(df_results.geometric_mean_return_of_index,
             # ToDo relative bin size regarding results
             bins=20)
    plt.title("Histogram of mean return of index")
    plt.xlabel("mean return of index in %")
    plt.ylabel("count of observations")

    plt.savefig(FILE_PATH_FOR_IMAGES
                + "Histogram_geometric_mean_return_of_index.svg")
    plt.close()

    # plt.hist(df_results.roi,
    #          # ToDo relative bin size regarding results
    #          bins=20)
    # plt.title("Histogram of return of investment")
    # plt.xlabel("return of investment in %")
    # plt.ylabel("count of observations")

    # plt.savefig(FILE_PATH_FOR_IMAGES
    #             + "Histogram_ROI.svg")
    # plt.close()


def generate_descriptive_info_results(results):
    """_summary_

    Args:
        results (_type_): _description_
    """
    results.describe().to_csv(
        './Results/tables/Result_Descriptive_Info.CSV'
    )


def plot_index_funds(initial_index_time_series):
    """_summary_
    """
    path = ""

    # do not change the initial_index_time_series
    index_time_series = initial_index_time_series

    for index in index_time_series:
        index['value_in_percent'] = [element / index.value_in_usd.values[0]
                                     for element in index.value_in_usd.values]

        plt.plot(index.date, index.value_in_percent)

    # name the dataframes for plotting
    if DEBUGGING is True:
        plt.legend("Testdata")
    else:
        plt.legend(["MSCI_World", "MSCI_EM", "MSCI_ACWI"])

    plt.title("Index values over time")
    plt.ylabel("price")
    plt.yscale("linear")
    plt.xlabel("date")
    plt.savefig(FILE_PATH_FOR_IMAGES
                + "Plot_index_performance.svg")
    plt.close()


def calc_results_on_df_calc(
        df_calc,
        t_counter,
        duration,
        monthly_investment_sum):
    """_summary_
    """
    # change to pandas series
    results = pd.DataFrame()

    results.loc[0, 'stock_market_index'] = t_counter
    results['duration_in_months'] = duration

    results['start_date'] = df_calc.date.iloc[0]
    results['end_date'] = df_calc.date.iloc[-1]

    results['start_value_index'] = df_calc.value_in_usd.iloc[0]
    results['end_value_index'] = df_calc.value_in_usd.iloc[-1]

    results['monthly_contribution'] = monthly_investment_sum

    results['realized_losses'] = df_calc[
        df_calc['taxed_profit'] < 0]['taxed_profit'].sum()

    results['realized_profits'] = df_calc[
        df_calc['taxed_profit'] > 0]['taxed_profit'].sum()

    results['reinvestment'] = df_calc.reinvestment.sum()

    results['transaction_cost'] = df_calc.transaction_cost.sum()

    # calculate value at the end of investment
    investment_value = [
        not_taxed_investment * df_calc.value_in_usd.values[-1] / current_value
        for not_taxed_investment, current_value in zip(df_calc.not_taxed_investment.values, df_calc.value_in_usd.values)
        if not_taxed_investment != 0
    ]

    results['investment_value_at_end'] = np.sum(investment_value)

    return results


def calc_results_on_results(results):
    """_summary_

    Args:
        results (_type_): _description_
    """

    results['trading_volume'] = (results['reinvestment']
                                 + results['monthly_contribution']
                                 * results['duration_in_months'])

    results['geometric_mean_return_of_index'] = (
        results['start_value_index']
        / results['end_value_index']
        / results['duration_in_months']
        * 100
    )

    results['investment'] = (results['transaction_cost']
                             + results['monthly_contribution']
                             * results['duration_in_months'])

    results['realized_losses_and_profits'] = (
        results['realized_losses']
        + results['realized_profits'])

    results['relative_profit'] = (
        (results['investment_value_at_end']
         - results['investment'])
        / results['investment']
    )

    results['profit'] = results['investment_value_at_end'] - \
        results['investment']

    return results


def export_results_df_to_csv(results):
    """ This method exports the results df to a excel file.

    Args:
        df_results (pandas dataframe): Result dataframe of all calculation runs
    """

    # Export df_results to CSV file
    results.to_csv('./Results/tables/Result_Export.CSV', index=True)


def export_calc_df_to_csv(df_calc, t_counter):
    """This method exports the calculation dfs to a excel file.

    Args:
        df (_type_): _description_
        t_counter (_type_): _description_
    """

    df_calc.sort_index(ascending=True)
    df_calc.to_csv('./Calculation_Files/Dataframe_Export_' +
                   str(t_counter) + '.CSV')


# main method call
main()
