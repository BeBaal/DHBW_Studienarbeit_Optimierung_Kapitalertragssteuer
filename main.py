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

# Declaration of class variables
MONTHLY_CONTRIBUTION = 200   # how much money you put in each month
# ToDo implement loop over periods
LIST_OF_INVESTMENT_DURATIONS = [240, 300, 360, 420, 480]
INVESTMENT_DURATION = 24     # how many months you invest for
TRANSACTION_COST = 0.01      # transaction cost per sell order
TAX_FREE_CAPITAL_GAIN = 800  # tax free capital gain allowance
FILE_PATH_FOR_IMAGES = r'./Results/Figures/'  # Path for image export
CM = 1/2.54                  # centimeters in inches


def main():
    """
    This is the main method of the program.

    Here the loop logic over the df array is implemented as well as the
    different method calls.
    """

    # get the start time
    start_time = time.time()

    # Create array of dataframes for looping
    initial_index_time_series = load_data()

    results = iteration_over_index_fund_list(initial_index_time_series)

    export_results_df_to_csv(results)
    generate_descriptive_info_index_funds(results)
    plot_results(results)

    # get end time
    end_time = time.time()

    # calculate calculation time
    elapsed_time = end_time - start_time
    print('Execution time:', elapsed_time, 'seconds')


def iteration_over_index_fund_list(initial_index_time_series):
    # Create result df for summary data
    results = pd.DataFrame()

    index_time_series = initial_index_time_series.copy()

    # loop over all index time series
    for index_counter, df_calc in enumerate(index_time_series):

        # Start Value / how many months in dataset
        upper_bound_month = INVESTMENT_DURATION
        lower_bound_month = 0      # Start Value / end upper_bound minus months
        max_lower_bound_month = len(df_calc.index) - INVESTMENT_DURATION
        # max_lower_bound_month = 1 # setting for debugging

        # Iterate over all time intervals in the dataset
        while lower_bound_month < max_lower_bound_month:

            # Reset df
            df_calc = initial_index_time_series[index_counter].copy()
            df_calc = setup_calculation(df_calc,
                                        lower_bound_month,
                                        upper_bound_month
                                        )

            df_calc, new_results = table_calculation(
                df_calc,
                INVESTMENT_DURATION,
                index_counter,
                TRANSACTION_COST,
                TAX_FREE_CAPITAL_GAIN)

            results = pd.concat([results, new_results])

            upper_bound_month += 1
            lower_bound_month += 1

        export_calc_df_to_csv(df_calc, index_counter)

    return results


def iteration_over_investment_duration():
    pass


def iteration_over_time_series():
    pass


def load_data():
    """This function load the index fund timeseries and gives the result
    back in a dataframe. There are four index timeseries. One for
    debugging as well as three different indexes.

    Returns:
        pandas dataframe:  Initial index fund time series
    """
    # Load Excel file
    __xls = pd.ExcelFile(
        '../daten/daten_Studienarbeit_Optimierung_Kapitalertragssteuer.xlsx')

    # Load Excel sheets
    sheets = []

    sheets.append(pd.read_excel(__xls, 'Testdata', header=0))
    # sheets.append(pd.read_excel(__xls, 'MSCI_World', header=0))
    # sheets.append(pd.read_excel(__xls, 'MSCI_EM', header=0))
    # sheets.append(pd.read_excel(__xls, 'MSCI_ACWI', header=0))

    # Create array of dataframes for looping
    initial_df_array = []
    for sheet in sheets:
        initial_df_array.append(pd.DataFrame(
            sheet,
            columns=['date', 'value_in_usd']))

    return initial_df_array


def table_calculation(df_calc,
                      months,
                      t_counter,
                      transaction_cost,
                      tax_free_capital_gain):
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

    for r_counter, row in enumerate(df_calc.iterrows()):
        # Loop over each period in table

        # Sell stocks each year in Dec except if first period of observation
        if row[1].date.startswith('Dec') and r_counter != 0:

            # ToDO Extract function end of capital year
            tax_free_capital_gain = TAX_FREE_CAPITAL_GAIN
            transaction_sum = 0.0
            count = 0

            # ToDo Extract function lookup former capital gains
            # Look back in table for capital gains and losses
            while count < r_counter:

                # Calculate still to be taxed_profit on a row basis
                # Current Value of fund in relation to buy in value
                df_calc.not_taxed_profit.values[count] = (
                    df_calc.not_taxed_investment.values[count]
                    * row[1].value_in_usd
                    / df_calc.value_in_usd.values[count]
                    - df_calc.not_taxed_investment.values[count]
                )

                # Check if tax free capital gain allowance is available
                # greater than the profit in the period and if profit is
                # greater then rest allowance.

                if tax_free_capital_gain > df_calc.not_taxed_profit.values[count] \
                        and tax_free_capital_gain != 0:

                    # Add order value to transaction sum
                    transaction_sum = transaction_sum \
                        + df_calc.not_taxed_profit.values[count] \
                        + df_calc.not_taxed_investment.values[count]

                    # Detract the profit from tax free allowance
                    tax_free_capital_gain = tax_free_capital_gain \
                        - df_calc.not_taxed_profit.values[count]

                    # Keep track of the taxed profits for statistics
                    df_calc.taxed_profit.values[count] \
                        += df_calc.not_taxed_profit.values[count]

                    # All profits of the period were sold
                    df_calc.not_taxed_profit.values[count] = 0

                    # Zero the buy order
                    df_calc.not_taxed_investment.values[count] = 0

                else:
                    # If the rest allowance is smaller then the profit, the
                    # buy order needs to be split into two parts.

                    # Skip zero lines, NaN and zero allowance
                    if (df_calc.not_taxed_profit.values[count] == 0
                            or df_calc.not_taxed_profit.values[count] is None
                            or tax_free_capital_gain == 0):
                        count += 1
                        continue

                    # Calculate the rest buy order for future taxation. The
                    # calculation is limited to the rest tax free allowance.
                    df_calc.not_taxed_investment.values[count] = [
                        df_calc.not_taxed_investment.values[count]
                        - df_calc.not_taxed_investment.values[count]
                        / (df_calc.not_taxed_profit.values[count]
                           / tax_free_capital_gain)
                    ]

                    # Code only runs when allowance is smaller then profits.
                    # Therefore the allowance is zero
                    tax_free_capital_gain = 0

                # print('df looks like this', temp_df)

                count += 1

            # Save the transaction sum as a reinvestment
            df_calc.reinvestment.values[r_counter] = transaction_sum

            # Create a new buy order so it can be taxed in the future
            df_calc.not_taxed_investment.values[r_counter] = (
                df_calc.reinvestment.values[r_counter]
                + MONTHLY_CONTRIBUTION)

            # Deduct the transaction cost from the value of investment from
            # previous periods
            df_calc.transaction_cost.values[r_counter] += [
                transaction_sum
                * transaction_cost
            ]

    # ToDo extract function save result data
    results = calc_results(df_calc,
                           t_counter,
                           months)

    # ToDo calculate value at the end

    return df_calc, results


def setup_calculation(df_calc,
                      lower_bound_month,
                      upper_bound_month):
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
    df_calc.loc[:, 'not_taxed_investment'] = float(MONTHLY_CONTRIBUTION)
    df_calc.loc[:, 'not_taxed_profit'] = float(0)
    df_calc.loc[:, 'taxed_profit'] = float(0)
    df_calc.loc[:, 'reinvestment'] = float(0)
    df_calc.loc[:, 'transaction_cost'] = float(0)

    return df_calc


def plot_results(df_results):
    """ Does the visualization of the results.

    Args:
        df_results (pandas dataframe): Result dataframe of all calculation runs
    """
    # plt.subplots(figsize=(15*CM, 5*CM))
    plt.hist(df_results.geometric_mean_return_of_index,
             bins=80)  # ToDo relative bin size regarding results
    plt.savefig(FILE_PATH_FOR_IMAGES
                + "Histogram_geometric_mean_return_of_index.svg")
    plt.close()


def generate_descriptive_info_index_funds(results):
    results.describe().to_csv(
        './Results/tables/Result_Descriptive_Info.CSV'
    )


def generate_graphs_index_funds():
    pass


def calc_results(df_calc, t_counter, months):
    """_summary_
    """
    # ToDO code generate_descriptive_info_index_funds():

    # Save results Data from Dataset
    results = pd.DataFrame()

    results.loc[0, 'stock_market_index'] = t_counter
    results.loc[0, 'duration_in_months'] = months

    results['start_date'] = df_calc.date.iloc[0]
    results['end_date'] = df_calc.date.iloc[-1]

    results['start_value_index'] = df_calc.value_in_usd.iloc[0]
    results['end_value_index'] = df_calc.value_in_usd.iloc[-1]

    results['geometric_mean_return_of_index'] = [
        results['start_value_index']
        / results['end_value_index']
        / results.loc[0, 'duration_in_months']
    ]

    results['monthly_contribution'] = MONTHLY_CONTRIBUTION

    results['losses'] = df_calc[
        df_calc['taxed_profit'] < 0
    ]['taxed_profit'].sum()

    results['profits'] = df_calc[
        df_calc['taxed_profit'] > 0
    ]['taxed_profit'].sum()

    results['losses_and_profits'] = results['losses'] + results['profits']

    return results


def export_results_df_to_csv(results):
    """ This method exports the results df to a excel file.

    Args:
        df_results (pandas dataframe): Result dataframe of all calculation runs
    """

    # Export df_results to CSV file
    results.sort_index(ascending=True)
    results.to_csv('./Results/tables/Result_Export.CSV')


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
