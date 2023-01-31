"""This module is part of my science project at the DHBW CAS. Primary goal is
   to calculate a timeseries of investments into exchange traded index funds.
   Each december the program decides to sell stocks according to the yearly
   tax free capital gain. Results are compared to all time intervals in the
   dataset for the amount of the investment period.
    """

# Import Statements
import matplotlib.pyplot as plt
import pandas as pd

# Declaration of class variables
MONTHLY_CONTRIBUTION = 200   # how much money you put in each month
MONTHS = 24                  # how many months you invest for
TRANSACTION_COST = 0.01      # transaction cost per sell order
TAX_FREE_CAPITAL_GAIN = 800  # tax free capital gain allowance


def main():
    """
    This is the main method of the program.

    Here the loop logic over the df array is implemented as well as the
    different method calls.
    """

    # Create result df for summary data
    results = pd.DataFrame()

    # Create array of dataframes for looping
    initial_index_time_series = load_data()
    index_time_series = initial_index_time_series.copy()

    index_counter = 1

    for index_counter, df_calc in enumerate(index_time_series):

        upper_bound_month = MONTHS  # Start Value / how many months in dataset
        lower_bound_month = 0      # Start Value / end upper_bound minus months
        max_lower_bound_month = len(df_calc.index) - MONTHS
        max_lower_bound_month = 1

        # Iterate over all time intervals in the dataset
        while lower_bound_month < max_lower_bound_month:
            # Reset df
            df_calc = initial_index_time_series[index_counter].copy()
            df_calc = setup_calculation(df_calc,
                                        lower_bound_month,
                                        upper_bound_month,
                                        )

            df_calc, new_results = table_calculation(
                df_calc,
                MONTHS,
                index_counter,
                TRANSACTION_COST,
                TAX_FREE_CAPITAL_GAIN)

            results = pd.concat([results, new_results])

            upper_bound_month += 1
            lower_bound_month += 1

        export_df(df_calc, index_counter)

    export_results_df(results)

    # plot_results(df_array, result_df)


def load_data():
    """_summary_

    Returns:
        _type_: _description_
    """
    # Load Excel file
    __xls = pd.ExcelFile(
        '../Daten/Daten_Studienarbeit_Optimierung_Kapitalertragssteuer.xlsx')

    # Load Excel sheets
    sheets = []

    sheets.append(pd.read_excel(__xls, 'Testdata', header=0))
    sheets.append(pd.read_excel(__xls, 'MSCI_World', header=0))
    sheets.append(pd.read_excel(__xls, 'MSCI_EM', header=0))
    sheets.append(pd.read_excel(__xls, 'MSCI_ACWI', header=0))

    # Create array of dataframes for looping
    initial_df_array = []
    for sheet in sheets:
        initial_df_array.append(pd.DataFrame(
            sheet, columns=['Date', 'Value in USD']))

    return initial_df_array


def table_calculation(df_calc,
                      months,
                      t_counter,
                      transaction_cost,
                      tax_free_capital_gain):
    """
    This method handles most of the df calculation on a row level.

    Args:
        df (dataframe): df on which the calculation is done
        monthly_contribution (integer): How much you monthly invest
        months (integer): How long you invest for
        t_counter (integer): Is put into result df
        transaction_cost (float): How much percent costs a buy sell order
        tax_free_capital_gain (integer): Tax free capital gain

    Returns:
        Two dataframes as an array: The calculation df and the result df
                                    are returned.
    """

    # method variables
    df_calc = df_calc.copy()
    df_calc.reset_index(drop=True)
    reset_value_tax_free_capital_gain = tax_free_capital_gain

    for r_counter, row in enumerate(df_calc.iterrows()):

        date = row[1][0]
        index_value = row[1][1]

        # Sell stocks each year in Dec except if first period of observation
        if date.startswith('Dec') and r_counter != 0:

            # ToDO Extract function end of capital year
            tax_free_capital_gain = reset_value_tax_free_capital_gain
            transaction_sum = 0

            count = 0

            # ToDo Extract function lookup former capital gains
            # Loop for checking former periods for capital gains
            while count < r_counter:

                # Calculate still to be taxed profit on a row basis
                # Current Value of fund in relation to buy in value
                df_calc.iloc[count,
                             df_calc.columns.get_loc("not taxed Profit")] \
                    = df_calc['not taxed investment'].iloc[count] \
                    * index_value  \
                    / df_calc['Value in USD'].iloc[count] \
                    - df_calc['not taxed investment'].iloc[count]

                # Check if tax free capital gain allowance is available
                # greater than the profit in the period and if profit is
                # greater then rest allowance.

                if tax_free_capital_gain \
                    > df_calc['not taxed Profit'].iloc[count] \
                        and tax_free_capital_gain != 0:

                    # Add order value to transaction sum
                    transaction_sum = transaction_sum \
                        + df_calc['not taxed Profit'].iloc[count] \
                        + df_calc['not taxed investment'].iloc[count]

                    # Detract the profit from tax free allowance
                    tax_free_capital_gain = tax_free_capital_gain \
                        - df_calc['not taxed Profit'].iloc[count]

                    # Keep track of the taxed profits for statistics
                    df_calc.iloc[count,
                                 df_calc.columns.get_loc("Taxed Profit")] \
                        += df_calc['not taxed Profit'].iloc[count]

                    # All profits of the period were sold
                    df_calc.iloc[count,
                                 df_calc.columns.get_loc("not taxed Profit")] \
                        = 0

                    # Zero the buy order
                    df_calc.iloc[count,
                                 df_calc.columns.get_loc("not taxed investment")] \
                        = 0

                else:
                    # If the rest allowance is smaller then the profit, the
                    # buy order needs to be split into two parts.

                    # Skip zero lines, NaN and zero allowance
                    if df_calc['not taxed Profit'].iloc[count] == 0 \
                            or df_calc['not taxed Profit'].iloc[count] is None \
                            or tax_free_capital_gain == 0:
                        count += 1
                        continue

                    # print(df['not taxed Profit'].iloc[counter])
                    # print(tax_free_capital_gain)

                    # Calculate the rest buy order for future taxation. The
                    # calculation is limited to the rest tax free allowance.
                    df_calc.iloc[count,
                                 df_calc.columns.get_loc("not taxed investment")] \
                        = df_calc.iloc[count,
                                       df_calc.columns.get_loc("not taxed investment")]\
                        - df_calc.iloc[count,
                                       df_calc.columns.get_loc("not taxed investment")]\
                        / (df_calc['not taxed Profit'].iloc[count]
                           / tax_free_capital_gain)

                    # Code only runs when allowance is smaller then profits.
                    # Therefore the allowance is zero
                    tax_free_capital_gain = 0

                # print('df looks like this', temp_df)

                count += 1

            # Save the transaction sum as a reinvestment
            df_calc.iloc[r_counter, df_calc.columns.get_loc("Reinvestment")]\
                = transaction_sum

            # Create a new buy order so it can be taxed in the future
            df_calc.iloc[r_counter,
                         df_calc.columns.get_loc("not taxed investment")]\
                = df_calc.iloc[r_counter,
                               df_calc.columns.get_loc("Reinvestment")] \
                + MONTHLY_CONTRIBUTION

            # Deduct the transaction cost from the value of investment from
            # previous periods
            df_calc.iloc[r_counter, df_calc.columns.get_loc(
                "transaction_cost")] += transaction_sum * transaction_cost

    # ToDo extract function save result data
    # Save results Data from Dataset
    results = pd.DataFrame()

    results.loc[0, 'Stock Market Index Nr.'] = t_counter
    results.loc[0, 'duration in months'] = months
    results['start date'] = df_calc['Date'].iloc[0]
    results['end date'] = df_calc['Date'].iloc[-1]
    results['not taxed investment'] = MONTHLY_CONTRIBUTION
    # calculate value at the end

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
        df (_type_): _description_
        lower_bound_month (_type_): _description_
        upper_bound_month (_type_): _description_

    Returns:
        _type_: _description_
    """

    # delete not necessary lines from dataframes with reference to months
    df_calc = df_calc.iloc[lower_bound_month:upper_bound_month].copy()

    # Calculate percentage changes in new column
    df_calc.loc[:, '%-change'] = 0
    df_calc.loc[:, '%-change'] = df_calc.loc[:, 'Value in USD'].pct_change()
    # Fill NaN rows with zeroes
    df_calc.loc[:, '%-change'] = df_calc.loc[:, '%-change'].fillna(0)
    df_calc.loc[:, 'not taxed investment'] = MONTHLY_CONTRIBUTION
    df_calc.loc[:, 'not taxed Profit'] = 0
    df_calc.loc[:, 'Taxed Profit'] = 0
    df_calc.loc[:, 'Reinvestment'] = 0
    df_calc.loc[:, 'transaction_cost'] = 0

    return df_calc


def plot_results(df_array,
                 results):
    """ Does the visualization of the results.

    Args:
        df_array (_type_): _description_
        result_df (_type_): _description_
    """

    for df_results in df_array:
        df_results.plot()
        plt.show()

    results.plot()
    plt.show()


def generate_descriptive_info():
    """_summary_
    """
    # ToDO code generate_descriptive_info():


def export_results_df(results):
    """ This method exports the results df to a excel file.

    Args:
        result_df (_type_): _description_
    """

    # Export df_results to CSV file
    results.sort_index(ascending=True)
    results.to_csv('./Results/tables/Result_Export.CSV')


def export_df(df_calc, t_counter):
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
