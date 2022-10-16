"""_summary_
   This module is part of my science project at the DHBW CAS. Primary goal is
   to calculate a timeseries of investments into exchange traded index funds.
   Each december the program decides to sell stocks according to the yearly
   tax free capital gain. Results are compared to all time intervalls in the
   dataset for the amount of the investment period.

    Returns:
        _type_: _description_
    """


# Import Statements
import matplotlib.pyplot as plt
import pandas as pd

# Declaration of class variables


def main():
    """
    This is the main method of the program.

    Here the loop logic over the table array is implemented as well as the
    different method calls.
    """

    # Create result table for summary data
    table_results = pd.DataFrame()

    # Create array of dataframes for looping
    initial_table_array = load_data()
    table_array = initial_table_array.copy()

    # declaration of method variables
    monthly_contribution = 200   # how much money you put in each month
    months = 25                   # how many months you invest for
    transaction_cost = 0.01      # transaction cost per sell order
    tax_free_capital_gain = 800

    for t_counter, table in enumerate(table_array):

        upper_bound_month = months  # Start Value / how many months in dataset
        lower_bound_month = 0      # Start Value / end upper_bound minus months
        max_lower_bound_month = len(table.index) - months
        max_lower_bound_month = 1

        # Iterate over all time intervalls in the dataset
        while lower_bound_month < max_lower_bound_month:
            table = initial_table_array[t_counter].copy()  # Reset table
            table = table_setup(table,
                                monthly_contribution,
                                lower_bound_month,
                                upper_bound_month,
                                )

            table = table_calc(
                table,
                monthly_contribution,
                months,
                t_counter,
                transaction_cost,
                tax_free_capital_gain)[0].copy()

            table_results = pd.concat(
                [table_results, table_calc(
                    table,
                    monthly_contribution,
                    months,
                    t_counter,
                    transaction_cost,
                    tax_free_capital_gain)[1].copy()], ignore_index=True).copy()

            upper_bound_month += 1
            lower_bound_month += 1

        export_table(table, t_counter)

    export_results_table(table_results)

    # plot_results(table_array, result_table)


def load_data():
    """_summary_

    Returns:
        _type_: _description_
    """
    # Load Excel file
    __xls = pd.ExcelFile(
        '../Daten/Daten_Studienarbeit_Optimierung_Kapitalertragssteuer.xlsx')

    # Load Excel sheets
    sheets = [pd.read_excel(__xls,  'MSCI_World', header=0),
              pd.read_excel(__xls, 'MSCI_EM', header=0),
              pd.read_excel(__xls, 'MSCI_ACWI', header=0)]

    # Create array of dataframes for looping
    initial_table_array = []
    for sheet in sheets:
        initial_table_array.append(pd.DataFrame(
            sheet, columns=['Date', 'Value in USD']))

    return initial_table_array


def table_calc(table,
               monthly_contribution,
               months,
               t_counter,
               transaction_cost,
               tax_free_capital_gain):
    """
    This method handles most of the table calculation on a row level.

    Args:
        table (dataframe): Table on which the calculation is done
        monthly_contribution (integer): How much you monthly invest
        months (integer): How long you invest for
        t_counter (integer): Is put into result table
        transaction_cost (float): How much percent costs a buy sell order
        tax_free_capital_gain (integer): How much tax free capital gain is there

    Returns:
        Two dataframes as an array: The calculation table and the result table
                                    are returned.
    """

    # method variables
    previous_account_balance = 0  # ToDo exchange for direct reference
    table = table.copy()
    table.reset_index(drop=True)
    reset_value_tax_free_capital_gain = tax_free_capital_gain

    # print(table)

    for r_counter, row in enumerate(table.iterrows()):

        date = row[1][0]
        index_value = row[1][1]

        # Row calculations
        table.iloc[r_counter, table.columns.get_loc("monthly Contribution accumulation")] = (
            r_counter + 1) * monthly_contribution

        # Calculate current account balance from change of previous period
        table.iloc[r_counter, table.columns.get_loc("Account Balance")] = monthly_contribution + \
            previous_account_balance * \
            (table.iloc[r_counter, table.columns.get_loc("%-change")] + 1)

        # Sell stocks each year in Dec
        if date.startswith('Dec') and r_counter != 0:
            tax_free_capital_gain = reset_value_tax_free_capital_gain
            transaction_sum = 0

            counter = 0
            # Check former periods for capital gains
            while counter < r_counter:

                # Calculate still to be taxed profit on a row basis
                table.iloc[counter, table.columns.get_loc("not taxed Profit")] \
                    = table['monthly not taxed investment'].iloc[counter] \
                    * index_value  \
                    / table['Value in USD'].iloc[counter] \
                    - table['monthly not taxed investment'].iloc[counter]

                # Check if tax free capital gain allowance is available
                # greater than the profit in the period and if profit is
                # greater then rest allowance.

                if tax_free_capital_gain > table['not taxed Profit'].iloc[counter] \
                        and tax_free_capital_gain != 0:

                    # Add order value to transaction sum
                    transaction_sum += table['not taxed Profit'].iloc[counter] + \
                        table['monthly not taxed investment'].iloc[counter]

                    # Detract the profit from tax free allowance
                    tax_free_capital_gain -= table['not taxed Profit'].iloc[counter]

                    # Keep track of the taxed profits for statistics
                    table.iloc[counter, table.columns.get_loc("Taxed Profit")] \
                        += table['not taxed Profit'].iloc[counter]

                    # All profits of the period were sold
                    table.iloc[counter, table.columns.get_loc("not taxed Profit")] \
                        = 0

                    # Zero the buy order
                    table.iloc[counter, table.columns.get_loc(
                        "monthly not taxed investment")] = 0

                else:
                    # If the rest allowance is smaller then the profit, the
                    # buy order needs to be split into two parts.

                    print(table.iloc[counter])

                    # Skip zero lines, NaN and zero allowance
                    if table['not taxed Profit'].iloc[counter] == 0 \
                        and table['not taxed Profit'].iloc[counter]\
                            and tax_free_capital_gain != 0:
                        counter += 1
                        continue

                    #print(table['not taxed Profit'].iloc[counter])
                    # print(tax_free_capital_gain)

                    # Calculate the rest buy order for future taxation. The
                    # calculation is limeted to the rest tax free allowance.
                    table.iloc[counter, table.columns.get_loc("monthly not taxed investment")] \
                        = table.iloc[counter, table.columns.get_loc(
                            "monthly not taxed investment")]\
                        - table.iloc[counter, table.columns.get_loc(
                            "monthly not taxed investment")]\
                        / (table['not taxed Profit'].iloc[counter]
                           / tax_free_capital_gain)

                    # Code only runs when allowance is smaller then profits.
                    # Therefore the allowance is zero
                    tax_free_capital_gain = 0

                 # print('Tabelle sieht momentan so aus', temp_table)

                counter += 1

            # Save the transaction sum as a reinvestment
            table.iloc[r_counter, table.columns.get_loc("Reinvestment")]\
                = transaction_sum

            # Create a new buy order so it can be taxed in the future
            table.iloc[r_counter, table.columns.get_loc("monthly not taxed investment")]\
                = table.iloc[r_counter, table.columns.get_loc("Reinvestment")] \
                - table.iloc[r_counter, table.columns.get_loc("Reinvestment")] \
                * transaction_cost

            # Deduct the transaction cost from the value of investment from
            # previous periods
            table.iloc[r_counter, table.columns.get_loc(
                "Account Balance")] -= transaction_sum * transaction_cost

        # Remember former Account Balance
        previous_account_balance = table.iloc[r_counter, table.columns.get_loc(
            "Account Balance")]

    # Save results Data from Dataset
    result_table_row = pd.DataFrame()

    # print('This is the result from table_calc')
    # print(temp_table)

    result_table_row.loc[0, 'Stock Market Index Nr.'] = t_counter
    result_table_row.loc[0, 'duration in months'] = months
    result_table_row['start date'] = table['Date'].iloc[0]
    result_table_row['end date'] = table['Date'].iloc[-1]
    result_table_row['monthly not taxed investment'] = monthly_contribution
    result_table_row['monthly Contribution accumulation'] = \
        table['monthly Contribution accumulation'].iloc[-1]
    result_table_row['Account Balance at end'] = table['Account Balance'].iloc[-1]

    # print(r_t_result)

    # __TABLE_RESULTS = pd.concat(
    #   [__TABLE_RESULTS, r_t_result], ignore_index=True).copy()

    return table, result_table_row


def table_setup(table,
                monthly_contribution,
                lower_bound_month,
                upper_bound_month):
    """
    This method setups the used tables.

    A few values are calculated, collumns calculated and NaN values cleaned up.
    Also it cuts the dataframe to the proper periode regarding to lower and
    upper bound of the current analysis.

    Args:
        table (_type_): _description_
        monthly_contribution (_type_): _description_
        lower_bound_month (_type_): _description_
        upper_bound_month (_type_): _description_

    Returns:
        _type_: _description_
    """

    # delete not nessesary lines from dataframes with reference to months
    table = table.iloc[lower_bound_month:upper_bound_month].copy()

    # Calculate prozentual changes in new collumn
    table.loc[:, '%-change'] = 0
    table.loc[:, '%-change'] = table.loc[:, 'Value in USD'].pct_change()
    # Fill NaN rows with zeroes
    table.loc[:, '%-change'] = table.loc[:, '%-change'].fillna(0)
    table.loc[:, 'monthly not taxed investment'] = monthly_contribution
    table.loc[:, 'monthly Contribution accumulation'] = 0
    table.loc[:, 'Account Balance'] = 0
    table.loc[:, 'not taxed Profit'] = 0
    table.loc[:, 'Taxed Profit'] = 0
    table.loc[:, 'Reinvestment'] = 0

    return table


def plot_results(table_array,
                 result_table):
    """ Does the visualization of the results.

    Args:
        table_array (_type_): _description_
        result_table (_type_): _description_
    """

    for table in table_array:
        table.plot()
        plt.show()

    result_table.plot()
    plt.show()


def export_results_table(result_table):
    """ This method exports the results table to a excel file.

    Args:
        result_table (_type_): _description_
    """

    # Export table_results to CSV file
    result_table.sort_index(ascending=True)
    result_table.to_csv('Result_Export.CSV')


def export_table(table, t_counter):
    """This method exports the calculation tables to a excel file.

    Args:
        table (_type_): _description_
        t_counter (_type_): _description_
    """

    table.sort_index(ascending=True)
    table.to_csv('Dataframe_Export_' + str(t_counter) + '.CSV')


# main method call
main()
