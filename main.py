# Import Statements
import pandas as pd
import matplotlib.pyplot as plt

# Declaration of class variables
# Load Excel file
xls = pd.ExcelFile(
    '../Daten/Daten_Studienarbeit_Optimierung_Kapitalertragssteuer.xlsx')

# Load Excel sheets
sheets = [pd.read_excel(xls,  'MSCI_World', header=0),
          pd.read_excel(xls, 'MSCI_EM', header=0),
          pd.read_excel(xls, 'MSCI_ACWI', header=0)]

# Create result table for summary data
t_results = pd.DataFrame()

# Create array of dataframes for looping
initial_data = []
data = []
for sheet in sheets:
    initial_data.append(pd.DataFrame(sheet, columns=['Date', 'Value in USD']))
    data.append(pd.DataFrame(sheet, columns=['Date', 'Value in USD']))


def main():
    # use global variables
    global data
    global t_results

    # declaration of method variables
    monthlyContribution = 200   # how much money you put in each month
    months = 12                 # how many months you invest for
    upper_bound = 0             # how many months in dataset
    lower_bound = 0             # start with zero and end with upper_bound minus months

    # Here goes the for loop which every x month long period in dataset is analyzed

    # delete not nessesary lines from dataframes with reference to months
    for t_counter, t_calculation in enumerate(data):
        data[t_counter] = t_calculation.iloc[:months]

    # Add Columns and do simple calculation
    for t_calculation in data:

        # Calculate prozentual changes in new collumn
        t_calculation['%-change'] = t_calculation['Value in USD'].pct_change()
        # Fill NaN rows with zeroes
        t_calculation['%-change'] = t_calculation['%-change'].fillna(0)
        t_calculation['monthly Contribution'] = monthlyContribution
        t_calculation['monthly Contribution accumulation'] = 0
        t_calculation['Account Balance'] = 0

    # print(t_calculation)  # Debugging Code

    # iterate over each row in dataframe and do the calculation
    for t_counter, t_calculation in enumerate(data):
        previousAccountBalance = 0  # help variable / ToDo exchange for direct reference

        # print()  # Debugging Code

        for r_counter, row in enumerate(t_calculation.iterrows()):

            if months - r_counter >= 1:

                # Row calculations
                row[1][4] = (r_counter + 1) * monthlyContribution
                row[1][5] = row[1][3] + \
                    previousAccountBalance * (row[1][2] + 1)

                # Save row to table
                t_calculation.loc[r_counter, 'monthly Contribution accumulation'] \
                    = row[1][4]
                t_calculation.loc[r_counter, 'Account Balance'] = row[1][5]

                # Debugging Code on line level
                # print(r_counter)
                # print(row[0])     # Table index
                # print(row[1][2])  # %-change (base 100)
                # print(row[1][3])  # monthlyContribution
                # print(row[1][4])  # monthlyContribution accumulation
                # print(row[1][5])  # Account Balance
                # print(row)
                # print(previousAccountBalance)
                # print()

                # Remember former Account Balance
                previousAccountBalance = row[1][5]

                # Sell stocks each year and add sold value to variable

        # Save some Data from Dataset
        t_results.loc[t_counter,
                      'duration'] = months
        t_results.loc[t_counter,
                      'start date'] = t_calculation.loc[0, 'Date']
        t_results.loc[t_counter,
                      'end date'] = t_calculation.loc[months - 1, 'Date']
        t_results.loc[t_counter,
                      'monthly Contribution'] = t_calculation.loc[0, 'monthly Contribution']
        t_results.loc[t_counter,
                      'monthly Contribution accumulation'] = t_calculation.loc[months - 1, 'monthly Contribution accumulation']
        t_results.loc[t_counter,
                      'Account Balance at end'] = t_calculation.loc[months - 1, 'Account Balance']

    print(t_results)  # Debugging Code

    export_results()

    # plot_results()


def plot_results():
    global data

    for t_calculation in data:
        t_calculation.plot()
        plt.show()


def export_results():
    global data
    # Export t_results for each dataframe to CSV file
    for t_counter, t_calculation in enumerate(data):
        t_calculation.to_csv('Dataframe_Export_' +
                             str(t_counter) + '.CSV')

    # Export summary to CSV file
    t_results.to_csv('Result_Export.CSV')


# main function call
main()
