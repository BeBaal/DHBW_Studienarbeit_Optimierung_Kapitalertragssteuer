# Import Statements
import pandas as pd
import matplotlib.pyplot as plt

# Declaration of class variables


def main():
    # declaration of method variables
    monthlyContribution = 200   # how much money you put in each month
    months = 360                 # how many months you invest for

    # Load Excel file
    xls = pd.ExcelFile(
        '../Daten/Daten_Studienarbeit_Optimierung_Kapitalertragssteuer.xlsx')

    # Load Excel sheets
    sheets = [pd.read_excel(xls,  'MSCI_World', header=0),
              pd.read_excel(xls, 'MSCI_EM', header=0),
              pd.read_excel(xls, 'MSCI_ACWI', header=0)]

    # Create array of dataframes for looping
    data = []
    for sheet in sheets:
        data.append(pd.DataFrame(sheet, columns=['Date', 'Value in USD']))

    # delete not nessesary lines from dataframes with reference to months
    for counter, dataframe in enumerate(data):
        data[counter] = dataframe.iloc[:months]

    # Add Columns and do simple calculation
    for dataframe in data:
        # Calculate prozentual changes in new collumn
        dataframe['%-change'] = dataframe['Value in USD'].pct_change()
        # Fill NaN rows with zeroes
        dataframe['%-change'] = dataframe['%-change'].fillna(0)
        dataframe['monthlyContribution'] = monthlyContribution
        dataframe['monthlyContribution accumulation'] = 0
        dataframe['Account Balance'] = 0

        # print(dataframe)  # Debugging Code

    # iterate over each row in dataframe and do the calculation
    for dataframe in data:
        previousAccountBalance = 0  # help variable / ToDo exchange for direct reference

        # print()  # Debugging Code

        for counter, row in enumerate(dataframe.iterrows()):

            if months - row[0] >= 1:

                # Row calculations
                row[1][4] = (row[0] + 1) * monthlyContribution
                row[1][5] = row[1][3] + \
                    previousAccountBalance * (row[1][2] + 1)

                # Save row to table
                dataframe.loc[counter, 'monthlyContribution accumulation'] \
                    = row[1][4]  # monthlyContribution accumulation
                dataframe.loc[counter, 'Account Balance'] = row[1][5]

                # Debugging Code on line level
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

    # Export results for each dataframe to CSV file
    for counter, dataframe in enumerate(data):
        dataframe.to_csv('Dataframe_Export_' + str(counter) + '.CSV')

    # Iterate with different values

    # Plot results of dataframes
    # for counter, dataframe in enumerate(data):
    # data[0].plot()
    # plt.show()


# main function call
main()
