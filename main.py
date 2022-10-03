# Import Statements
import pandas as pd
import matplotlib.pyplot as plt

# Declaration of class variables


def main():
    # declaration of method variables
    monthlyContribution = 200   # how much money you put in each month
    months = 12                 # how many months you invest for
    previousAccountBalance = 0  # help variable / ToDo exchange for direct reference

    # Load Excel file
    xls = pd.ExcelFile(
        '../Daten/Daten_Studienarbeit_Optimierung_Kapitalertragssteuer.xlsx')

    # Load Excel sheets
    df = [pd.read_excel(xls,  'MSCI_World', header=0),
          pd.read_excel(xls, 'MSCI_EM', header=0),
          pd.read_excel(xls, 'MSCI_ACWI', header=0)]

    # Create array of dataframes for looping
    data = [pd.DataFrame(df[0], columns=['Date', 'Value in USD']),
            pd.DataFrame(df[1], columns=['Date', 'Value in USD']),
            pd.DataFrame(df[2], columns=['Date', 'Value in USD'])
            ]

    # print(data[0], data[1], data[2])    # Debugging Code

    # Add Columns and do simple calculation
    for dataframe in data:
        dataframe['%-change (base 100)'] = dataframe['Value in USD'] / \
            dataframe['Value in USD'][0]  # set relative values from first value
        dataframe['monthlyContribution'] = monthlyContribution
        dataframe['monthlyContribution accumulation'] = 0
        dataframe['Account Balance'] = 0

        # print(dataframe)  # Debugging Code

    # iterate over each row in dataframe and do the calculation
    for row in data[0].iterrows():

        if months - row[0] >= 1:
            row[1][4] = (row[0] + 1) * monthlyContribution
            # row[1][2] # i do not think this is the right logic
            row[1][5] = row[1][3] + previousAccountBalance * 1.10  # placeholder

            # Row to table Reference
            dataframe['monthlyContribution accumulation'][row[0]
                                                          ] = row[1][4]  # monthlyCont accu
            dataframe['Account Balance'][row[0]] = row[1][5]  # Account Balance

            print(row[0])     # Table index
            print(row[1][2])  # %-change (base 100)
            print(row[1][3])  # monthlyContribution
            print(row[1][4])  # monthlyContribution accumulation
            print(row[1][5])  # Account Balance
            print(row)
            print()

            # Remember former Account Balance
            previousAccountBalance = row[1][5]

    # Sell stocks each year and add sold value to variable

    # Export results for each dataframe to CSV file
    for counter, dataframe in enumerate(data):
        dataframe.to_csv('Dataframe_Export_' + str(counter) + '.CSV')

    # Iterate with different values

    # Plot results


# main function call
main()
