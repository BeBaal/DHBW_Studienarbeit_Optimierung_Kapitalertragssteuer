import pandas as pd
import matplotlib.pyplot as plt


def main():

    # Load Excel files
    xls = pd.ExcelFile(
        '../Daten/Daten_Studienarbeit_Optimierung_Kapitalertragssteuer.xlsx')
    df_MSCI_World = pd.read_excel(
        xls,  'MSCI_World', header=0)
    df_MSCI_EM = pd.read_excel(xls, 'MSCI_EM')
    df_MSCI_ACWI = pd.read_excel(xls, 'MSCI_ACWI')

    # Load Dataframes
    d_MSCI_World = pd.DataFrame(
        df_MSCI_World, columns=['Date', 'Value in USD'])
    d_MSCI_EM = pd.DataFrame(
        df_MSCI_EM, columns=['Date', 'Value in USD'])
    d_MSCI_ACWI = pd.DataFrame(
        df_MSCI_ACWI, columns=['Date', 'Value in USD'])
    # print(d_MSCI_World, d_MSCI_EM, d_MSCI_ACWI)    # Debugging Code

    # Change Type to numeric and strings are parsed to NaN, replace by 0
    # d_MSCI_World['Value in USD'] = pd.to_numeric(
    #     d_MSCI_World['Value in USD'], errors='coerce').fillna(0).astype(complex)
    # d_MSCI_EM['Value in USD'] = pd.to_numeric(d_MSCI_EM['Value in USD'])
    # d_MSCI_ACWI['Value in USD'] = pd.to_numeric(
    #    d_MSCI_ACWI['Value in USD'], errors='coerce').fillna(0).astype(complex)

    # Add Column
    d_MSCI_World['%-change (base 100)'] = d_MSCI_World['Value in USD'] / \
        d_MSCI_World['Value in USD'][0]
    d_MSCI_EM['%-change (base 100)'] = d_MSCI_EM['Value in USD'] / \
        d_MSCI_EM['Value in USD'][0]
    d_MSCI_ACWI['%-change (base 100)'] = d_MSCI_ACWI['Value in USD'] / \
        d_MSCI_ACWI['Value in USD'][0]

    print(d_MSCI_World, d_MSCI_EM, d_MSCI_ACWI)    # Debugging Code


# main function call
main()
