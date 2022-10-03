import pytest
import pandas as pd
import datatest as dt
from main import main
main()

# load the excel files
# ToDo simplify code and return array with tables / dataframes to loop over


@pytest.fixture(scope='module')
@dt.working_directory(__file__)
def df_1():
    xls = pd.ExcelFile(
        '../Daten/Daten_Studienarbeit_Optimierung_Kapitalertragssteuer.xlsx')
    return pd.read_excel(xls, 'MSCI_World', header=0)


@pytest.fixture(scope='module')
@dt.working_directory(__file__)
def df_2():
    xls = pd.ExcelFile(
        '../Daten/Daten_Studienarbeit_Optimierung_Kapitalertragssteuer.xlsx')
    return pd.read_excel(xls, 'MSCI_EM', header=0)


@pytest.fixture(scope='module')
@dt.working_directory(__file__)
def df_3():
    xls = pd.ExcelFile(
        '../Daten/Daten_Studienarbeit_Optimierung_Kapitalertragssteuer.xlsx')
    return pd.read_excel(xls, 'MSCI_ACWI', header=0)


# Do the title columns have the right names
def test_columns(df_1, df_2, df_3):
    dt.validate(df_1.columns[0], {'Date'})
    dt.validate(df_1.columns[1], {'Value in USD'})
    dt.validate(df_2.columns[0], {'Date'})
    dt.validate(df_2.columns[1], {'Value in USD'})
    dt.validate(df_3.columns[0], {'Date'})
    dt.validate(df_3.columns[1], {'Value in USD'})


# Are the values numbers
@pytest.mark.mandatory
def test_values(df_1, df_2, df_3):
    dt.validate(df_1['Value in USD'], int)
    dt.validate(df_2['Value in USD'], int)
    dt.validate(df_3['Value in USD'], int)
