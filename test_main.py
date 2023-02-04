"""This file is used for validation of the calculation results. For testing
    purposes a few files were calculated manually on a table. The results from
    the manual calculations are being compared to the automatic results.

    The test calculation was done on a index time series which was created for
    testing purposes and contains most common exemplary distributions. For
    example only positive changes from month to month, only negative and mixed
    cases.

    Additionally some simple tests are being done on the data import and so on.

    Returns:
        logfiles: logfile.txt for each specified test range
        output command line: Error report of pytest in command line
    """
import pytest
import datatest as dt
import pandas as pd
import datacompy
from main import load_data
from main import table_calculation
from main import setup_calculation


@pytest.mark.mandatory
def test_column_names():
    """Do the title columns have the right names?
    """
    dataframe_array = load_data()
    for dataframe in dataframe_array:
        dt.validate(dataframe.columns[0], {'date'})
        dt.validate(dataframe.columns[1], {'value_in_usd'})


@pytest.mark.mandatory
def test_data_type():
    """Are the values integers?
    """
    dataframe_array = load_data()
    for dataframe in dataframe_array:
        dt.validate(dataframe['value_in_usd'], int)


@pytest.mark.mandatory
def test_null_values():
    """Are the values integers?
    """
    table_array = load_data()
    for dataframe in table_array:
        dt.validate(dataframe.isnull().values.any(), False)


@pytest.mark.mandatory
def test_table_calculation():
    """Are the values calculated correct?
    """
    path = './Testfiles/'

    table_array = load_data()

    test_months = [11, 12, 24, 36]  # Corresponding to the test files

    # Load Excel sheets
    test_files = []

    for counter, month in enumerate(test_months):
        # Load Excel file
        test_files.append(pd.read_csv(
            path + 'Dataframe_Expected_Result_' + str(month) + '_Months.CSV',
            header=0))

        calc_table = pd.DataFrame(data=table_array[0])

        calc_table = setup_calculation(calc_table, 0, month)

        calc_table = table_calculation(calc_table, month, 0, 0.01, 800)[0]

        calc_table = pd.DataFrame(data=calc_table)

        # use datacompy

        compare = datacompy.Compare(
            test_files[counter],  # First dataframe to compare
            calc_table,  # Second dataframe to compare
            on_index=True,  # Join dataframes on index
            abs_tol=0.01,
            rel_tol=0,
            df1_name='Test_Files',
            df2_name='Generated_Output'
        )
        #dt.validate(compare.matches(ignore_extra_columns=False), True)

        # This method prints out a human-readable report summarizing
        # and sampling differences
        print(compare.report())
        with open('Logfiles/logfile_Month_'
                  + str(month)
                  + '.txt',
                  'w',
                  encoding='utf-8') as file:
            file.write(compare.report())
