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
from main import set_debugging


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
        dt.validate(dataframe.value_in_usd, int)


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
    path_testfiles = './Testfiles/'
    path_output = './Calculation_Files/'

    set_debugging()

    table_array = load_data()

    durations = [11, 12, 24, 36]  # Corresponding to the test files

    monthly_investment_sum = 200
    transaction_cost = 0.01

    # Load Excel sheets
    test_files = []

    for counter, duration in enumerate(durations):
        # Load Excel file
        test_files.append(pd.read_csv(
            path_testfiles
            + 'Dataframe_Expected_Result_'
            + str(duration)
            + '_Months.CSV',
            header=0))

        df_calc = pd.DataFrame(data=table_array[0])

        df_calc = setup_calculation(df_calc,
                                    lower_bound_month=0,
                                    upper_bound_month=duration,
                                    monthly_investment_sum=monthly_investment_sum)

        table_calculation(df_calc,
                          duration,
                          counter,
                          monthly_investment_sum,
                          transaction_cost)

        df_calc = pd.read_csv(
            path_output
            + 'Dataframe_Export_'
            + str(counter)
            + '.CSV',
            header=0,
            index_col=0)

        # use datacompy
        compare = datacompy.Compare(
            test_files[counter],  # First dataframe to compare
            df_calc,  # Second dataframe to compare
            on_index=True,  # Join dataframes on index
            abs_tol=0.01,
            rel_tol=0,
            df1_name='Test_Files',
            df2_name='Generated_Output'
        )
        # dt.validate(compare.matches(ignore_extra_columns=False), True)

        # This method prints out a human-readable report summarizing
        # and sampling differences
        print(compare.report())
        with open('Logfiles/logfile_Month_'
                  + str(duration)
                  + '.txt',
                  'w',
                  encoding='utf-8') as file:
            file.write(compare.report())
