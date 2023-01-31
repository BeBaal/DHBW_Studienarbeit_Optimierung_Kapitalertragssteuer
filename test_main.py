"""_summary_

    Returns:
        _type_: _description_
    """
import pytest
import datatest as dt
import pandas as pd
import datacompy
from main import load_data
from main import table_calculation
from main import setup_calculation


@pytest.mark.mandatory
def test_columns():
    """Do the title columns have the right names?
    """
    dataframe_array = load_data()
    for dataframe in dataframe_array:
        dt.validate(dataframe.columns[0], {'Date'})
        dt.validate(dataframe.columns[1], {'Value in USD'})


@pytest.mark.mandatory
def test_data_type():
    """Are the values integers?
    """
    dataframe_array = load_data()
    for dataframe in dataframe_array:
        dt.validate(dataframe['Value in USD'], int)


@pytest.mark.mandatory
def test_null_values():
    """Are the values integers?
    """
    table_array = load_data()
    for dataframe in table_array:
        dt.validate(dataframe.isnull().values.any(), False)


@pytest.mark.mandatory
def test_table_calculation():
    """Are the values calculated correct
    """
    path = './Testfiles/'

    table_array = load_data()

    test_months = [11, 12, 24, 36]

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
            test_files[counter],  # old
            calc_table,  # new
            on_index=True,
            abs_tol=0.01,  # Optional, defaults to 0
            rel_tol=0,  # Optional, defaults to 0
            df1_name='Original',  # Optional, defaults to 'df1'
            df2_name='New'  # Optional, defaults to 'df2'
        )
        dt.validate(compare.matches(ignore_extra_columns=False), True)

        # This method prints out a human-readable report summarizing
        # and sampling differences
        print(compare.report())
        with open('Logfiles/logfile_Month_'
                  + str(month)
                  + '.txt',
                  'w',
                  encoding='utf-8') as file:
            file.write(compare.report())
