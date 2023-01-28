"""_summary_

    Returns:
        _type_: _description_
    """
import pytest
import datatest as dt
from main import load_data


@pytest.mark.mandatory
def test_columns():
    """Do the title columns have the right names?
    """
    dataframe_array = load_data()
    for t_counter, dataframe in enumerate(dataframe_array):
        dt.validate(dataframe.columns[0], {'Date'})
        dt.validate(dataframe.columns[1], {'Value in USD'})

        # dataframe = table_calc(
        #     dataframe,
        #     200,
        #     12,
        #     t_counter,
        #     0.01,
        #     800)[0].copy()
        # dt.validate(dataframe.columns[2], {'%-change'})
        # dt.validate(dataframe.columns[3], {'not taxed investment'})
        # dt.validate(dataframe.columns[4], {
        #             'monthly Contribution accumulation'})
        # dt.validate(dataframe.columns[5], {'Account Balance'})
        # dt.validate(dataframe.columns[6], {'Not taxed Profit'})
        # dt.validate(dataframe.columns[7], {'Taxed Profit'})
        # dt.validate(dataframe.columns[8], {'Reinvestment'})


@pytest.mark.mandatory
def test_data_type():
    """Are the values numbers?
    """
    dataframe_array = load_data()
    for dataframe in dataframe_array:
        dt.validate(dataframe['Value in USD'], int)


@pytest.mark.mandatory
def test_null_values():
    """Are the values numbers?
    """
    dataframe_array = load_data()
    for dataframe in dataframe_array:
        dt.validate(dataframe.isnull().values.any(), False)
