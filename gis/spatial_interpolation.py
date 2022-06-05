
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

# load crosswalk
block_groups = pd.read_csv('./data/block_group_areas.csv')  # df of block_group, area
# crosswalk: df where index is beats, columns are block groups, values are fraction of block group area in beat area
crosswalk = pd.read_csv('./data/blockgroup_beat_crosswalk.csv').set_index("beat_num")

def blockgroups_to_beats(data,variables={},source_geo=None):
    """
    Aggregates dataframe of variables at 2010s block group level to current (as of 2022) police beats.

    Must specify variable type for each aggregated variable. Options:
    - "extensive": e.g. number of households. Aggregated via areal sum.
    - "area-weighted": e.g. average temperature. Aggregated via areal mean. 
    - weighted: e.g. mean income. Pass the weight column name (ex: "population"). Aggregated via [weight variable] mean.
    - "margin of error": Aggregated as sqrt(variance).

    Args:
        data: Dataframe containing columns to aggregate. 
        variables: Dictionary with {"column_to_aggregate":variable_type}. 
            Columns not specified in dictionary will be dropped.
        source_geo (opt): String name of column identifying block groups.
            Defaults to "block group".

    Returns:
        (Geo)Dataframe containing aggregated columns by beat.

    Example:
        We have dataframe with "Population", "Households", "Park Cover", "Per Capita Income", "Mean Household Income".
        The proper argument to pass to variable is:
        {
            "Population": "extensive",
            "Households": "extensive",
            "Park Cover": "area-weighted",
            "Per Capita Income": "Population",
            "Mean Household Income": "Households"
        }
    """

    # prep/validate arguments
    if source_geo is None:
        if "block_group" in data.columns:
            source_geo = "block_group"  # default
        else:
            raise ValueError(f"Please specify source_geo argument.")
    else:
        if source_geo not in data.columns: 
            raise ValueError(f"Specified block group column '{source_geo}' is not in dataframe.")
    extensive_vars = []
    weighted_vars = []
    margins = []
    for variable, method in variables.items():
        if variable == "area":
            raise ValueError("Please rename the 'area' variable, it might cause issues!")
        if variable not in data.columns:
            raise ValueError(f"Specified variable '{variable}' is not in dataframe.")
        if method == "extensive":
            extensive_vars.append(variable)
        elif method == "area-weighted":
            weighted_vars.append([variable,"area"])
        elif method == "margin of error":
            margins.append(variable)
        elif method in data.columns:
            weighted_vars.append([variable,method])
        else:
            raise ValueError(f"Specified weight '{method}' is not in dataframe.")
    data = data[list(variables.keys())+[source_geo]]  # drop unmentioned columns
    if data.isnull().values.any():  # implementation note: could alternatively leave NAs and allow them to become NAs in results
        raise ValueError("Remove NAs from passed data and try again.")

    # convert margins of error to variances
    for variable in margins:
        data[variable] = data[variable]**2

    # merge with list of block groups (plus areas)
    try:
        data[source_geo] = data[source_geo].astype('int64')
        data = pd.merge(block_groups,data,how="left",on=source_geo)  # left outer join: NA where block group is missing
        # note that this also ensures block groups are in order
        if len(data) == 0:
            raise
    except:
        raise Exception(f"Failed to merge data with block groups - check values of block group column '{source_geo}'.")

    # multiply weighted variables by weighting variables 
    # (intuition: for example, calculate total income per block group by multiplying mean household income * households)
    for weighted_var in weighted_vars:
        data[weighted_var[0]] *= data[weighted_var[1]]

    # matrix multiplication interpolates to beats
    data = data.set_index(source_geo)
    output = crosswalk.dot(np.array(data))
    labels = data.columns.values.tolist()
    output = output.rename(columns=dict(zip(np.linspace(0,len(labels),len(labels),endpoint=False),labels)))

    # divide weighted variables by sum of weighting variables
    for weighted_var in weighted_vars:
        output[weighted_var[0]] /= sum(output[weighted_var[1]])

    # convert variances to margins of error
    for variable in margins:
        output[variable] = output[variable]**0.5

    return output

# for popweighted variables: 
# (1) calculate population of each fragment as an extensive variable
# (2) multiply FAKE_mean_income by population of each fragment ("total_FAKE_mean_income")
# (3) aggregate as above
# (4) divide total_FAKE_mean_income of each beat by population of each beat

# for margins of error:
# (1) square to get variance
# (2) aggregate as above
# (3) take square root to return to margin of error
# (Var(A+B)=Var(A)+Var(B)+2*Cov(A,B) and assume Cov(A,B) are uncorrelated for sampling error)

# Matrix approach for extensive variables:
# 1. Convert to array (sorted_block_groups x variables, contents = variable per block group)
# (add NAs for missing block groups)
# 2. Load crosswalk as array (sorted_beats x sorted_block_groups, contents = bg_frac)
# (bg_frac = overlap area / block group area)
# 3. Matrix multiplication: (sorted_beats x sorted_block_groups)*(sorted_block_groups x variables)= (sorted_beats x variables), contents = variable summed across block groups, weighted by fraction of each block group in beat

# Comprehensive matrix approach:
# 0. Square margins of error, multiply by weighting variable
# 1. Convert to array (sorted_block_groups x variables, contents = variable per block group) (add NAs for missing block groups)
# 2. Load crosswalk as array (sorted_beats x sorted_block_groups, contents = bg_frac)
# (bg_frac = overlap area / block group area)
# 3. Matrix multiplication: (sorted_beats x sorted_block_groups)*(sorted_block_groups x variables)= (sorted_beats x variables), contents = variable summed across block groups, weighted by fraction of each block group in beat (NAs for beats with missing block groups)
# 4. Extract results
# 5. Square root margins of error, divide weighted variables by sum of weighting variable