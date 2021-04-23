# Use this template to parse adjustments when running the GDP model

adjustment_dict = {
    'adjustment_summary_1': {
        'adj_type': 'adj_1',
        'adj_year': 2026,
        'adj_country': ['China', 'France'],
        'adj_value': 3
    }
}

"""
Provide percentage change, effective year, adjustment type and country name list for each adjustment (multiple 
adjustments allowed but they need to be specified under individual 'adjustment_summary' key.

Items description:
    - percentage change (adj_value): adjustment percentage applied to the GDP forecast
    - effective year: year to apply % change
    - country name: name of country used in scenario
    - adjustment type: adj_1 or adj_2

Types of adjustment:
    - Adjustment1 (adj_1): Adjust forecast on a specific year and smooth forecasts for subsequent years for a given 
      list of countries.
    - Adjustment2 (adj_2): Apply same adjustment on forecasts across all years from the effective year for a given
      list of countries.
      
Example:

adj_dict = {
    'adjustment_summary_1': {
        'adj_type': adj_1,
        'adj_year': 2021,
        'adj_country': ['China', 'France'],
        'adj_value': 2.6
    }
    'adjustment_summary_2': {
        'adj_type': adj_2,
        'adj_year': 2030,
        'adj_country': ['Germany'],
        'adj_value': -4
    }
}
   
"""