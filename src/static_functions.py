import pandas as pd
from static_data import rename_dict

# Static Functions


def map_country_to_region(df: pd.DataFrame, region_df: pd.DataFrame) -> pd.DataFrame:
    """Return df with new column levels: region & sub_region."""

    df.columns = (pd.MultiIndex.from_tuples([(c[0],
                                              c[1],
                                              dict(zip(region_df.ISO3C, region_df.Region))[c[1]],
                                              dict(zip(region_df.ISO3C, region_df.Sub_regions_others))[c[1]])
                                             for c in df.columns]))
    df.columns.set_names(['country', 'iso3c', 'region', 'other_region'], inplace=True)
    return df


def add_other_regions_col_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with new column levels: country name & iso3c for new 'other_regions' series."""

    df.columns = (pd.MultiIndex.from_tuples(
        [(dict(zip(df.columns,
                   ['Other_Africa', 'Other_Asia', 'Other_Europe', 'Other_FSU', 'Other_Latam', 'Other_Middle']))[c],
          c, '', '') for c in df.columns]))
    return df


def add_regions_col_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with new column levels: country name & iso3c for new 'region' series."""

    df.columns = (
        pd.MultiIndex.from_tuples([(c, dict(zip(df.columns, ['AFR', 'AP', 'EU', 'FSU', 'LA', 'ME', 'NA', 'W']))[c],
                                    '', '') for c in df.columns]))
    return df


def country_name_index_formatting(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with formatted country names."""

    # Data cleaning
    df.index = df.index.set_levels(df.index.levels[0].str.replace(' ', '_'), level=0)
    df.index = df.index.set_levels(df.index.levels[0].to_series().replace(rename_dict), level=0)
    df.index = df.index.set_levels(df.index.levels[0].str.replace(r'St.', 'Saint', regex=True), level=0)
    return df


def get_sub_region_aggregates(df):
    covered_cty_df = df.loc(axis=1)[:, :, :,
                     ['Africa', 'Asia-Pacific', 'Europe', 'FSU', 'Latin_America', 'Middle_East']].groupby(level=3,
                                                                                                          axis=1).sum()
    other_region_df = df.loc(axis=1)[:, :, :, ['oafrc', 'oasia', 'oeuro', 'ofsu', 'olatam', 'omiddle']].groupby(
        level=3, axis=1).sum()

    # Add columns levels
    other_region_df = add_other_regions_col_levels(other_region_df)
    covered_cty_df = add_other_regions_col_levels(covered_cty_df)

    return other_region_df, covered_cty_df


def get_region_aggregates_standard(df):
    region_df = df.groupby(level=2, axis=1).sum()
    region_df = region_df.assign(World=region_df.sum(axis=1))

    return add_regions_col_levels(region_df)


def get_region_aggregates_multi(df, other_df):
    other_region_df = other_df.groupby(level=0, axis=1).sum()
    other_region_df.columns = other_region_df.columns.str.replace('Other_', "")
    other_region_df.rename({'Asia': 'Asia-Pacific', 'Latam': 'Latin_America', 'Middle': 'Middle_East'}, axis=1, inplace=True)

    region_df = df.groupby(level=2, axis=1).sum()
    region_df.update(other_region_df, overwrite=True)
    region_df = region_df.assign(World=region_df.sum(axis=1))

    return add_regions_col_levels(region_df)



