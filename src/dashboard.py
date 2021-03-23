import streamlit as st
import shooju
import pandas as pd
import plotly.express as px
import numpy as np
import os
from widget_dict import model_arguments_dict, model_adj_arguments_dict
from main import main_streamlit
import joblib

##
# Static Functions
##


@st.cache(suppress_st_warning=True)
def run_model(f_mdl_run_type):
    """
    Run the GDP model with selected run mode. Cache used to avoid systematic re-run.
    Clear cache to reset selection and re-run the model.
    """

    if f_mdl_run_type == 'train & predict':
        model_arguments_dict['train_model'] = True
        main_streamlit(model_arguments_dict=model_arguments_dict)

    elif f_mdl_run_type == 'predict only':
        model_arguments_dict['train_model'] = False
        main_streamlit(model_arguments_dict=model_arguments_dict)

    elif f_mdl_run_type == 'Not run':
        st.warning('Using current forecasts')


def adj_model_run(**kwargs):
    """Run GDP model using default 'Predict only' mode and adjustment(s)/scenario(s) summary."""
    main_streamlit(**kwargs)


# @st.cache(ttl=300)
@st.cache()
def load_data(n_query):
    """load baseline GDP growth data from SJ."""

    sj = shooju.Connection(server='https://energyaspects.shooju.com/',
                           user='emilie.allen',
                           api_key='Z18A504hYJGbO3yzONbYNA9Hr1IJFdUvPFglTKDmhy6HYW3jnJ6vrEvNGgyuaANqU')

    df = sj.get_df('sid:users\emilie.allen\GDP\* economic_property="global_macro_GDP_growth" scenario="Baseline"',
                   fields=['iso3c', 'Region', 'Country'],
                   max_points=-1)
    df = df.drop(columns='series_id').sort_values(by=['region', 'iso3c', 'date'])
    df.date = df.date.dt.date

    return df


# Keep the state of the button press between actions
def widgets_states():
    return {}

# # Widgets functions (dynamically create new widgets and selection values, updating the widget_values dictionary to
# keep track of states) #


def add_year(n, s_year_selection, s_country, s_region):
    n_country_list = country_selection(s_country, s_region)
    s_year, s_value = st.beta_columns(2)
    s_year = s_year.selectbox('Year', [s_year_selection], key=f'adj1_year_{n}')
    s_value = s_value.slider('Adj %', min_value=-5.0, max_value=5.0, step=0.5, value=0.0, format='%f', key=f'adj1_value_{n}')
    widget_values.update(
        {f'adj_summary_{n}': dict(adj_type='adj1', adj_country=n_country_list, adj_year=s_year, adj_value=s_value)})


def add_year_country(n, country_id):
    s_country, s_value, s_year = st.beta_columns(3)
    s_country = s_country.selectbox('Country', [country_id], key=f'c{n}')
    s_year = s_year.selectbox('Year', ['<year>'] + list(new_df.date), key=f'y{n}')
    s_value = s_value.slider('Adj %', min_value=-5.0, max_value=5.0, step=0.3, value=0.0, format='%f', key=f'x{n}')
    widget_values.update({f'adj_summary_{n}': dict(adj_type='adj3', adj_country=s_country, adj_year=s_year, adj_value=s_value)})


def country_selection(s_country, s_region):
    """
    Return a country list used in the widgets' selection values according to the user selection (region, country,
    none, region & country).
    """

    if s_country:
        s_country_list = s_country

    elif s_region and not s_country:
        s_country_list = list(new_df.query(f'region == {region}').country.unique())

    else:
        s_country_list = list(new_df.country.unique())

    return s_country_list


def add_new_adjustment_widgets(adj_n, n_country_list, n_year_selection):
    n_country, n_year, n_value = st.beta_columns(3)
    n_country = n_country.multiselect('Country', n_country_list, key=f'new_adj_wid_country_{adj_n}')
    n_year = n_year.selectbox('Year', [n_year_selection], key=f'adj1_year_{adj_n}')
    n_value = n_value.slider('Adj %', min_value=-5.0, max_value=5.0, step=0.5, value=0.0, format='%f', key=f'adj1_value_{adj_n}')
    widget_values.update(
        {f'adj_summary_{adj_n}': dict(adj_type='adj1', adj_country=n_country, adj_year=n_year, adj_value=n_value)})
    widget_values.update(dict(adj_type_1='is_true'))


def scenario_options_widgets(n, s_country, s_region):
    f_my_scenario = st.selectbox('Select GDP growth adjustment scenario:', ['<select>', 'Single/Multiple years',
                                                                            'All years'], 0, key=f'sow_sb_{n}')
    if f_my_scenario != '<select>':
        f_scenario_country_list = country_selection(s_country, s_region)

        if f_scenario_country_list:
            if my_scenario == 'All years':
                widget_values.update({'adj_type_3': 'is_true'})
                f_scenario_country_selection = st.multiselect('Country', ['<select>'] + f_scenario_country_list,
                                                              key=f'sow_my_c_ms_{n}')
                if f_scenario_country_selection:
                    x = 0
                    while x < len(f_scenario_country_selection):
                        x += 1
                        add_year_country(x, f_scenario_country_selection[x - 1])

            if f_my_scenario == 'Single/Multiple years':
                f_year_selection = st.multiselect('Select years', ['<year>'] + list(new_df.date), key=f'my_{n}')
                widget_values.update(dict(adj_type_1='is_true'))
                if f_year_selection:
                    x = 0
                    while x < len(f_year_selection):
                        x += 1
                        add_new_adjustment_widgets(x, f_scenario_country_list, f_year_selection[x - 1])

##
# The Streamlit Application Design
##


st.title('Macroeconomic Model Adjustment Dashboard')

st.write('This data app allows users to forecast GDP using different scenarios. The selection of the scenario'
         ' and the adjustment parameters can be selected and parsed through this app.')
###
# Part I
###

st.header('Step 1: Refresh Data')

st.write('This step is important to guarantee data consistency and to make sure the latest data is being imported. '
         'Users can choose if they want to re-train the models or use existing models. It will return forecasts '
         'for the baseline scenario.')

st.warning('Always clear cache at the beginning of a session!')
clear_cache = st.button('Clear cache')

if clear_cache:
    st.spinner()
    with st.spinner(text='In progress'):
        st.caching.clear_cache()
        st.success('Cache cleared successfully!')

st.subheader('Select how you wish to run the model (re-train or predict only)')
mdl_run_type = st.selectbox('Select option', ['<select>', 'train & predict', 'predict only', 'Not run'])


run_selection = run_model(mdl_run_type)


st.subheader('Select where you wish to store the outputs in SJ')
output_path = st.selectbox('Select option', ['<select>', 'PROD', 'USER AREA'])

# if output_path == 'PROD':
#     # model_arguments_dict["sj_path"] = 'prod sids'
#
# if output_path == 'USER AREA':
#     # model_arguments_dict["sj_path"] = f"users\{os.environ['SHOOJU_USER']}\GDP"

widget_values = widgets_states()

###
# Part II
###

if mdl_run_type != '<select>':
    st.header('Step 2: GDP Growth Adjustment(s)')
    'Apply adjustment(s) to the baseline predictions. Adjustment(s) can be applied to an entire forecast period ' \
    ' or from a selected year. These adjustments can be compound (multiple years/countries). You can view the ' \
    'adjustment(s) locally before amending the output values.'

    new_df = load_data(1)

    my_expander = st.beta_expander('Select country selection')

    with my_expander:
        region, country = st.beta_columns(2)
        region_list = list(new_df.region.unique())
        region = region.multiselect('Select Region', region_list, 'EUR')

        if region:
            chart_data = new_df.query(f"region in {region}")
            country_list = list(chart_data.country.unique())
            country = country.multiselect('Select Country', options=country_list, key='region_selection')
            fig = px.line(chart_data, x="date", y="points", color="country", hover_name="country")
            display_df = pd.pivot_table(chart_data, index='country', values='points', columns='date')

            if country:
                chart_data = new_df.query(f"country in {country}")
                fig = px.line(chart_data, x="date", y="points", color="country", hover_name="country")
                display_df = pd.pivot_table(chart_data, index='country', values='points', columns='date')
                param = st.checkbox('Show model parameters')
                if param:
                    iso3_list = list(chart_data.iso3c.unique())
                    for iso3c in iso3_list:
                        iso, p = st.beta_columns(2)
                        m_param = joblib.load(f'model_params/{iso3c}_model.pkl')
                        iso = iso.write(f'{iso3c} (p,d,q):')
                        p = p.write(m_param)

        else:
            country_list = list(new_df.country.unique())
            country = country.multiselect('Select Country', options=country_list, default=country_list[0])

            if country:
                chart_data = new_df.query(f"country in {country}")
                fig = px.line(chart_data, x="date", y="points", color="country", hover_name="country")
                display_df = pd.pivot_table(chart_data, index='country', values='points', columns='date')
                st.checkbox('Show model parameters')

    expander2 = st.beta_expander('Select options and enter adjustments')

    with expander2:
        adj_option = st.selectbox('Select adjustment type:',
                                  ['<select>', 'Apply a unique adjustment to the country selection',
                                   'Apply individual adjustments to the country selection'])

        if adj_option == 'Apply a unique adjustment to the country selection' and (region or country):
            my_scenario = st.selectbox('Select GDP growth adjustment:', ['<select>', 'Single/Multiple years',
                                                                         'All years'], 0, key='sow_sb')
            scenario_country_list = country_selection(country, region)

            if my_scenario == 'Single/Multiple years':
                st.subheader('Adjustment board')
                year_selection = st.multiselect('Select years', ['<year>'] + list(new_df.date), key='sow_my_ms')
                x = 0
                while x < len(year_selection):
                    x += 1
                    add_year(f'adjustment_{x}', year_selection[x - 1], country, region)

                if len(year_selection) > 0:
                    apply_adj, nul, push_adj = st.beta_columns(3)
                    apply_adj = apply_adj.button('Apply adjustment locally', key='apply_button1')
                    adj_sum_dict = {key: value for key, value in widget_values.items() if
                                    key.startswith('adj_summary_adjustment')}
                    if apply_adj:
                        adj_chart_data = chart_data.copy()
                        for k in adj_sum_dict.keys():
                            adj_c = adj_sum_dict[k]['adj_country']
                            adj_y = adj_sum_dict[k]['adj_year']
                            adj_v = adj_sum_dict[k]['adj_value']
                            for c in adj_c:
                                adj_df = adj_chart_data[
                                    (adj_chart_data.date >= adj_y) & (adj_chart_data.country == c)]
                                adj_df.loc[:, 'points'] = adj_df.loc[:, 'points'].apply(
                                    lambda p: (1 + 0.01 * adj_v) * p)
                                adj_chart_data.update(adj_df)
                                adj_df['country'] = f'{c}_adj_{adj_y}'
                                chart_data = chart_data.append(adj_df)
                        fig = px.line(chart_data, x="date", y="points", color="country", hover_name="country")
                        display_df = pd.pivot_table(chart_data, index='country', values='points', columns='date')

                    push_adj = push_adj.button('Overwrite output', key='apush_button1')
                    if push_adj:
                        with st.spinner(text='In progress'):
                            model_adj_arguments_dict["scenario"] = 'baseline'
                            model_adj_arguments_dict["adjustment"] = True
                            adj_model_run(model_arguments_dict=model_adj_arguments_dict, adjustment_dict=adj_sum_dict)
                            new_df = load_data(2)
                            st.success('Done')

            if my_scenario == 'All years':
                base_year, adj = st.beta_columns(2)
                base_year = base_year.selectbox('Define year 0', ['<year>'] + list(new_df.date), key=f'adj2_all_year')
                adj = adj.slider('Adj %', min_value=-5.0, max_value=5.0, step=0.5, value=0.0, format='%f',
                                 key='adj2_value')
                widget_values.update({'adj_summary_adjustment': {'adj_type': 'adj3',
                                                                 'adj_country': scenario_country_list,
                                                                 'adj_year': base_year, 'adj_value': adj}})
                if base_year != '<select>':
                    apply_adj, nul, push_adj = st.beta_columns(3)
                    apply_adj = apply_adj.button('Apply adjustment locally', key='apply_button2')
                    adj_sum_dict = {key: value for key, value in widget_values.items() if
                                    key.startswith('adj_summary_adjustment')}

                    if apply_adj:
                        for c in country:
                            adj_df = chart_data[(chart_data.date >= base_year) & (chart_data.country == c)]
                            adj_arr = np.array(range(0, adj_df[adj_df.date >= base_year].shape[0])).reshape(
                                adj_df[adj_df.date >= base_year].shape[0], 1)
                            test = (1 + 0.01 * adj) ** adj_arr * adj_df[adj_df.date >= base_year][['points']]
                            adj_df['points'] = test
                            adj_df['country'] = f'{c}_adj'
                            chart_data = chart_data.append(adj_df)
                            fig = px.line(chart_data, x="date", y="points", color="country", hover_name="country")
                            display_df = pd.pivot_table(chart_data, index='country', values='points', columns='date')

                    push_adj = push_adj.button('Overwrite output', key='apush_button1')
                    if push_adj:
                        with st.spinner(text='In progress'):
                            model_adj_arguments_dict["scenario"] = 'baseline'
                            model_adj_arguments_dict["adjustment"] = True
                            adj_model_run(model_arguments_dict=model_adj_arguments_dict, adjustment_dict=adj_sum_dict)
                            new_df = load_data(3)
                            st.success('Done')

        if adj_option == 'Apply individual adjustments to the country selection':
            st.subheader(f'Adjustment')
            my_scenario = st.selectbox('Select GDP growth adjustment scenario:', ['<select>', 'Single/Multiple years',
                                                                                  'All years'], 0, key=f'allen3')

            if my_scenario == 'Single/Multiple years':
                scenario_country_list = country_selection(country, region)
                year_selection = st.multiselect('Select years', ['<year>'] + list(new_df.date), key=f'my')
                if year_selection:
                    x = 0
                    while x < len(year_selection):
                        x += 1
                        add_new_adjustment_widgets(f'adjustment_{x}', scenario_country_list, year_selection[x - 1])
                    if widget_values['adj_type_1']:
                        apply_adj, nul, push_adj = st.beta_columns(3)
                        apply_adj = apply_adj.button('Apply adjustment locally', key='apply_button3')
                        adj_sum_dict = {key: value for key, value in widget_values.items() if
                                        key.startswith('adj_summary_adjustment')}
                        if apply_adj:
                            adj_chart_data = chart_data.copy()
                            for k in adj_sum_dict.keys():
                                adj_c = adj_sum_dict[k]['adj_country']
                                adj_y = adj_sum_dict[k]['adj_year']
                                adj_v = adj_sum_dict[k]['adj_value']
                                for c in adj_c:
                                    adj_df = adj_chart_data[
                                        (adj_chart_data.date >= adj_y) & (adj_chart_data.country == c)]
                                    adj_df.loc[:, 'points'] = adj_df.loc[:, 'points'].apply(
                                        lambda p: (1 + 0.01 * adj_v) * p)
                                    adj_chart_data.update(adj_df)
                                    adj_df['country'] = f'{c}_adj_{adj_y}'
                                    chart_data = chart_data.append(adj_df)
                            fig = px.line(chart_data, x="date", y="points", color="country", hover_name="country")
                            display_df = pd.pivot_table(chart_data, index='country', values='points', columns='date')
                        push_adj = push_adj.button('Overwrite output', key='push_button1')
                        if push_adj:
                            with st.spinner(text='In progress'):
                                model_adj_arguments_dict["scenario"] = 'baseline'
                                model_adj_arguments_dict["adjustment"] = True
                                adj_model_run(model_arguments_dict=model_adj_arguments_dict, adjustment_dict=adj_sum_dict)
                                new_df = load_data(4)
                                st.success('Done')

            if my_scenario == 'All years':
                widget_values.update({'adj_type_3': 'is_true'})
                scenario_country_list = country_selection(country, region)
                scenario_country_selection = st.multiselect('Country', ['<select>'] + scenario_country_list,
                                                            key=f'allen8')
                if scenario_country_selection:
                    x = 0
                    while x < len(scenario_country_selection):
                        x += 1
                        add_year_country(f'adjustment_{x}', scenario_country_selection[x - 1])

                    if widget_values['adj_type_3']:
                        apply_adj, nul, push_adj = st.beta_columns(3)
                        apply_adj = apply_adj.button('Apply adjustment locally', key='apply_button4')
                        adj_sum_dict = {key: value for key, value in widget_values.items() if
                                        key.startswith('adj_summary_adjustment')}
                        if apply_adj:
                            for k in adj_sum_dict.keys():
                                adj_c = adj_sum_dict[k]['adj_country']
                                adj_y = adj_sum_dict[k]['adj_year']
                                adj_v = adj_sum_dict[k]['adj_value']
                                adj_df = chart_data[
                                    (chart_data.date >= adj_y) & (chart_data.country == adj_c)]
                                adj_arr = np.array(range(0, adj_df[adj_df.date >= adj_y].shape[0])).reshape(
                                    adj_df[adj_df.date >= adj_y].shape[0], 1)
                                test = (1 + 0.01 * adj_v) ** adj_arr * adj_df[adj_df.date >= adj_y][['points']]
                                adj_df['points'] = test
                                adj_df['country'] = f'{adj_c}_adj'
                                chart_data = chart_data.append(adj_df)
                            fig = px.line(chart_data, x="date", y="points", color="country", hover_name="country")
                            display_df = pd.pivot_table(chart_data, index='country', values='points', columns='date')
                        push_adj = push_adj.button('Overwrite output', key='push_button1')
                        if push_adj:
                            with st.spinner(text='In progress'):
                                model_adj_arguments_dict["scenario"] = 'baseline'
                                model_adj_arguments_dict["adjustment"] = True
                                adj_model_run(model_arguments_dict=model_adj_arguments_dict, adjustment_dict=adj_sum_dict)
                                new_df = load_data(5)
                                st.success('Done')
    st.plotly_chart(fig)
    st.write(display_df)

    ###
    # Part III
    ###

    st.header('Step 3: GDP Growth Scenario(s) ')
    'Apply a scenario to the final baseline values. If the baseline predictions were adjusted above, they will ' \
    'be the latest values used in this section. Please choose a meaningfull name for your scenario as it will be used' \
    ' to create specific SIDs.'

    scenario_name = st.text_input('Enter scenario name (no space (_) e.g: China_low_growth):')
    widget_values.update({'scenario_name': scenario_name})

    if scenario_name:
        new_df = load_data(6)

        my_expander3 = st.beta_expander('Select country selection')

        with my_expander3:
            region, country = st.beta_columns(2)
            region_list = list(new_df.region.unique())
            region = region.multiselect('Select Region', region_list, 'EUR', key='scenario_ms1')

            if region:
                chart_data = new_df.query(f"region in {region}")
                country_list = list(chart_data.country.unique())
                country = country.multiselect('Select Country', options=country_list, key='region_selection2')
                fig = px.line(chart_data, x="date", y="points", color="country", hover_name="country")
                display_df = pd.pivot_table(chart_data, index='country', values='points', columns='date')

                if country:
                    chart_data = new_df.query(f"country in {country}")
                    fig = px.line(chart_data, x="date", y="points", color="country", hover_name="country")
                    display_df = pd.pivot_table(chart_data, index='country', values='points', columns='date')

            else:
                country_list = list(new_df.country.unique())
                country = country.multiselect('Select Country', options=country_list, default=country_list[0],
                                              key='scenario_ms2')

                if country:
                    chart_data = new_df.query(f"country in {country}")
                    fig = px.line(chart_data, x="date", y="points", color="country", hover_name="country")
                    display_df = pd.pivot_table(chart_data, index='country', values='points', columns='date')

        expander4 = st.beta_expander('Select options and enter scenario adjustments')

        with expander4:
            adj_option = st.selectbox('Select adjustment type:',
                                      ['<select>', 'Apply a unique scenario adjustment to the country selection',
                                       'Apply individual scenario adjustments to the country selection'],
                                      key='scenario_sb1')

            if adj_option == 'Apply a unique scenario adjustment to the country selection' and (region or country):
                my_scenario = st.selectbox('Select GDP growth adjustment:', ['<select>', 'Single/Multiple years',
                                                                             'All years'], 0, key='sow_sb2')
                scenario_country_list = country_selection(country, region)

                if my_scenario == 'Single/Multiple years':
                    st.subheader('Adjustment board')
                    year_selection = st.multiselect('Select years', ['<year>'] + list(new_df.date), key='sow_my_ms2')
                    x = 0
                    while x < len(year_selection):
                        x += 1
                        add_year(f'scenario_{x}', year_selection[x - 1], country, region)

                    if len(year_selection) > 0:
                        apply_adj, nul, push_adj = st.beta_columns(3)
                        apply_adj = apply_adj.button('Apply adjustment locally', key='s_apply_button8')
                        adj_sum_dict = {key: value for key, value in widget_values.items() if
                                        key.startswith('adj_summary_scenario')}
                        if apply_adj:
                            adj_chart_data = chart_data.copy()
                            for k in adj_sum_dict.keys():
                                adj_c = adj_sum_dict[k]['adj_country']
                                adj_y = adj_sum_dict[k]['adj_year']
                                adj_v = adj_sum_dict[k]['adj_value']
                                for c in adj_c:
                                    adj_df = adj_chart_data[
                                        (adj_chart_data.date >= adj_y) & (adj_chart_data.country == c)]
                                    adj_df.loc[:, 'points'] = adj_df.loc[:, 'points'].apply(
                                        lambda p: (1 + 0.01 * adj_v) * p)
                                    adj_chart_data.update(adj_df)
                                    adj_df['country'] = f'{c}_adj_{adj_y}'
                                    chart_data = chart_data.append(adj_df)
                            fig = px.line(chart_data, x="date", y="points", color="country", hover_name="country")
                            display_df = pd.pivot_table(chart_data, index='country', values='points', columns='date')

                        push_adj = push_adj.button('Overwrite output', key='s_apush_button2')
                        if push_adj:
                            with st.spinner(text='In progress'):
                                model_adj_arguments_dict["scenario"] = 'scenario'
                                model_adj_arguments_dict["adjustment"] = True
                                model_adj_arguments_dict["scenario_name"] = widget_values['scenario_name']
                                adj_model_run(model_arguments_dict=model_adj_arguments_dict, adjustment_dict=adj_sum_dict)
                                st.success('Done')

                if my_scenario == 'All years':
                    base_year, adj = st.beta_columns(2)
                    base_year = base_year.selectbox('Define year 0', ['<year>'] + list(new_df.date), key=f'adj2_all_year2')
                    adj = adj.slider('Adj %', min_value=-5.0, max_value=5.0, step=0.5, value=0.0, format='%f',
                                     key='adj2_value2')
                    widget_values.update({'adj_summary_scenario': {'adj_type': 'adj3',
                                                                   'adj_country': scenario_country_list,
                                                                   'adj_year': base_year, 'adj_value': adj}})
                    if base_year != '<select>':
                        apply_adj, nul, push_adj = st.beta_columns(3)
                        apply_adj = apply_adj.button('Apply adjustment locally', key='s_apply_button2')
                        adj_sum_dict = {key: value for key, value in widget_values.items() if
                                        key.startswith('adj_summary_scenario')}

                        if apply_adj:
                            for c in country:
                                adj_df = chart_data[(chart_data.date >= base_year) & (chart_data.country == c)]
                                adj_arr = np.array(range(0, adj_df[adj_df.date >= base_year].shape[0])).reshape(
                                    adj_df[adj_df.date >= base_year].shape[0], 1)
                                test = (1 + 0.01 * adj) ** adj_arr * adj_df[adj_df.date >= base_year][['points']]
                                adj_df['points'] = test
                                adj_df['country'] = f'{c}_adj'
                                chart_data = chart_data.append(adj_df)
                                fig = px.line(chart_data, x="date", y="points", color="country", hover_name="country")
                                display_df = pd.pivot_table(chart_data, index='country', values='points', columns='date')

                        push_adj = push_adj.button('Overwrite output', key='s_apush_button3')
                        if push_adj:
                            with st.spinner(text='In progress'):
                                model_adj_arguments_dict["scenario"] = 'scenario'
                                model_adj_arguments_dict["adjustment"] = True
                                model_adj_arguments_dict["scenario_name"] = widget_values['scenario_name']
                                adj_model_run(model_arguments_dict=model_adj_arguments_dict, adjustment_dict=adj_sum_dict)
                                st.success('Done')

            if adj_option == 'Apply individual scenario adjustments to the country selection':
                st.subheader(f'Scenario Adjustment')
                my_scenario = st.selectbox('Select GDP growth adjustment scenario:', ['<select>', 'Single/Multiple years',
                                                                                      'All years'], 0, key=f's_allen3')
                if my_scenario == 'Single/Multiple years':
                    scenario_country_list = country_selection(country, region)
                    year_selection = st.multiselect('Select years', ['<year>'] + list(new_df.date), key=f'my2')
                    if year_selection:
                        x = 0
                        while x < len(year_selection):
                            x += 1
                            add_new_adjustment_widgets(f'scenario_{x}', scenario_country_list, year_selection[x - 1])
                        if widget_values['adj_type_1']:
                            apply_adj, nul, push_adj = st.beta_columns(3)
                            apply_adj = apply_adj.button('Apply adjustment locally', key='s_apply_button3')
                            adj_sum_dict = {key: value for key, value in widget_values.items() if
                                            key.startswith('adj_summary_scenario')}
                            if apply_adj:
                                adj_chart_data = chart_data.copy()
                                for k in adj_sum_dict.keys():
                                    adj_c = adj_sum_dict[k]['adj_country']
                                    adj_y = adj_sum_dict[k]['adj_year']
                                    adj_v = adj_sum_dict[k]['adj_value']
                                    for c in adj_c:
                                        adj_df = adj_chart_data[
                                            (adj_chart_data.date >= adj_y) & (adj_chart_data.country == c)]
                                        adj_df.loc[:, 'points'] = adj_df.loc[:, 'points'].apply(
                                            lambda p: (1 + 0.01 * adj_v) * p)
                                        adj_chart_data.update(adj_df)
                                        adj_df['country'] = f'{c}_adj_{adj_y}'
                                        chart_data = chart_data.append(adj_df)
                                fig = px.line(chart_data, x="date", y="points", color="country", hover_name="country")
                                display_df = pd.pivot_table(chart_data, index='country', values='points', columns='date')
                            push_adj = push_adj.button('Overwrite output', key='push_button1')
                            if push_adj:
                                with st.spinner(text='In progress'):
                                    model_adj_arguments_dict["scenario"] = 'scenario'
                                    model_adj_arguments_dict["adjustment"] = True
                                    model_adj_arguments_dict["scenario_name"] = widget_values['scenario_name']
                                    adj_model_run(model_arguments_dict=model_adj_arguments_dict, adjustment_dict=adj_sum_dict)
                                    st.success('Done')

                if my_scenario == 'All years':
                    widget_values.update({'adj_type_3': 'is_true'})
                    scenario_country_list = country_selection(country, region)
                    scenario_country_selection = st.multiselect('Country', ['<select>'] + scenario_country_list,
                                                                key=f'sallen8')
                    if scenario_country_selection:
                        x = 0
                        while x < len(scenario_country_selection):
                            x += 1
                            add_year_country(f'scenario_{x}', scenario_country_selection[x - 1])

                        if widget_values['adj_type_3']:
                            apply_adj, nul, push_adj = st.beta_columns(3)
                            apply_adj = apply_adj.button('Apply adjustment locally', key='s_apply_button4')
                            adj_sum_dict = {key: value for key, value in widget_values.items() if
                                            key.startswith('adj_summary_scenario')}
                            if apply_adj:
                                for k in adj_sum_dict.keys():
                                    adj_c = adj_sum_dict[k]['adj_country']
                                    adj_y = adj_sum_dict[k]['adj_year']
                                    adj_v = adj_sum_dict[k]['adj_value']
                                    adj_df = chart_data[
                                        (chart_data.date >= adj_y) & (chart_data.country == adj_c)]
                                    adj_arr = np.array(range(0, adj_df[adj_df.date >= adj_y].shape[0])).reshape(
                                        adj_df[adj_df.date >= adj_y].shape[0], 1)
                                    test = (1 + 0.01 * adj_v) ** adj_arr * adj_df[adj_df.date >= adj_y][['points']]
                                    adj_df['points'] = test
                                    adj_df['country'] = f'{adj_c}_adj'
                                    chart_data = chart_data.append(adj_df)
                                fig = px.line(chart_data, x="date", y="points", color="country", hover_name="country")
                                display_df = pd.pivot_table(chart_data, index='country', values='points', columns='date')
                            push_adj = push_adj.button('Overwrite output', key='s_push_button1')
                            if push_adj:
                                with st.spinner(text='In progress'):
                                    model_adj_arguments_dict["scenario"] = 'scenario'
                                    model_adj_arguments_dict["adjustment"] = True
                                    model_adj_arguments_dict["scenario_name"] = widget_values['scenario_name']
                                    adj_model_run(model_arguments_dict=model_adj_arguments_dict, adjustment_dict=adj_sum_dict)
                                    st.success('Done')
            if adj_option != '<select>':
                st.plotly_chart(fig)
                st.write(display_df)

    ###
    # Part IV
    ###

    st.header('Step 4: Run energy demand model with adjustment(s)/scenario(s)')
    "Run the energy demand model with the adjustment(s)/scenario(s) defined above. Select model's output data you " \
    " wish to view here."

    st.checkbox('Run energy demand model')

    st.text_input('Enter shooju query:')


