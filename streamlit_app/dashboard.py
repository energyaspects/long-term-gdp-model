import streamlit as st
import shooju
import pandas as pd
import plotly.io as pio
import plotly.express as px

pio.templates.default = "seaborn"

st.title('Macroeconomic Model Adjustment Dashboard')

st.write('This data app allows users to forecast GDP using different scenarios. The selection of the scenario'
         ' and the adjustment parameters can be selected and parsed through this app.')

st.header('Step 1: Refresh Data')

st.write('This step is important to guarantee data consistency and to make sure the latest data is being imported. '
         'Users can choose if they want to re-train the models or use existing models. It will return forecasts '
         'for the baseline scenario.')

st.selectbox('Select option', ['train & predict', 'predict only'])

st.header('Step 2: GDP Data Import & Visualisation')


def get_options(df, **region):
    if region:
        query = df.query(f"region=='{region[0]}'")
        country_list_options = list(query.country.unique())

    else:
        country_list_options = list(new_df.country.unique())

    return country_list_options


@st.cache
def load_data(query):
    sj = shooju.Connection(server='https://energyaspects.shooju.com/',
                           user='emilie.allen',
                           api_key='Z18A504hYJGbO3yzONbYNA9Hr1IJFdUvPFglTKDmhy6HYW3jnJ6vrEvNGgyuaANqU')

    df = sj.get_df('sid:users\emilie.allen\GDP\* scenario=Baseline', fields=['iso3c', 'Region', 'Country'],
                   max_points=-1)
    df = df.drop(columns='series_id').sort_values(by=['region', 'iso3c', 'date'])
    df.date = df.date.dt.date
    return df

@st.cache
def display_selection(chart, df):
    return st.plotly_chart(chart), st.write(df)


def add_year(n,year_selection):
    T, x = st.beta_columns(2)
    T = T.selectbox('Year', [year_selection], key=f'y{n}')
    x = x.number_input('Enter adj %', key=f'x{n}')
    return T, x


def add_year_country(n, y_selection, country_id):
    c, x, T = st.beta_columns(3)
    c = c.selectbox('Country', [country_id], key=f'c{n}')
    T = T.selectbox('Year', [y_selection], key=f'y{n}')
    x = x.number_input('Enter adj %', min_value=0, step=1, key=f'x{n}')
    return T, x, c


def country_selection(s_country=None, s_region=None):
    if s_country:
        s_country_list = s_country

    if s_region and not s_country:
        s_country_list = list(new_df.query(f'region == {region}').country.unique())

    else:
        s_country_list = list(new_df.country.unique())

    return s_country_list


def add_new_adjustment_widgets(scenario_country_list, adj_n):
    c2, T2, adj = st.beta_columns(3)
    c2 = c2.multiselect('Country', scenario_country_list, key=f'sow_my_c_ms_{adj_n}')
    T2 = T2.multiselect('Year', ['<year>'] + new_df.date.to_list(), key=f'sow_my_y_ms_{adj_n}')
    adj = adj.number_input('Enter adj %', min_value=0, step=1, key=f'my_sc{adj_n}')
    b, b0, b1, b2 = st.beta_columns(4)
    b2 = b2.button('Add new scenario', key=f'b2_{adj_n}')
    return b2


def scenario_options_widgets(n, s_country=None, s_region=None):
    adj_s = st.empty()
    my_scenario = st.selectbox('Select GDP growth adjustment scenario:', ['<select>', 'Single year', 'Multiple '
                                                                                                     'years',
                                                                            'All years'], 0, key=f'sow_sb_{n}')
    if my_scenario != '<select>' and my_scenario != 'Multiple years':
        scenario_country_list = country_selection(s_country, s_region)
        scenario_country_selection = st.multiselect('Country',['<select>'] + scenario_country_list, key=f'sow_my_c_ms_{n}')
        if scenario_country_selection:
            if my_scenario == 'Single year':
                T = st.selectbox('Year', ['<year>'] + list(new_df.date), key=f'sow_sy_sb_{n}')
                x = 0
                while x < len(scenario_country_selection):
                    x += 1
                    add_year_country(x, T, scenario_country_selection[x-1])

            if my_scenario == 'All years':
                T = st.selectbox('Year', ['<year>'] + list(new_df.date), key=f'sow_ay_sb_{n}')
                x = 0
                while x < len(scenario_country_selection):
                    x += 1
                    add_year_country(x, T, scenario_country_selection[x-1])

    if my_scenario == 'Multiple years':
        scenario_country_list = country_selection(s_country, s_region)
        adj_s = add_new_adjustment_widgets(scenario_country_list, 1)

    return adj_s

#@st.cache(suppress_st_warning=True)


new_df = load_data(query='sid:users\emilie.allen\GDP\* scenario=Baseline')

my_expander = st.beta_expander('Select viewing options')

with my_expander:
    region, country = st.beta_columns(2)
    region_list = list(new_df.region.unique())
    region = region.multiselect('Select Region', region_list, 'EUR')

    if region:
        chart_data = new_df.query(f"region in {region}")
        country_list = list(chart_data.country.unique())
        country = country.multiselect('Select Country', options=country_list)

        fig = px.line(chart_data, x="date", y="points", color="country", hover_name="country")
        display_df = pd.pivot_table(chart_data, index='country', values='points', columns='date')

        if country:
            chart_data = new_df.query(f"country in {country}")
            fig = px.line(chart_data, x="date", y="points", color="country", hover_name="country")
            display_df = pd.pivot_table(chart_data, index='country', values='points', columns='date')
            st.checkbox('Show model parameters')

    else:
        country_list = list(new_df.country.unique())
        country = country.multiselect('Select Country', options=country_list, default=country_list[0])

        if country:
            chart_data = new_df.query(f"country in {country}")
            fig = px.line(chart_data, x="date", y="points", color="country", hover_name="country")
            display_df = pd.pivot_table(chart_data, index='country', values='points', columns='date')
            st.checkbox('Show model parameters')

expander2 = st.beta_expander('Select scenario and enter adjustments')

with expander2:
    adj_option = st.selectbox('Select adjustment type:',
                              ['<select>', 'Apply a unique adjustment to the country selection',
                               'Apply individual adjustments to the country selection'])

    if adj_option == 'Apply a unique adjustment to the country selection' and (region or country):
        my_scenario = st.selectbox('Select GDP growth adjustment scenario:', ['<select>', 'Single year', 'Multiple '
                                                                                                         'years',
                                                                              'All years'], 0, key='sow_sb')
        if my_scenario == 'Single year':
            st.subheader('Scenario 1')
            T, x = st.beta_columns(2)
            T = T.selectbox('Year', ['<year>'] + list(new_df.date), key=f'sow_sb')
            x = x.slider('Adj %', min_value=0.0, max_value=5.0, step=0.5, format='%f', key='sow_sy_sl')

        if my_scenario == 'All years':
            x = st.slider('Adj %', min_value=0.0, max_value=5.0, step=0.5, format='%f', key='sow_ay_sl')

        if my_scenario == 'Multiple years':
            year_selection = st.multiselect('Select years', ['<year>'] + list(new_df.date), key='sow_my_ms')
            x = 0
            while x < len(year_selection):
                x += 1
                add_year(x, year_selection[x-1])

    if adj_option == 'Apply individual adjustments to the country selection':
        n = 1
        st.subheader(f'Scenario {n}')
        add_scenario = scenario_options_widgets(n, country, region)
        if add_scenario:
            n += 1
            st.subheader(f'Scenario {n}')
            scenario_options_widgets(200, country, region)




# container_cr = st.beta_container()
# container_cr.write("this is name-referenced container")
#
# container_cr2 = st.beta_container()
# container_cr2.write("this is name-referenced container")


#fig = px.line(new_df, x="date", y="points", color="country", hover_name="country")
#display_df = pd.pivot_table(new_df, index=new_df.country, values='points', columns='date')

st.plotly_chart(fig)
st.write(display_df)



#st.line_chart(display_df.T)
# query = st.text_input('Enter Shooju query:', 'e.g. "users\emilie.allen\GDP\LBY_GDP_baseline"')
#     if query:
#         new_df = load_data(query=f'{query}')
#     else:
#         new_df = load_data(query='sid:users\emilie.allen\GDP\* scenario=Baseline')
