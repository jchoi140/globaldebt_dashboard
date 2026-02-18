import streamlit as st
import pandas as pd
import math
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from fetch_indicator_metadata import get_or_fetch_metadata

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Sovereign debt dashboard',
    page_icon=':chart_with_upwards_trend:',
    layout='wide',
)
st.markdown(
    """
    <style>
    .sidebar-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: inherit;
        text-align: center;
        color: grey;
        font-size: 0.8em;
        padding: 0.5rem;
        # background: white;
    }
    .sidebar-footer a {
        color: grey;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def format_axis_values(df, value_col='Value', base_label='Value', additional_dfs=None, indicator_props=None):
    """
    Dynamically format values for readable y-axis labels.
    
    Uses 'unit' from indicator metadata to determine scaling:
    - Percent: no scaling, y-axis shows %
    - Million: multiply by 1e6, Plotly auto-formats with M
    - Billion: multiply by 1e9, Plotly auto-formats with B
    - Thousand: multiply by 1e3, Plotly auto-formats with K
    """
    if additional_dfs is None:
        additional_dfs = []
    
    unit = ''
    if indicator_props:
        unit = indicator_props.get('unit', '') or ''
    
    # Handle percentages/rates
    if 'Percent' in unit or '%' in unit or '(%)' in base_label:
        # y_title = f'{base_label} (%)'
        y_title = '%'
        return value_col, y_title, ',.2f'
    
    # Scale values UP based on unit so Plotly auto-formats tick labels
    if 'Million' in unit or 'million' in unit:
        df[value_col] = df[value_col] * 1e6
        for add_df in additional_dfs:
            if len(add_df) > 0 and value_col in add_df.columns:
                add_df[value_col] = add_df[value_col] * 1e6
    elif 'Billion' in unit or 'billion' in unit:
        df[value_col] = df[value_col] * 1e9
        for add_df in additional_dfs:
            if len(add_df) > 0 and value_col in add_df.columns:
                add_df[value_col] = add_df[value_col] * 1e9
    elif 'Thousand' in unit or 'thousand' in unit:
        df[value_col] = df[value_col] * 1e3
        for add_df in additional_dfs:
            if len(add_df) > 0 and value_col in add_df.columns:
                add_df[value_col] = add_df[value_col] * 1e3
    
    y_title = f'{base_label}'
    return value_col, y_title, None

def is_rate_indicator(indicator_props):
    """Check if indicator is a rate/percentage based on metadata unit."""
    if not indicator_props:
        return False
    unit = indicator_props.get('unit', '') or ''
    return 'Percent' in unit or '%' in unit


def apply_plotly_style(fig, y_title=None, height=450):
    """Apply consistent styling to Plotly figures."""
    fig.update_layout(
        plot_bgcolor='white',
        height=height,
        xaxis=dict(
            showgrid=False,
            dtick=5,  # Show tick every 5 years
            title=None
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor='lightgrey', 
            gridwidth=0.5,
            title=y_title
        ),
        legend=dict(
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='left',
            x=1.02
        )
    )
    return fig


# -----------------------------------------------------------------------------
# Data Loading Functions
# -----------------------------------------------------------------------------

@st.cache_data
def get_debt_interest_data():
    """Grab external debt ratio interest data from a CSV file."""
    DATA_FILENAME = Path(__file__).parent/'data/external_debt_ratio_interest_cleaned.csv'
    raw_debt_df = pd.read_csv(DATA_FILENAME)

    debt_df = raw_debt_df.rename(columns={
        'debtor': 'Country Code',
        'year': 'Year',
        'value': 'Value',
        'Series': 'Indicator Name',
        'Counterpart-Area': 'Counterpart Area'
    })

    debt_df['Year'] = pd.to_numeric(debt_df['Year'])
    debt_df['Value'] = pd.to_numeric(debt_df['Value'], errors='coerce')

    return debt_df


@st.cache_data
def get_dod_color_map():
    """Define color mapping for DOD creditor types and series names."""
    return {
        # Blues for official/bilateral/multilateral
        'GG, multilateral': '#1f77b4',
        'GG, bilateral': '#6baed6',
        'OPS, official creditors': '#9ecae1',
        'CB, official creditors': '#c6dbef',
        # darker blue for DAC creditor
        'DAC member': '#08519c',  
        # Purples for private creditors, bonds, and commercial banks
        'PNG, bonds': '#7b2d8e',
        'PNG, commercial banks and other creditors': '#a855c9',
        'OPS, private creditors': '#c492d6',
        'CB, private creditors': '#e0cce8',
        # Reds for IMF/SDR/short-term
        'Use of IMF credit': '#fb6a4a',      # light red
        'SDR allocations': '#de2d26',        # medium red
        'External debt stocks, short-term': '#a50f15',  # dark red

        # creditor_type colors #
        'official creditors': '#1f77b4',
        'private creditors': '#a855c9',
        'bondholders': '#7b2d8e',
        'IMF': '#D25353',
        'other': 'grey'
    }

@st.cache_data
def get_available_indicators():
    """Get list of available interest payment indicators from the mapping file."""
    MAPPING_FILE = Path(__file__).parent / 'data' / 'interest_payments' / '_indicator_mapping.csv'

    if not MAPPING_FILE.exists():
        st.error(f"Indicator mapping file not found: {MAPPING_FILE}")
        return []

    mapping_df = pd.read_csv(MAPPING_FILE)
    return mapping_df['indicator'].tolist()

@st.cache_data
def get_dod_data():
    """Load Debt Outstanding and Disbursed (DOD) data."""
    DATA_DIR = Path(__file__).parent / 'data'
    
    parquet_file = DATA_DIR / 'DOD.parquet'
    csv_file = DATA_DIR / 'DOD.csv'
    
    if parquet_file.exists():
        return pd.read_parquet(parquet_file)
    elif csv_file.exists():
        return pd.read_csv(csv_file)
    else:
        return None


@st.cache_data
def load_indicator_data(indicator_name):
    """Load data for a specific indicator from its split file."""
    MAPPING_FILE = Path(__file__).parent / 'data' / 'interest_payments' / '_indicator_mapping.csv'
    DATA_DIR = Path(__file__).parent / 'data' / 'interest_payments'

    mapping_df = pd.read_csv(MAPPING_FILE)
    filename_match = mapping_df[mapping_df['indicator'] == indicator_name]['filename'].values

    if len(filename_match) == 0:
        st.error(f"Indicator not found: {indicator_name}")
        return pd.DataFrame()

    filename = filename_match[0] + '.parquet'
    filepath = DATA_DIR / filename

    if not filepath.exists():
        st.error(f"Data file not found: {filepath}")
        return pd.DataFrame()

    raw_df = pd.read_parquet(filepath)

    df = raw_df.rename(columns={
        'debtor': 'Country Code',
        'year': 'Year',
        'value': 'Value',
        'Series': 'Indicator Name',
        'Counterpart-Area': 'Counterpart Area'
    })

    df['Year'] = pd.to_numeric(df['Year'])
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

    return df

@st.cache_data
def load_creditor_colors():
    """Load creditor color mapping from CSV file."""
    COLOR_FILE = Path(__file__).parent / 'metadata' / 'creditor_codes_with_colors.csv'
    if COLOR_FILE.exists():
        color_mapping_df = pd.read_csv(COLOR_FILE)
        color_mapping = dict(zip(color_mapping_df['Counterpart-Area Name'], color_mapping_df['color']))
        color_mapping['Other/Unreported'] = '#D3D3D3'
        color_mapping['Bondholders'] = '#C71585'
        color_mapping['World'] = '#ff7f0e'
        return color_mapping
    else:
        return {
            'Other/Unreported': '#D3D3D3',
            'Bondholders': '#C71585',
            'China': '#8B0000',
        }

@st.cache_data
def load_debtor_names():
    """Load debtor code to name mapping from CSV file."""
    DEBTOR_FILE = Path(__file__).parent / 'metadata' / 'debtor_names.csv'
    if DEBTOR_FILE.exists():
        debtor_df = pd.read_csv(DEBTOR_FILE)
        return dict(zip(debtor_df['id'], debtor_df['value']))
    else:
        return {}

def format_country_option(code, debtor_names):
    """Format country code with full name for dropdowns."""
    name = debtor_names.get(code, '')
    if name:
        return f'{code} - {name}'
    return code

def get_country_name(code, debtor_names):
    """Get full country name from code."""
    return debtor_names.get(code, code)

# -----------------------------------------------------------------------------
# DOD Chart Helper Function (Plotly version)
# -----------------------------------------------------------------------------

def create_dod_chart_plotly(dod_df, iso, color_variable='creditor_type', from_year=None, to_year=None, show_percentage=False):
    """
    Create a stacked bar chart with line overlays for DOD data using Plotly.
    """
    color_map = get_dod_color_map()
    
    data = dod_df[dod_df['debtor_code'] == iso].copy()
    debtor_name = dod_df[dod_df['debtor_code'] == iso]['debtor_name'].iloc[0]

    if from_year is not None:
        data = data[data['year'] >= from_year]
    if to_year is not None:
        data = data[data['year'] <= to_year]
    
    if len(data) == 0:
        return None
    
    
    # Total WLD values for line plot
    total_df = data[data['creditor_code'] == 'WLD']
    total_grouped = total_df.groupby(['year'], as_index=False)['value'].sum()
    
    # WLD data only for Long-term (exclude IMFcredit_SDR.csv and short_term.csv)
    LT_df = data[
        (data['creditor_code'] == 'WLD') &
        (~data['source_file'].isin(['IMFcredit_SDR.csv', 'short_term.csv']))
    ]
    LT_grouped = LT_df.groupby(['year'], as_index=False)['value'].sum()
    
    # Include short_term.csv (with WLD) if it only has WLD non-zero values, otherwise exclude WLD
    remaining_df = data[
        (data['creditor_code'] != 'WLD') | 
        (data['source_file'] == 'IMFcredit_SDR.csv')
    ]

    short_term_wld = data[
        (data['source_file'] == 'short_term.csv') & 
        (data['creditor_code'] == 'WLD')
    ]
    remaining_df = pd.concat([remaining_df, short_term_wld])
    
    grouped = remaining_df.groupby(['year', color_variable], as_index=False)['value'].sum()

    # Calculate percentage if needed
    if show_percentage:
        year_totals = grouped.groupby('year')['value'].sum().reset_index()
        year_totals = year_totals.rename(columns={'value': 'year_total'})
        grouped = grouped.merge(year_totals, on='year')
        grouped['pct_value'] = (grouped['value'] / grouped['year_total']) * 100
        y_col = 'pct_value'
        y_title = 'Share of Total (%)'
    else:
        y_col = 'value'
        y_title = 'current USD'
    
    # Define order from bottom to top
    category_order = [
        'official creditors',
        'private creditors',
        'bondholders',
        'IMF',
        'DAC member',
        'GG, multilateral',
        'GG, bilateral',
        'OPS, official creditors',
        'CB, official creditors',
        'OPS, private creditors',
        'CB, private creditors',
        'PNG, commercial banks and other creditors',
        'PNG, bonds',
        'SDR allocations', 
        'Use of IMF credit', 
        'External debt stocks, short-term'
    ]
    
    # Create a stacked bar plot with custom order and colors
    fig = px.bar(
        grouped, 
        x='year', 
        y=y_col,
        color=color_variable, 
        title=f'{debtor_name}',
        barmode='relative',
        category_orders={color_variable: category_order},
        color_discrete_map=color_map
    )
    
    # Add line plots only for absolute values (not percentage view)
    if not show_percentage:
        # Add line plot for total
        fig.add_scatter(
            x=total_grouped['year'], 
            y=total_grouped['value'], 
            mode='lines', 
            name='total', 
            line=dict(color='#C3110C', width=3)
        )
        
        # Add line plot for LT_grouped
        fig.add_scatter(
            x=LT_grouped['year'], 
            y=LT_grouped['value'], 
            mode='lines', 
            name='long-term', 
            line=dict(color='orange', width=3)
        )
    
    fig.update_layout(
        xaxis_title=None, 
        yaxis_title=y_title,
        plot_bgcolor='white',
        yaxis=dict(showgrid=True, gridcolor='lightgrey', gridwidth=0.5),
        height=500,
        legend=dict(
            title='creditor type',
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='left',
            x=1.02
        )
    )
    
    return fig


# -----------------------------------------------------------------------------
# Load Data
# -----------------------------------------------------------------------------

debt_df = get_debt_interest_data()
dod_df = get_dod_data()
creditor_color_mapping = load_creditor_colors()
debtor_names = load_debtor_names()

# Initialize session state for selected countries
if 'selected_countries' not in st.session_state:
    st.session_state.selected_countries = ['LIC', 'JAM', 'MEX']

# -----------------------------------------------------------------------------
# Draw the actual page
# -----------------------------------------------------------------------------

'''
# :chart_with_upwards_trend: Sovereign Debt Dashboard

Visualize debt-related indicators across different countries, years, and creditor counterparties.
'''

# Sidebar navigation
st.sidebar.markdown("## Navigation")
selected_tab = st.sidebar.radio(
    "Select View",
    ["External Debt", "Debt Service", "Interest Payments"],
    label_visibility = "collapsed"
)
st.sidebar.markdown("---")

# -----------------------------------------------------------------------------
# TAB 1: External Debt
# -----------------------------------------------------------------------------

if selected_tab == "External Debt":
    # st.sidebar.markdown("## Select")

    indicators = sorted(debt_df['Indicator Name'].unique())
    # Remove specific indicators
    indicators_to_remove = [
        'Interest payments on external debt (% of GNI)',
        'Interest payments on external debt (% of exports of goods, services and primary income)'
    ]
    indicators = [ind for ind in indicators if ind not in indicators_to_remove]
    
    # Add DOD indicator if data is available
    if dod_df is not None:
        if 'Debt Outstanding and Disbursed' not in indicators:
            indicators = ['Debt Outstanding and Disbursed'] + indicators
    
    counterpart_areas = sorted(debt_df['Counterpart Area'].unique())

    selected_indicator = st.sidebar.selectbox(
        'Select Indicator',
        indicators,
        index=0 if len(indicators) > 0 else 0
    )

    is_dod_indicator = selected_indicator == 'Debt Outstanding and Disbursed'

    # -----------------------------------------------------------------------------
    # DOD Indicator
    # -----------------------------------------------------------------------------
    if is_dod_indicator and dod_df is not None:
        st.subheader('Debt Outstanding and Disbursed (DOD)')
        
        dod_countries = sorted(dod_df['debtor_code'].unique())
        
        dod_min_year = int(dod_df['year'].min())
        dod_max_year = int(dod_df['year'].max())
        
        from_year, to_year = st.sidebar.slider(
            'Select Time Range',
            min_value=dod_min_year,
            max_value=dod_max_year,
            value=[dod_min_year, dod_max_year],
            key='dod_year_slider'
        )
        
        # Filter session state countries to only those available in DOD data
        default_dod_countries = [c for c in st.session_state.selected_countries if c in dod_countries]

        selected_dod_countries = st.sidebar.multiselect(
            'Select countries, region or income group',
            dod_countries,
            default_dod_countries,
            format_func=lambda x: format_country_option(x, debtor_names),
            key='dod_countries'
        )

        # Update session state when selection changes
        st.session_state.selected_countries = selected_dod_countries

        # Chart type toggle for percentage view
        st.sidebar.markdown("## Chart Type")
        show_dod_percentage = st.sidebar.toggle('Show as % share of total', value=False, key='dod_pct_toggle')

        # Chart type toggle for creditor breakdown
        show_detailed_breakdown = st.sidebar.toggle('Show detailed breakdown', value=False, key='dod_detail_toggle')
        color_variable = 'series_name' if show_detailed_breakdown else 'creditor_type'
        
        with st.expander("ℹ️ About this indicator", expanded=False):
            st.write("""
            **Debt Outstanding and Disbursed (DOD)** represents the total amount of external debt 
            that has been disbursed and is currently outstanding. This includes both long-term 
            and short-term debt across various creditor types.
            
            - **Total** (red line): Sum of all debt including short-term and IMF credits
            - **Long-term** (orange line): Excludes short-term debt and IMF credit/SDR allocations
            - **Stacked bars**: Breakdown by creditor type or series
            """)
            st.caption("*Source: World Bank International Debt Statistics*")
        
        if len(selected_dod_countries) > 0:
            for iso in selected_dod_countries:
                country_dod = dod_df[dod_df['debtor_code'] == iso]
                
                if len(country_dod) > 0:
                    fig = create_dod_chart_plotly(dod_df, iso, color_variable, from_year, to_year, show_dod_percentage)
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f'No DOD data available for {iso}')
                else:
                    st.info(f'No DOD data available for {iso} in selected time range')
                
                st.markdown('---')
            
            # Summary statistics
            st.subheader('Summary Statistics')
            
            filtered_dod = dod_df[
                (dod_df['year'] >= from_year) & 
                (dod_df['year'] <= to_year)
            ]
            
            cols = st.columns(min(len(selected_dod_countries), 5))
            
            for i, iso in enumerate(selected_dod_countries):
                with cols[i % len(cols)]:
                    country_data = filtered_dod[
                        (filtered_dod['debtor_code'] == iso) &
                        (filtered_dod['creditor_code'] == 'WLD')
                    ]
                    
                    if len(country_data) > 0:
                        latest_year = country_data['year'].max()
                        first_year = country_data['year'].min()
                        
                        latest_value = country_data[country_data['year'] == latest_year]['value'].sum()
                        first_value = country_data[country_data['year'] == first_year]['value'].sum()
                        
                        if latest_value >= 1e9:
                            display_value = f'${latest_value/1e9:.2f}B'
                        elif latest_value >= 1e6:
                            display_value = f'${latest_value/1e6:.2f}M'
                        else:
                            display_value = f'${latest_value:,.0f}'
                        
                        if first_value > 0:
                            change = ((latest_value - first_value) / first_value)
                            change_str = f'{change:+.1f}'
                        else:
                            change_str = 'n/a'
                        
                        st.metric(
                            label=f'{iso} ({int(latest_year)})',
                            value=display_value,
                            delta=change_str,
                            delta_color='inverse'
                        )
                    else:
                        st.metric(
                            label=f'{iso}',
                            value='n/a',
                            delta='n/a',
                            delta_color='off'
                        )
        else:
            st.info('Select countries in the sidebar to view DOD data')
    
    # -----------------------------------------------------------------------------
    # non-DOD indicators
    # -----------------------------------------------------------------------------
    else:
        # Check if indicator is a ratio (% of GNI) - no counterpart selection needed
        is_ratio_indicator = '% of' in selected_indicator
        
        if is_ratio_indicator:
            # For ratio indicators, default to World and don't show selector
            selected_counterpart = 'World'
        else:
            selected_counterpart = st.sidebar.selectbox(
                'Select Creditor',
                counterpart_areas,
                index=counterpart_areas.index('World') if 'World' in counterpart_areas else 0
            )

        filtered_by_series = debt_df[
            (debt_df['Indicator Name'] == selected_indicator) &
            (debt_df['Counterpart Area'] == selected_counterpart)
        ]

        min_value = int(filtered_by_series['Year'].min())
        max_value = int(filtered_by_series['Year'].max())

        from_year, to_year = st.sidebar.slider(
            'Select Time Range',
            min_value=min_value,
            max_value=max_value,
            value=[min_value, max_value])

        countries = sorted(filtered_by_series['Country Code'].unique())

        if not len(countries):
            st.warning("No data available for selected filters")

        # Filter session state countries to only those available
        default_countries = [c for c in st.session_state.selected_countries if c in countries]

        selected_countries = st.sidebar.multiselect(
            'Select countries',
            countries,
            default_countries,
            format_func=lambda x: format_country_option(x, debtor_names)
            )

        # Update session state when selection changes
        st.session_state.selected_countries = selected_countries

        filtered_debt_df = filtered_by_series[
            (filtered_by_series['Country Code'].isin(selected_countries))
            & (filtered_by_series['Year'] <= to_year)
            & (from_year <= filtered_by_series['Year'])
        ].copy()

        st.subheader(f'{selected_indicator}')

        indicator_code = None
        if len(filtered_by_series) > 0 and 'series' in filtered_by_series.columns:
            indicator_code = filtered_by_series['series'].iloc[0]

        if indicator_code:
            metadata = get_or_fetch_metadata(indicator_code)
            if metadata:
                with st.expander("ℹ️ About this indicator", expanded=False):
                    st.write(metadata['definition'])
                    if metadata.get('unit'):
                        st.caption(f"**Unit of measure:** {metadata['unit']}")
                    if metadata.get('source'):
                        st.caption(f"*Source: {metadata['source']}*")

        # Check if interest payment overlay toggle should be provided
        interest_overlay_indicator = None
        add_interest_line = False
        if selected_indicator == 'External debt stocks (% of GNI)':
            add_interest_line = st.sidebar.toggle('Add interest payment (% of GNI)', value=False, key='add_int_gni')
            interest_overlay_indicator = 'Interest payments on external debt (% of GNI)'
        elif 'External debt stocks (% of exports' in selected_indicator:
            add_interest_line = st.sidebar.toggle('Add interest payment (% of exports)', value=False, key='add_int_exports')
            interest_overlay_indicator = 'Interest payments on external debt (% of exports of goods, services and primary income)'
        
        if len(filtered_debt_df) > 0:
            # Get metadata for formatting
            metadata = get_or_fetch_metadata(indicator_code) if indicator_code else None
            
            value_col, y_title, fmt = format_axis_values(
                filtered_debt_df,
                value_col='Value',
                base_label=selected_indicator,
                indicator_props=metadata
            )
            
            # Create Plotly line chart
            fig = px.line(
                filtered_debt_df,
                x='Year',
                y=value_col,
                color='Country Code',
                markers=False
            )
            
            # Add interest payment overlay if toggled
            if add_interest_line and interest_overlay_indicator:
                interest_df_overlay = debt_df[
                    (debt_df['Indicator Name'] == interest_overlay_indicator) &
                    (debt_df['Counterpart Area'] == selected_counterpart) &
                    (debt_df['Country Code'].isin(selected_countries)) &
                    (debt_df['Year'] <= to_year) &
                    (from_year <= debt_df['Year'])
                ].copy()
                
                if len(interest_df_overlay) > 0:
                    # Get colors directly from the existing figure traces
                    country_colors = {}
                    for trace in fig.data:
                        if trace.name in selected_countries:
                            country_colors[trace.name] = trace.line.color
                    
                    # Add dashed lines for each country's interest payments
                    for country in selected_countries:
                        country_interest = interest_df_overlay[interest_df_overlay['Country Code'] == country]
                        if len(country_interest) > 0 and country in country_colors:
                            fig.add_scatter(
                                x=country_interest['Year'],
                                y=country_interest['Value'],
                                mode='lines',
                                name=f'{country} (Interest)',
                                line=dict(color=country_colors[country], width=2, dash='dash'),
                                showlegend=True
                            )
            
            fig = apply_plotly_style(fig, y_title=y_title)
            st.plotly_chart(fig, use_container_width=True)
        
        # Always show interest payments as separate chart (if applicable indicator selected)
        if interest_overlay_indicator:
            interest_df = debt_df[
                (debt_df['Indicator Name'] == interest_overlay_indicator) &
                (debt_df['Counterpart Area'] == selected_counterpart) &
                (debt_df['Country Code'].isin(selected_countries)) &
                (debt_df['Year'] <= to_year) &
                (from_year <= debt_df['Year'])
            ].copy()
            
            if len(interest_df) > 0:
                st.subheader(f'{interest_overlay_indicator}')
                
                # Get metadata for interest indicator
                interest_code = None
                if 'series' in interest_df.columns:
                    interest_code = interest_df['series'].iloc[0]
                interest_metadata = get_or_fetch_metadata(interest_code) if interest_code else None
                
                if interest_metadata:
                    with st.expander("ℹ️ About this indicator", expanded=False):
                        st.write(interest_metadata['definition'])
                        if interest_metadata.get('unit'):
                            st.caption(f"**Unit of measure:** {interest_metadata['unit']}")
                        if interest_metadata.get('source'):
                            st.caption(f"*Source: {interest_metadata['source']}*")
                
                int_value_col, int_y_title, int_fmt = format_axis_values(
                    interest_df,
                    value_col='Value',
                    base_label=interest_overlay_indicator,
                    indicator_props=interest_metadata
                )
                
                fig_interest = px.line(
                    interest_df,
                    x='Year',
                    y=int_value_col,
                    color='Country Code',
                    markers=False
                )
                fig_interest = apply_plotly_style(fig_interest, y_title=int_y_title)
                st.plotly_chart(fig_interest, use_container_width=True)

        first_year_data = filtered_by_series[filtered_by_series['Year'] == from_year]
        last_year_data = filtered_by_series[filtered_by_series['Year'] == to_year]

        st.subheader(f'change from {from_year} to {to_year}', divider='gray')

        ''

        cols = st.columns(4)

        for i, country in enumerate(selected_countries):
            col = cols[i % len(cols)]

            with col:
                first_values = first_year_data[first_year_data['Country Code'] == country]['Value'].values
                last_values = last_year_data[last_year_data['Country Code'] == country]['Value'].values

                if len(last_values) > 0:
                    last_value = last_values[0]

                    if not math.isnan(last_value):
                        if len(first_values) > 0 and not math.isnan(first_values[0]):
                            first_value = first_values[0]
                            change = f'{last_value - first_value:+.2f}pp'
                            delta_color = 'inverse'
                        else:
                            change = 'n/a'
                            delta_color = 'off'

                        st.metric(
                            label=f'{country}',
                            value=f'{last_value:.2f}%',
                            delta=change,
                            delta_color=delta_color
                        )
                    else:
                        st.metric(label=f'{country}', value='n/a', delta='n/a', delta_color='off')
                else:
                    st.metric(label=f'{country}', value='n/a', delta='n/a', delta_color='off')

# -----------------------------------------------------------------------------
# TAB 2: Debt Service
# -----------------------------------------------------------------------------

elif selected_tab == "Debt Service":
    # st.sidebar.markdown("## Select")
    
    # Load the debt service indicator
    debt_service_indicator = 'Debt service on external debt, long-term (TDS, current US$)'
    
    int_df = load_indicator_data(debt_service_indicator)
    
    # Get indicator code for metadata lookup
    indicator_code = None
    if len(int_df) > 0 and 'series' in int_df.columns:
        indicator_code = int_df['series'].iloc[0]
    
    # Get metadata
    metadata = get_or_fetch_metadata(indicator_code) if indicator_code else None

    world_data = int_df[int_df['Counterpart Area'] == 'World']

    all_years = sorted(world_data['Year'].dropna().unique())
    if len(all_years) > 0:
        ds_min_year = int(min(all_years))
        ds_max_year = int(max(all_years))

        ds_from_year, ds_to_year = st.sidebar.slider(
            'Time Range',
            min_value=ds_min_year,
            max_value=ds_max_year,
            value=[ds_min_year, ds_max_year],
            key='ds_year_slider'
        )
    else:
        ds_from_year = 1980
        ds_to_year = 2023

    all_creditor_data = int_df[int_df['Counterpart Area'] != 'World']

    countries_with_creditor_data = all_creditor_data[
        (all_creditor_data['Value'].notna()) & 
        (all_creditor_data['Value'] != 0)
    ]['Country Code'].unique()

    has_creditor_breakdown = len(countries_with_creditor_data) > 0

    if has_creditor_breakdown:
        # Filter session state countries to only those available
        default_creditor_countries = [c for c in st.session_state.selected_countries if c in countries_with_creditor_data]

        selected_creditor_countries = st.sidebar.multiselect(
            'Countries',
            sorted(countries_with_creditor_data),
            default=default_creditor_countries,
            format_func=lambda x: format_country_option(x, debtor_names),
            key='ds_countries'
        )

        # Update session state when selection changes
        st.session_state.selected_countries = selected_creditor_countries
    else:
        selected_creditor_countries = []

    if len(selected_creditor_countries) > 0:
        st.subheader(f'{debt_service_indicator}')

        if metadata:
            with st.expander("ℹ️ About this indicator", expanded=False):
                st.write(metadata['definition'])
                if metadata.get('unit'):
                    st.caption(f"**Unit of measure:** {metadata['unit']}")
                if metadata.get('source'):
                    st.caption(f"*Source: {metadata['source']}*")

        world_filtered = world_data[
            (world_data['Country Code'].isin(selected_creditor_countries)) &
            (world_data['Year'] <= ds_to_year) &
            (ds_from_year <= world_data['Year']) &
            (world_data['Value'].notna())
        ].copy()

        if len(world_filtered) > 0:
            value_col, y_title, fmt = format_axis_values(
                world_filtered, 
                value_col='Value', 
                base_label=debt_service_indicator, 
                indicator_props=metadata
            )
            fig = px.line(world_filtered, x='Year', y=value_col, color='Country Code', markers=True)
            fig = apply_plotly_style(fig, y_title=y_title)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('No World aggregate data available for selected countries')

        ''

        st.subheader('Creditor Breakdown by Country')

        st.sidebar.markdown("## Chart Type")
        show_percentage = st.sidebar.toggle('Show as % share of total', value=False, key='ds_pct_toggle')

        creditor_data = all_creditor_data[
            (all_creditor_data['Country Code'].isin(selected_creditor_countries)) &
            (all_creditor_data['Year'] <= ds_to_year) &
            (ds_from_year <= all_creditor_data['Year'])
        ]

        creditor_data = creditor_data[(creditor_data['Value'].notna()) & (creditor_data['Value'] != 0)].copy()

        if len(creditor_data) > 0:
            creditor_grouped = creditor_data.groupby(['Year', 'Country Code', 'Counterpart Area'], as_index=False)['Value'].sum()

            for country in selected_creditor_countries:
                country_data = creditor_grouped[creditor_grouped['Country Code'] == country].copy()

                if len(country_data) > 0:
                    st.markdown(f'##### **{get_country_name(country, debtor_names)}**')

                    creditor_sum = country_data.groupby('Year')['Value'].sum().reset_index()
                    creditor_sum['Type'] = 'Reported Creditors'

                    country_world_raw = world_data[
                        (world_data['Country Code'] == country) &
                        (world_data['Year'] <= ds_to_year) &
                        (ds_from_year <= world_data['Year']) &
                        (world_data['Value'].notna())
                    ].copy()
                    country_world = country_world_raw[['Year', 'Value']].copy()
                    country_world['Type'] = 'World Total'

                    merged = pd.merge(
                        country_world.rename(columns={'Value': 'World_Value'}),
                        creditor_sum.rename(columns={'Value': 'Creditor_Sum'}),
                        on='Year',
                        how='left'
                    )
                    merged['Creditor_Sum'] = merged['Creditor_Sum'].fillna(0)
                    merged['Residual'] = merged['World_Value'] - merged['Creditor_Sum']

                    residual_df = merged[merged['Residual'] > 0][['Year', 'Residual']].copy()
                    residual_df['Counterpart Area'] = 'Other/Unreported'
                    residual_df = residual_df.rename(columns={'Residual': 'Value'})

                    combined_data = pd.concat([country_data, residual_df], ignore_index=True)

                    if show_percentage:
                        year_totals = combined_data.groupby('Year')['Value'].sum().reset_index()
                        year_totals = year_totals.rename(columns={'Value': 'Year_Total'})
                        combined_data = combined_data.merge(year_totals, on='Year')
                        combined_data['Pct_Value'] = (combined_data['Value'] / combined_data['Year_Total']) * 100
                        value_col = 'Pct_Value'
                        y_title = 'Share of Total (%)'
                    else:
                        line_data = country_world[['Year', 'Value']].copy()
                        value_col, y_title, fmt = format_axis_values(
                            combined_data,
                            value_col='Value',
                            base_label=debt_service_indicator,
                            additional_dfs=[line_data],
                            indicator_props=metadata
                        )

                    # Sort creditors by total value for legend ordering
                    creditor_totals = combined_data.groupby('Counterpart Area')['Value'].sum().sort_values(ascending=False)
                    sorted_creditors = creditor_totals.index.tolist()
                    
                    # Move Other/Unreported to top of stack
                    if 'Other/Unreported' in sorted_creditors:
                        sorted_creditors.remove('Other/Unreported')
                        sorted_creditors.append('Other/Unreported')

                    # Create stacked bar chart
                    fig = px.bar(
                        combined_data,
                        x='Year',
                        y=value_col,
                        color='Counterpart Area',
                        barmode='relative',
                        category_orders={'Counterpart Area': sorted_creditors},
                        color_discrete_map=creditor_color_mapping
                    )

                    # Add World total line for comparison (only for absolute values)
                    if not show_percentage:
                        fig.add_scatter(
                            x=line_data['Year'],
                            y=line_data[value_col] if value_col in line_data.columns else line_data['Value'],
                            mode='lines',
                            name='World Total',
                            line=dict(color='grey', width=2, dash='dash')
                        )

                    fig = apply_plotly_style(fig, y_title=y_title, height=450)
                    st.plotly_chart(fig, use_container_width=True)

                    if not show_percentage:
                        avg_residual_pct = (merged['Residual'].sum() / merged['World_Value'].sum() * 100) if merged['World_Value'].sum() > 0 else 0
                        if avg_residual_pct > 5:
                            merged['residual_pct'] = (merged['Residual'] / merged['World_Value'] * 100).fillna(0)
                            years_with_gaps = merged[merged['residual_pct'] > 5]['Year'].tolist()

                            if len(years_with_gaps) > 5:
                                years_str = f"{min(years_with_gaps)}-{max(years_with_gaps)}"
                            else:
                                years_str = ', '.join([str(int(y)) for y in years_with_gaps])

                            st.caption(f"⚠️ Note: {avg_residual_pct:.1f}% of total debt service is from unreported/other creditors (shown in gray). Grey dashed line shows World total. Significant gaps (>5%) in years: {years_str}")

            ''

            st.subheader('Top Creditors by Total Debt Service')

            for country in selected_creditor_countries:
                country_data = creditor_grouped[creditor_grouped['Country Code'] == country]
                top_creditors = country_data.groupby('Counterpart Area')['Value'].sum().sort_values(ascending=False).head(10)

                if len(top_creditors) > 0:
                    st.write(f"**{get_country_name(country, debtor_names)}**")
                    
                    top_creditors_df = top_creditors.reset_index()
                    
                    # Determine scale factor from metadata unit
                    unit = metadata.get('unit', '') if metadata else ''
                    if 'Million' in unit or 'million' in unit:
                        scale_factor = 1e6
                    elif 'Billion' in unit or 'billion' in unit:
                        scale_factor = 1e9
                    elif 'Thousand' in unit or 'thousand' in unit:
                        scale_factor = 1e3
                    else:
                        scale_factor = 1
                    
                    def format_value(x):
                        x_scaled = x * scale_factor
                        if x_scaled >= 1e9:
                            return f"${x_scaled/1e9:.2f}B"
                        elif x_scaled >= 1e6:
                            return f"${x_scaled/1e6:.2f}M"
                        elif x_scaled >= 1e3:
                            return f"${x_scaled/1e3:.2f}K"
                        else:
                            return f"${x_scaled:,.0f}"
                    
                    top_creditors_df['Total Debt Service (US$)'] = top_creditors_df['Value'].apply(format_value)
                    
                    st.dataframe(
                        top_creditors_df[['Counterpart Area', 'Total Debt Service (US$)']].rename(columns={'Counterpart Area': 'Creditor'}),
                        hide_index=True,
                        use_container_width=False
                    )
        else:
            st.info('No creditor breakdown data available for selected countries in this time range')

    else:
        st.info('Select countries in the sidebar to view debt service data')

# -----------------------------------------------------------------------------
# TAB 2: Interest Payments by Creditor
# -----------------------------------------------------------------------------

elif selected_tab == "Interest Payments":
    # st.sidebar.markdown("## Select")

    interest_indicators = sorted(get_available_indicators())

    # Remove unwanted indicators (moved to other tabs or shown as overlays)
    indicators_to_remove = [
        # 'Average interest on new external debt commitments, official (%)',
        # 'Average interest on new external debt commitments, private (%)',
        'Debt service on external debt, long-term (TDS, current US$)',
    ]
    interest_indicators = [ind for ind in interest_indicators if ind not in indicators_to_remove]


    selected_int_indicator = st.sidebar.selectbox(
        'Select Indicator',
        interest_indicators,
        index=0 if len(interest_indicators) > 0 else 0
    )

    int_df = load_indicator_data(selected_int_indicator)
    
    # Get indicator code for metadata lookup
    indicator_code = None
    if len(int_df) > 0 and 'series' in int_df.columns:
        indicator_code = int_df['series'].iloc[0]
    
    # Get metadata (contains 'unit' for format_axis_values)
    metadata = get_or_fetch_metadata(indicator_code) if indicator_code else None

    world_data = int_df[int_df['Counterpart Area'] == 'World']

    all_years = sorted(world_data['Year'].dropna().unique())
    if len(all_years) > 0:
        int_min_year = int(min(all_years))
        int_max_year = int(max(all_years))

        int_from_year, int_to_year = st.sidebar.slider(
            'Time Range',
            min_value=int_min_year,
            max_value=int_max_year,
            value=[int_min_year, int_max_year],
            key='int_year_slider'
        )
    else:
        int_from_year = 1980
        int_to_year = 2023

    all_creditor_data = int_df[int_df['Counterpart Area'] != 'World']

    countries_with_creditor_data = all_creditor_data[
        all_creditor_data['Value'].notna()
    ]['Country Code'].unique()

    has_creditor_breakdown = len(countries_with_creditor_data) > 0

    if has_creditor_breakdown:
        # Filter session state countries to only those available
        default_creditor_countries = [c for c in st.session_state.selected_countries if c in countries_with_creditor_data]

        selected_creditor_countries = st.sidebar.multiselect(
            'Select countries',
            sorted(countries_with_creditor_data),
            default=default_creditor_countries,
            format_func=lambda x: format_country_option(x, debtor_names)
        )

        # Update session state when selection changes
        st.session_state.selected_countries = selected_creditor_countries
    else:
        selected_creditor_countries = []

    if not has_creditor_breakdown:
        if metadata:
            with st.expander("ℹ️ About this indicator", expanded=False):
                st.write(metadata['definition'])
                if metadata.get('unit'):
                    st.caption(f"**Unit:** {metadata['unit']}")
                if metadata.get('source'):
                    st.caption(f"*Source: {metadata['source']}*")

        world_countries = world_data['Country Code'].unique()

        # Filter session state countries to only those available
        default_world_countries = [c for c in st.session_state.selected_countries if c in world_countries]

        selected_countries_world = st.sidebar.multiselect(
            'Select countries',
            sorted(world_countries),
            default=default_world_countries,
            format_func=lambda x: format_country_option(x, debtor_names),
            key='countries_world_only'
        )

        # Update session state when selection changes
        st.session_state.selected_countries = selected_countries_world

        if len(selected_countries_world) > 0:
            world_display = world_data[
                (world_data['Country Code'].isin(selected_countries_world)) &
                (world_data['Year'] <= int_to_year) &
                (int_from_year <= world_data['Year']) &
                (world_data['Value'].notna())
            ].copy()

            if len(world_display) > 0:
                value_col, y_title, fmt = format_axis_values(
                    world_display, 
                    value_col='Value', 
                    base_label=selected_int_indicator, 
                    indicator_props=metadata
                )
                fig = px.line(world_display, x='Year', y=value_col, color='Country Code', markers=False)
                fig = apply_plotly_style(fig, y_title=y_title)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info('No data available for selected countries in this time range')


    elif len(selected_creditor_countries) > 0:
        st.subheader(f'{selected_int_indicator}')

        if metadata:
            with st.expander("ℹ️ About this indicator", expanded=False):
                st.write(metadata['definition'])
                if metadata.get('unit'):
                    st.caption(f"**Unit of measure:** {metadata['unit']}")
                if metadata.get('source'):
                    st.caption(f"*Source: {metadata['source']}*")


    # Check interest rate overlay toggles should be provided
        official_overlay = None
        private_overlay = None
        if selected_int_indicator == 'Average interest on new external debt commitments (%)':
            add_official_line = st.sidebar.toggle('Add average interest, official creditors (%)', value=False, key='add_int_official')
            add_private_line = st.sidebar.toggle('Add average interest, private creditors (%)', value=False, key='add_int_private')
            if add_official_line:
                official_overlay = 'Average interest on new external debt commitments, official (%)'
            if add_private_line:
                private_overlay = 'Average interest on new external debt commitments, private (%)'

        world_filtered = world_data[
            (world_data['Country Code'].isin(selected_creditor_countries)) &
            (world_data['Year'] <= int_to_year) &
            (int_from_year <= world_data['Year']) &
            (world_data['Value'].notna())
        ].copy()

        if len(world_filtered) > 0:
            value_col, y_title, fmt = format_axis_values(
                world_filtered, 
                value_col='Value', 
                base_label=selected_int_indicator, 
                indicator_props=metadata
            )
            fig = px.line(world_filtered, x='Year', y=value_col, color='Country Code', markers=False)
            
            # Add official interest rate overlay if toggled
            if official_overlay:
                official_df = load_indicator_data(official_overlay)
                if len(official_df) > 0:
                    official_world = official_df[
                        (official_df['Counterpart Area'] == 'World') &
                        (official_df['Country Code'].isin(selected_creditor_countries)) &
                        (official_df['Year'] <= int_to_year) &
                        (int_from_year <= official_df['Year']) &
                        (official_df['Value'].notna())
                    ].copy()
                    
                    if len(official_world) > 0:
                        # Get colors directly from the existing figure traces
                        country_colors = {}
                        for trace in fig.data:
                            if trace.name in selected_creditor_countries:
                                country_colors[trace.name] = trace.line.color
                        
                        for country in selected_creditor_countries:
                            country_official = official_world[official_world['Country Code'] == country]
                            if len(country_official) > 0 and country in country_colors:
                                fig.add_scatter(
                                    x=country_official['Year'],
                                    y=country_official['Value'],
                                    mode='lines',
                                    name=f'{country} (Official)',
                                    line=dict(color=country_colors[country], width=2, dash='dash'),
                                    showlegend=True
                                )
            
            # Add private interest rate overlay if toggled
            if private_overlay:
                private_df = load_indicator_data(private_overlay)
                if len(private_df) > 0:
                    private_world = private_df[
                        (private_df['Counterpart Area'] == 'World') &
                        (private_df['Country Code'].isin(selected_creditor_countries)) &
                        (private_df['Year'] <= int_to_year) &
                        (int_from_year <= private_df['Year']) &
                        (private_df['Value'].notna())
                    ].copy()
                    
                    if len(private_world) > 0:
                        # Get colors directly from the existing figure traces
                        country_colors = {}
                        for trace in fig.data:
                            if trace.name in selected_creditor_countries:
                                country_colors[trace.name] = trace.line.color
                        
                        for country in selected_creditor_countries:
                            country_private = private_world[private_world['Country Code'] == country]
                            if len(country_private) > 0 and country in country_colors:
                                fig.add_scatter(
                                    x=country_private['Year'],
                                    y=country_private['Value'],
                                    mode='lines',
                                    name=f'{country} (Private)',
                                    line=dict(color=country_colors[country], width=2, dash='dot'),
                                    showlegend=True
                                )
            
            fig = apply_plotly_style(fig, y_title=y_title)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('No World aggregate data available for selected countries')

        ''

        # Check if this is a rate indicator using the helper function
        if is_rate_indicator(metadata):
            st.subheader('Interest Rates by Creditor')

            creditor_data = all_creditor_data[
                (all_creditor_data['Country Code'].isin(selected_creditor_countries)) &
                (all_creditor_data['Year'] <= int_to_year) &
                (int_from_year <= all_creditor_data['Year'])
            ]

            creditor_data = creditor_data[(creditor_data['Value'].notna()) & (creditor_data['Value'] != 0)].copy()

            if len(creditor_data) > 0:
                for country in selected_creditor_countries:
                    st.markdown(f'##### **{get_country_name(country, debtor_names)}**')

                    country_creditor_data = creditor_data[creditor_data['Country Code'] == country]

                    # Only include creditors in dropdown that have non-null, non-zero values
                    creditors_with_values = country_creditor_data[
                        (country_creditor_data['Value'].notna()) & 
                        (country_creditor_data['Value'] != 0)
                    ]['Counterpart Area'].unique()

                    available_creditors = sorted(
                        [creditor for creditor in creditors_with_values if creditor != 'Bondholders']
                    )

                    if len(available_creditors) > 0:
                        creditor_options = ['Select'] + available_creditors
                        selected_creditor = st.selectbox(
                            'Select Creditor',
                            creditor_options,
                            key=f'creditor_{country}'
                        )

                        world_filtered_country = world_data[
                            (world_data['Country Code'] == country) &
                            (world_data['Year'] <= int_to_year) &
                            (int_from_year <= world_data['Year']) &
                            (world_data['Value'].notna())
                        ].copy()
                        
                        country_world_data = world_filtered_country[['Year', 'Value']].copy()
                        country_world_data['Type'] = 'World Average'

                        bondholders_data = country_creditor_data[
                            country_creditor_data['Counterpart Area'] == 'Bondholders'
                        ][['Year', 'Value']].copy()
                        bondholders_data['Type'] = 'Bondholders'

                        if selected_creditor == 'Select':
                            combined = pd.concat([country_world_data, bondholders_data], ignore_index=True)
                        else:
                            creditor_line_data = country_creditor_data[
                                country_creditor_data['Counterpart Area'] == selected_creditor
                            ][['Year', 'Value']].copy()
                            creditor_line_data['Type'] = selected_creditor
                            combined = pd.concat([creditor_line_data, country_world_data, bondholders_data], ignore_index=True)

                        color_map_chart = {
                            'World Average': '#ff7f0e',
                            'Bondholders': '#C71585',
                            selected_creditor: '#1f77b4'
                        }

                        if len(combined) > 0:
                            # Separate creditor data from the rest
                            creditor_type_data = combined[combined['Type'] == selected_creditor]
                            other_data = combined[combined['Type'] != selected_creditor]
                            
                            # Plot World Average and Bondholders as lines
                            fig = px.line(
                                other_data, 
                                x='Year', 
                                y='Value', 
                                color='Type',
                                markers=False,
                                color_discrete_map=color_map_chart
                            )
                            
                            # Add selected creditor as points only
                            if len(creditor_type_data) > 0:
                                fig.add_scatter(
                                    x=creditor_type_data['Year'],
                                    y=creditor_type_data['Value'],
                                    mode='markers',
                                    name=selected_creditor,
                                    marker=dict(color=color_map_chart.get(selected_creditor, '#1f77b4'), size=8)
                                )
                            fig = apply_plotly_style(fig, 
                                                     y_title='%', 
                                                     height=350)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info(f'No data available for {selected_creditor} in {country}')
                    else:
                        st.info(f'No creditor data available for {country}')

                    st.markdown('---')
            else:
                st.info('No creditor breakdown data available for selected countries in this time range')
    else:
        st.info('Select countries in the sidebar to view creditor composition and World aggregate trends')


# ---------------credits --------------------------------------------------------------#
# st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div class='sidebar-footer'>"
    "Built by <a href='https://junechoi.com' target='_blank'>June Choi</a> | Data source: <a href='https://www.worldbank.org/en/programs/debt-statistics/ids' target='_blank'>World Bank IDS</a>"
    "</div>",
    unsafe_allow_html=True
)