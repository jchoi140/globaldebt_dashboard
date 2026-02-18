# Global Sovereign Debt Dashboard

An interactive Streamlit dashboard for exploring sovereign debt indicators across countries, years, and creditor counterparties. Built with data from the World Bank International Debt Statistics (IDS).

## Features

The dashboard has three main views, accessible via the sidebar:

- **External Debt** -- Debt Outstanding and Disbursed (DOD) with creditor breakdowns, external debt stock ratios (% of GNI, % of exports), and interest payment overlays
- **Debt Service** -- Long-term debt service payments with stacked creditor composition charts, percentage share views, and top creditor rankings
- **Interest Payments** -- Interest payment indicators across 9 categories including average interest rates by creditor type, interest rescheduled/forgiven, and creditor-level rate comparisons

## Data

All data is sourced from the [World Bank International Debt Statistics](https://www.worldbank.org/en/programs/debt-statistics/ids) database. Indicator metadata (definitions, units) is fetched from the World Bank API and cached locally.

## Run locally

1. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Launch the app:

   ```
   streamlit run app_v4.py
   ```

## License

Apache 2.0 -- see [LICENSE](LICENSE) for details.
