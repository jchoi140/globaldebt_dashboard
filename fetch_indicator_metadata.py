"""
Fetch and cache indicator metadata from the World Bank API.

This module retrieves indicator definitions, units of measure, and source information
from the World Bank International Debt Statistics API. It implements caching to avoid
repeated API calls and includes fallback definitions for offline use.
"""

import json
import requests
from pathlib import Path
from typing import Dict, Optional

# Path to cache file
CACHE_FILE = Path(__file__).parent / 'data' / 'indicator_metadata.json'

# Fallback metadata for all 10 indicators in the dashboard
FALLBACK_METADATA = {
    "DT.TDS.DLXF.CD": {
        "definition": "Debt service on external debt, long-term - total payments of principal and interest on long-term external debt.",
        "unit": "Current US$",
        "source": "World Bank International Debt Statistics"
    },
    "DT.INT.DLXF.CD": {
        "definition": "Interest payments on external debt, long-term - actual amounts of interest paid by the borrower in currency, goods, or services.",
        "unit": "Current US$",
        "source": "World Bank International Debt Statistics"
    },
    "DT.INT.DSTC.CD": {
        "definition": "Interest payments on external debt, short-term - interest payments on debt with maturity of one year or less.",
        "unit": "Current US$",
        "source": "World Bank International Debt Statistics"
    },
    "DT.INR.DPPG": {
        "definition": "Average interest rate on new external debt commitments - weighted average interest rate on new public and publicly guaranteed loans.",
        "unit": "Percent (%)",
        "source": "World Bank International Debt Statistics"
    },
    "DT.INR.OFFT": {
        "definition": "Average interest rate on new external debt commitments from official creditors.",
        "unit": "Percent (%)",
        "source": "World Bank International Debt Statistics"
    },
    "DT.INR.PRVT": {
        "definition": "Average interest rate on new external debt commitments from private creditors.",
        "unit": "Percent (%)",
        "source": "World Bank International Debt Statistics"
    },
    "DT.IXR.OFFT.CD": {
        "definition": "Interest rescheduled from official creditors - interest payments rescheduled under debt relief agreements.",
        "unit": "Current US$",
        "source": "World Bank International Debt Statistics"
    },
    "DT.IXR.PRVT.CD": {
        "definition": "Interest rescheduled from private creditors - interest payments rescheduled under debt relief agreements.",
        "unit": "Current US$",
        "source": "World Bank International Debt Statistics"
    },
    "DT.IXF.DPPG.CD": {
        "definition": "Interest forgiven - interest payments that have been forgiven or cancelled under debt relief initiatives.",
        "unit": "Current US$",
        "source": "World Bank International Debt Statistics"
    },
    "DT.IXR.DPPG.CD": {
        "definition": "Interest rescheduled (capitalized) - rescheduled interest payments that have been added to the principal amount of the debt.",
        "unit": "Current US$",
        "source": "World Bank International Debt Statistics"
    },
}


def load_cache() -> Dict:
    """Load cached metadata from JSON file."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_cache(cache: Dict):
    """Save metadata cache to JSON file."""
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
    except IOError as e:
        print(f"Warning: Could not save cache: {e}")


def fetch_metadata_from_worldbank(indicator_code: str) -> Optional[Dict]:
    """
    Fetch metadata from World Bank API.

    Args:
        indicator_code: The indicator code (e.g., 'DT.INT.DLXF.CD')

    Returns:
        Dictionary with keys: definition, unit, source
        None if fetch fails
    """
    url = f"https://api.worldbank.org/v2/indicator/{indicator_code}?format=json"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()

        # World Bank API returns [metadata_obj, [indicator_data]]
        if len(data) >= 2 and len(data[1]) > 0:
            indicator_info = data[1][0]

            # Extract metadata
            definition = indicator_info.get('sourceNote', '')
            unit = indicator_info.get('unit', '')
            source_org = indicator_info.get('sourceOrganization', '')

            # If unit is empty, try to infer from indicator name or use fallback
            if not unit:
                name = indicator_info.get('name', '')
                if 'current US$' in name:
                    unit = 'Current US$'
                elif '%' in name or 'percent' in name.lower():
                    unit = 'Percent (%)'

            return {
                'definition': definition if definition else 'No definition available.',
                'unit': unit,
                'source': source_org if source_org else 'World Bank'
            }
    except requests.RequestException as e:
        print(f"Warning: Failed to fetch metadata for {indicator_code}: {e}")
        return None
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"Warning: Failed to parse metadata for {indicator_code}: {e}")
        return None


def get_or_fetch_metadata(indicator_code: str) -> Dict:
    """
    Get metadata from cache or fetch from API with fallback.

    Args:
        indicator_code: The indicator code (e.g., 'DT.INT.DLXF.CD')

    Returns:
        Dictionary with keys: definition, unit, source
    """
    if not indicator_code:
        return {"definition": "No indicator code provided.", "unit": "", "source": ""}

    # Load cache
    cache = load_cache()

    # Check cache first
    if indicator_code in cache:
        return cache[indicator_code]

    # Try fetching from API
    metadata = fetch_metadata_from_worldbank(indicator_code)

    # If API fetch succeeded, cache and return
    if metadata:
        cache[indicator_code] = metadata
        save_cache(cache)
        return metadata

    # Fall back to hardcoded metadata
    if indicator_code in FALLBACK_METADATA:
        fallback = FALLBACK_METADATA[indicator_code]
        # Cache the fallback too
        cache[indicator_code] = fallback
        save_cache(cache)
        return fallback

    # Last resort: return minimal metadata
    return {
        "definition": f"World Bank indicator: {indicator_code}",
        "unit": "",
        "source": "World Bank International Debt Statistics"
    }


# For testing purposes
if __name__ == "__main__":
    print("Testing metadata fetcher...\n")

    test_codes = [
        "DT.INT.DLXF.CD",
        "DT.TDS.DLXF.CD",
        "DT.INR.DPPG"
    ]

    for code in test_codes:
        print(f"Fetching metadata for: {code}")
        metadata = get_or_fetch_metadata(code)
        print(f"  Definition: {metadata['definition'][:100]}...")
        print(f"  Unit: {metadata['unit']}")
        print(f"  Source: {metadata['source']}")
        print()

    print(f"Cache saved to: {CACHE_FILE}")
