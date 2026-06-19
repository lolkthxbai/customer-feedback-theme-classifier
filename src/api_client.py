"""Optional CFPB API client for small demo samples."""

from __future__ import annotations

import pandas as pd
import requests

from src.config import BASE_API_URL


def fetch_complaints_from_api(size: int = 100) -> dict:
    """
    Fetch complaint records from the CFPB API.
    Handle request errors and return JSON data.
    """
    params = {
        "size": size,
        "sort": "created_date_desc",
        "field": "all",
    }

    try:
        response = requests.get(BASE_API_URL, params=params, timeout=10)
        response.raise_for_status()
    except requests.exceptions.Timeout as error:
        raise RuntimeError("The CFPB API request timed out.") from error
    except requests.exceptions.RequestException as error:
        raise RuntimeError(f"CFPB API request failed: {error}") from error

    return response.json()


def api_results_to_dataframe(api_response: dict) -> pd.DataFrame:
    """
    Convert CFPB API JSON response into a pandas DataFrame.
    """
    hits = api_response.get("hits", {}).get("hits", [])
    records = [item.get("_source", {}) for item in hits]
    complaints_df = pd.DataFrame(records)

    column_mapping = {
        "complaint_what_happened": "Consumer complaint narrative",
        "product": "Product",
        "issue": "Issue",
        "sub_product": "Sub-product",
        "sub_issue": "Sub-issue",
        "company": "Company",
        "state": "State",
        "submitted_via": "Submitted via",
        "company_response": "Company response to consumer",
        "timely": "Timely response?",
        "date_received": "Date received",
    }
    return complaints_df.rename(columns=column_mapping)
