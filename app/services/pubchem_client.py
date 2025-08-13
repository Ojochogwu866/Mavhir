"""
PubChem API client for chemical database lookups.
"""

import logging
import asyncio
import time
from typing import Dict, Optional, Any, Union
import requests
from dataclasses import dataclass

from ..core.config import get_settings
from ..core.exceptions import (
    PubChemError,
    PubChemAPIError,
    PubChemNotFoundError,
    PubChemRateLimitError,
)

logger = logging.getLogger(__name__)


@dataclass
class CompoundInfo:
    """Information retrieved from PubChem."""

    cid: Optional[int]
    name: Optional[str]
    smiles: Optional[str]
    molecular_formula: Optional[str]
    molecular_weight: Optional[float]
    synonyms: Optional[list]
    found: bool


# PUBCHEM API CLIENT
class PubChemClient:
    """
    Simple PubChem REST API client.
    """

    def __init__(self):
        """Initialize PubChem client."""

        self.settings = get_settings()

        self.base_url = self.settings.pubchem_base_url
        self.timeout = self.settings.pubchem_timeout
        self.rate_delay = self.settings.pubchem_rate_limit_delay
        self.max_retries = self.settings.pubchem_max_retries

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "ToxicityPredictor/1.0 (https://github.com/Ojochogwu866/Mavhir)"
            }
        )

        self._last_request_time = 0.0

        logger.info(
            f"PubChemClient initialized (timeout: {self.timeout}s, delay: {self.rate_delay}s)"
        )

    def search_by_name(self, name: str) -> CompoundInfo:
        """
        Search for compound by name.

        EXAMPLES:
        - "ethanol" → Returns ethanol info
        """

        name = name.strip().lower()

        if not name:
            return CompoundInfo(
                cid=None,
                name=None,
                smiles=None,
                molecular_formula=None,
                molecular_weight=None,
                synonyms=None,
                found=False,
            )

        try:
            cid = self._get_cid_by_name(name)
            if cid is None:
                return CompoundInfo(
                    cid=None,
                    name=name,
                    smiles=None,
                    molecular_formula=None,
                    molecular_weight=None,
                    synonyms=None,
                    found=False,
                )

            return self._get_compound_properties(cid, query_name=name)

        except PubChemError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error searching for '{name}': {e}")
            raise PubChemAPIError(name, details=str(e))

    def get_by_cid(self, cid: int) -> CompoundInfo:
        """Get compound information by PubChem CID."""

        try:
            return self._get_compound_properties(cid)
        except Exception as e:
            raise PubChemAPIError(str(cid), details=str(e))

    def _get_cid_by_name(self, name: str) -> Optional[int]:
        """Get PubChem CID from compound name."""

        url = f"{self.base_url}/compound/name/{name}/cids/JSON"

        try:
            response = self._make_request(url)
            data = response.json()

            if "IdentifierList" in data and "CID" in data["IdentifierList"]:
                cids = data["IdentifierList"]["CID"]
                if cids:
                    return cids[0]

            return None

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None  # Compound not found
            else:
                raise PubChemAPIError(name, status_code=e.response.status_code)

    def _get_compound_properties(
        self, cid: int, query_name: Optional[str] = None
    ) -> CompoundInfo:
        """Get compound properties from PubChem CID."""

        # Properties we want to retrieve
        properties = [
            "MolecularFormula",
            "MolecularWeight",
            "CanonicalSMILES",
            "Title",  # Primary name
        ]

        property_string = ",".join(properties)
        url = f"{self.base_url}/compound/cid/{cid}/property/{property_string}/JSON"

        try:
            response = self._make_request(url)
            data = response.json()

            if "PropertyTable" in data and "Properties" in data["PropertyTable"]:
                props = data["PropertyTable"]["Properties"][0]

                return CompoundInfo(
                    cid=cid,
                    name=props.get("Title", query_name),
                    smiles=props.get("CanonicalSMILES"),
                    molecular_formula=props.get("MolecularFormula"),
                    molecular_weight=props.get("MolecularWeight"),
                    synonyms=None,
                    found=True,
                )

            return CompoundInfo(
                cid=cid,
                name=query_name,
                smiles=None,
                molecular_formula=None,
                molecular_weight=None,
                synonyms=None,
                found=False,
            )

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise PubChemNotFoundError(str(cid))
            else:
                raise PubChemAPIError(str(cid), status_code=e.response.status_code)

    def _make_request(self, url: str) -> requests.Response:
        """
        Make HTTP request with rate limiting and retry logic.
        """

        current_time = time.time()
        time_since_last = current_time - self._last_request_time

        if time_since_last < self.rate_delay:
            sleep_time = self.rate_delay - time_since_last
            time.sleep(sleep_time)

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Making PubChem request: {url}")

                response = self.session.get(url, timeout=self.timeout)
                self._last_request_time = time.time()

                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    raise PubChemRateLimitError(retry_after)

                response.raise_for_status()

                return response

            except requests.exceptions.Timeout:
                logger.warning(f"PubChem request timeout (attempt {attempt + 1})")
                if attempt == self.max_retries - 1:
                    raise PubChemAPIError(url, details="Request timeout")

            except requests.exceptions.ConnectionError:
                logger.warning(f"PubChem connection error (attempt {attempt + 1})")
                if attempt == self.max_retries - 1:
                    raise PubChemAPIError(url, details="Connection error")

            except PubChemRateLimitError:
                # Don't retry rate limit errors
                raise

            except requests.exceptions.RequestException as e:
                logger.error(f"PubChem request failed: {e}")
                raise PubChemAPIError(url, details=str(e))

        # Should never reach here
        raise PubChemAPIError(url, details="Max retries exceeded")


# cf
def create_pubchem_client() -> PubChemClient:
    """Factory function to create PubChemClient."""
    return PubChemClient()


def quick_lookup(name: str) -> CompoundInfo:
    """
    Quick compound lookup function.

    USAGE:
        info = quick_lookup("caffeine")
        if info.found:
            print(f"Caffeine SMILES: {info.smiles}")
    """

    client = create_pubchem_client()
    return client.search_by_name(name)


# tf
def _test_pubchem_client():
    """Test PubChem client with common compounds."""

    client = create_pubchem_client()

    test_compounds = [
        "ethanol",
        "caffeine",
        "aspirin",
        "benzene",
        "invalid_compound_name_12345",
    ]

    print("Testing PubChem Client:")
    print("=" * 40)

    for compound_name in test_compounds:
        try:
            print(f"\nSearching: {compound_name}")

            info = client.search_by_name(compound_name)

            if info.found:
                print(f"   Found: {info.name}")
                print(f"   CID: {info.cid}")
                print(f"   SMILES: {info.smiles}")
                print(f"   Formula: {info.molecular_formula}")
                print(f"   MW: {info.molecular_weight}")
            else:
                print(f"Not found")

        except PubChemError as e:
            print(f"  PubChem error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    print(f"\n{'='*40}")
    print("PubChem client test complete")


if __name__ == "__main__":
    _test_pubchem_client()
