"""Download and cache academic factor return data for strategy distillation.

Supports Fama-French 5-factor (daily) and Fung-Hsieh 7-factor (monthly)
datasets.  Follows the same memory-dict + disk-file caching pattern used
by :class:`hydra.data.feature_store.FeatureStore`.

Backtesting and training only.
"""

from __future__ import annotations

import io
import logging
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("hydra.distillation.factor_data")

_FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
_FH7_URL = "https://people.duke.edu/~dah7/DataLibrary/TF-Fac.xls"

_USER_AGENT = "HydraCorp/1.0"
_CACHE_MAX_AGE_SECONDS = 30 * 24 * 60 * 60  # 30 days


def _urlopen(url: str, verify_ssl: bool = True) -> bytes:
    """Fetch *url* with a custom User-Agent header.  Returns raw bytes.

    Parameters
    ----------
    verify_ssl:
        When ``False``, skip SSL certificate verification.  Used as a
        fallback for servers whose certificate chain is not trusted by
        the local Python/Anaconda installation (common on Windows).
    """
    import ssl as _ssl

    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    ctx = None
    if not verify_ssl:
        ctx = _ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = _ssl.CERT_NONE
    with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
        return resp.read()


# ---------------------------------------------------------------------------
# Kenneth French CSV parser
# ---------------------------------------------------------------------------

def _parse_ff5_csv(raw_text: str) -> pd.DataFrame:
    """Parse the quirky Kenneth French CSV format.

    The file has:
      * Several descriptive text lines at the top
      * A blank line
      * A header row (,Mkt-RF,SMB,HML,RMW,CMA,RF)
      * Numeric data rows  (YYYYMMDD, float, float, ...)
      * Eventually another text section (annual factors) that must be ignored

    Strategy: scan for the first line whose first non-whitespace token looks
    like an 8-digit date, collect rows until the first non-numeric row, then
    build a DataFrame.
    """
    lines = raw_text.splitlines()

    # -- locate header row and data start ----------------------------------
    header_idx: Optional[int] = None
    data_start: Optional[int] = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        first_token = stripped.split(",")[0].strip()
        # Data rows start with an 8-digit numeric date
        if first_token.isdigit() and len(first_token) == 8:
            if data_start is None:
                data_start = i
                # The header is the closest preceding non-blank line
                for j in range(i - 1, -1, -1):
                    if lines[j].strip():
                        header_idx = j
                        break
                break

    if data_start is None or header_idx is None:
        raise ValueError("Could not locate data section in FF5 CSV")

    # -- parse header ------------------------------------------------------
    header_parts = [c.strip() for c in lines[header_idx].split(",")]
    # First column is the date (may be empty label); ensure a name
    if not header_parts[0]:
        header_parts[0] = "date"
    else:
        header_parts[0] = "date"
    n_cols = len(header_parts)

    # -- collect data rows -------------------------------------------------
    rows: list[list[str]] = []
    for i in range(data_start, len(lines)):
        stripped = lines[i].strip()
        if not stripped:
            # Blank line between daily and annual sections -> stop
            break
        first_token = stripped.split(",")[0].strip()
        if not first_token.isdigit():
            break
        parts = [p.strip() for p in stripped.split(",")]
        if len(parts) == n_cols:
            rows.append(parts)

    if not rows:
        raise ValueError("No data rows found in FF5 CSV")

    df = pd.DataFrame(rows, columns=header_parts)

    # Convert date column to DatetimeIndex
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date")

    # Convert factor columns to float and scale from percent to decimal
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0

    df = df.dropna(how="all")
    df = df.sort_index()
    return df


class FactorDataStore:
    """Download, cache, and serve academic factor return datasets.

    Mirrors the caching pattern of :class:`hydra.data.feature_store.FeatureStore`:
    an in-memory dict checked first, then a disk file, then a remote download.
    """

    def __init__(self, cache_dir: str = "data/factor_cache"):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _disk_path(self, name: str) -> Path:
        return self._cache_dir / f"{name}.csv"

    def _is_cache_fresh(self, path: Path) -> bool:
        """Return True if *path* exists and is less than 30 days old."""
        if not path.exists():
            return False
        age = time.time() - path.stat().st_mtime
        return age < _CACHE_MAX_AGE_SECONDS

    def _read_disk_cache(self, name: str) -> Optional[pd.DataFrame]:
        path = self._disk_path(name)
        if not self._is_cache_fresh(path):
            return None
        try:
            df = pd.read_csv(str(path), index_col=0, parse_dates=True)
            logger.debug("Loaded %s from disk cache (%s)", name, path)
            return df
        except Exception as exc:
            logger.warning("Failed to read disk cache %s: %s", path, exc)
            return None

    def _write_disk_cache(self, name: str, df: pd.DataFrame) -> None:
        path = self._disk_path(name)
        try:
            df.to_csv(str(path))
            logger.debug("Saved %s to disk cache (%s)", name, path)
        except Exception as exc:
            logger.warning("Failed to write disk cache %s: %s", path, exc)

    def _get_cached(self, name: str) -> Optional[pd.DataFrame]:
        """Check memory then disk cache.  Returns None on miss."""
        if name in self._memory_cache:
            return self._memory_cache[name]
        df = self._read_disk_cache(name)
        if df is not None:
            self._memory_cache[name] = df
        return df

    def _put_cache(self, name: str, df: pd.DataFrame) -> None:
        self._memory_cache[name] = df
        self._write_disk_cache(name, df)

    @staticmethod
    def _filter_as_of(
        df: pd.DataFrame, as_of_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Return rows where index <= *as_of_date* for point-in-time correctness."""
        if as_of_date is None:
            return df
        cutoff = pd.Timestamp(as_of_date)
        return df.loc[df.index <= cutoff]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_fama_french_5(
        self, as_of_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Fama-French 5-factor daily returns.

        Columns: Mkt-RF, SMB, HML, RMW, CMA, RF  (decimal, not percent).

        Parameters
        ----------
        as_of_date : str or None
            If given, only rows up to and including this date are returned
            (``YYYY-MM-DD`` format).

        Returns
        -------
        pd.DataFrame or None
            DataFrame with a DatetimeIndex and float64 columns, or None if
            the download failed.
        """
        cache_name = "ff5_daily"
        df = self._get_cached(cache_name)

        if df is None:
            logger.info("Downloading Fama-French 5-factor data ...")
            try:
                raw = _urlopen(_FF5_URL)
                with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                    # There is typically one CSV inside the zip
                    csv_names = [
                        n for n in zf.namelist() if n.lower().endswith(".csv")
                    ]
                    if not csv_names:
                        raise ValueError("No CSV found inside FF5 zip archive")
                    csv_bytes = zf.read(csv_names[0])
                    csv_text = csv_bytes.decode("utf-8", errors="replace")

                df = _parse_ff5_csv(csv_text)
                self._put_cache(cache_name, df)
                logger.info(
                    "Fama-French 5-factor data cached (%d rows, %s to %s)",
                    len(df),
                    df.index.min().date(),
                    df.index.max().date(),
                )
            except Exception as exc:
                logger.error("Failed to download FF5 data: %s", exc)
                return None

        return self._filter_as_of(df, as_of_date)

    def get_fung_hsieh_7(
        self, as_of_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Fung-Hsieh 7-factor monthly trend-following returns.

        Columns: PTFSBD, PTFSFX, PTFSCOM, PTFSIR, PTFSSTK,
                 bond_factor, stock_factor.

        Parameters
        ----------
        as_of_date : str or None
            If given, only rows up to and including this date are returned
            (``YYYY-MM-DD`` format).

        Returns
        -------
        pd.DataFrame or None
            DataFrame with a DatetimeIndex, or None on failure.
        """
        cache_name = "fh7_monthly"
        df = self._get_cached(cache_name)

        if df is None:
            logger.info("Downloading Fung-Hsieh trend-following factor data ...")
            try:
                try:
                    raw = _urlopen(_FH7_URL)
                except urllib.error.URLError:
                    # Duke's SSL cert often fails on Windows/Anaconda — retry
                    # without verification (data is public academic research).
                    logger.info("FH7 SSL verification failed, retrying without verification")
                    raw = _urlopen(_FH7_URL, verify_ssl=False)

                # Data is distributed as an XLS file (not CSV).
                df = pd.read_excel(io.BytesIO(raw))

                # Normalise: first column is the date index.
                # Dates are typically YYYYMM integers (e.g. 199401 = Jan 1994).
                date_col = df.columns[0]
                df[date_col] = pd.to_datetime(
                    df[date_col].astype(str).str.strip(),
                    format="%Y%m",
                    errors="coerce",
                )
                df = df.dropna(subset=[date_col])
                df = df.set_index(date_col)
                df.index.name = "date"

                # Rename columns to canonical names.  The XLS typically has
                # columns like PTFSBD, PTFSFX, PTFSCOM, etc. but labels vary.
                canonical = [
                    "PTFSBD",
                    "PTFSFX",
                    "PTFSCOM",
                    "PTFSIR",
                    "PTFSSTK",
                    "bond_factor",
                    "stock_factor",
                ]
                if len(df.columns) >= 7:
                    rename_map = {
                        old: new
                        for old, new in zip(df.columns[:7], canonical)
                    }
                    df = df.rename(columns=rename_map)
                    df = df[canonical]

                # Ensure float dtype
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                df = df.dropna(how="all")
                df = df.sort_index()
                self._put_cache(cache_name, df)
                logger.info(
                    "Fung-Hsieh trend-following factor data cached (%d rows, %s to %s)",
                    len(df),
                    df.index.min().date(),
                    df.index.max().date(),
                )
            except Exception as exc:
                logger.error("Failed to download FH7 data: %s", exc)
                return None

        return self._filter_as_of(df, as_of_date)

    def get_hfri_composite(
        self, as_of_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """HFRI Fund Weighted Composite Index (best-effort).

        This dataset requires a paid subscription to Hedge Fund Research, Inc.
        The method is provided as a stub so that downstream code can call it
        uniformly; it always returns ``None``.

        Returns
        -------
        None
        """
        logger.info(
            "HFRI composite data is not available (requires subscription). "
            "Returning None."
        )
        return None

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_memory_cache(self) -> None:
        """Clear in-memory cache (disk cache persists)."""
        self._memory_cache.clear()

    def clear_all(self) -> None:
        """Clear both memory and disk cache."""
        self._memory_cache.clear()
        for f in self._cache_dir.glob("*.csv"):
            f.unlink()
