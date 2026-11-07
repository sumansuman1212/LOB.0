# Market Data Directory (`data/`)

This directory is a designated storage location for offline market-data archives and historical order-book tick datasets used by the C++ tick replayer and Python simulation environments.

### Supported File Formats

- **Parquet (`*.parquet`):** Preferred columnar format for high-speed replay. Expected schema:
  - `timestamp_ns` (int64): Event timestamp in nanoseconds since UNIX epoch.
  - `order_id` (uint64): Exchange-assigned order identifier.
  - `price` (int64): Integer tick price ($P_{\text{ticks}} = \text{Price} / \text{tick\_size}$).
  - `qty` (uint32): Quantity in lots.
  - `type` (string / int): `Add`, `Cancel`, `Modify`, `Fill`, `PartialFill`.
  - `side` (string / int): `Bid` or `Ask`.
- **CSV (`*.csv`):** Standard delimited tick stream data.

### Important Notes

- Raw tick data files (`*.parquet`, `*.csv`) are ignored by `.gitignore` to prevent committing multi-gigabyte binary archives to Git.
- In the absence of external market data files, `LobEnv` and `mm_backtest` automatically utilize deterministic synthetic tick generators.

