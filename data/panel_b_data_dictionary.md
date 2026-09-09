# Panel B data dictionary

## Main panel

`panel_b_log_price_returns.csv` contains one ISO date column and 150 ticker
columns. Each value is the one-source-date log price return after the event
rules below. The file has 3,247 rows spanning 2 July 2010 through 30 June 2023.

1. Both the current and immediately previous source-date prices must be finite
   and strictly positive for every ticker. No missing value is imputed and a
   return is never formed across a missing source observation.
2. A raw absolute log change of at least 0.50 is checked against the frozen
   conventional factor grid within 5% relative tolerance.
3. `config/panel_b_event_ledger.csv` takes precedence over the mechanical rule.
   Verified splits/ADS-ratio changes are adjusted; dates containing verified
   distributions whose wealth return cannot be reconstructed are removed in
   full; reviewed economic jumps are retained.
4. All remaining absolute log returns above 0.40 must be ledgered. The current
   reconstruction has zero unresolved moves.

## Sensitivity panel

`panel_b_event_day_exclusion_log_returns.csv` removes every date affected by a
verified or mechanical factor adjustment, in addition to the distribution
dates already absent from the main panel. It contains 3,207 rows and provides a
direct check that results are not driven by repaired event days.

## Audit files

- `panel_b_corporate_action_candidates.csv`: raw ratios, decisions, factors,
  sources, and adjusted returns for all automatic candidates and reviewed
  events.
- `panel_b_unresolved_large_moves.csv`: must be empty before analysis.
- `panel_b_fixed_universe.csv`: ordered ticker universe.
- `panel_b_construction_report.json`: construction counts and claim limits.
- `panel_b_validation_report.json`: dimensions, finite-value checks, maximum
  move, ordering checks, status counts, and SHA-256 hashes.

## Interpretation limit

These are split-adjusted **price returns**, not total returns. Dividends and
delisting returns are not present. Ticker membership is frozen from the
supplied workbook, so the panel is survivorship-conditioned and is not a
point-in-time S&P 500 universe.
