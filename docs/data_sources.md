# Tennis match data sources

## Primary source

The project uses yearly Excel files from Tennis-Data.

Base Source pattern: [here](http://tennis-data.co.uk/2026/2026.xlsx)

## ATP enrichment source

Player biographies and historical match statistics originate from Jeff
Sackmann's ATP dataset. Because the original upstream repository is currently
unavailable, the reproducible pipeline uses the [June 2026 archival
mirror](https://github.com/Aneeshers/tennis-sackmann-archive), specifically its
`atp/` directory.

The Sackmann data is licensed CC BY-NC-SA 4.0. Attribution is required and
commercial use is not permitted by that license.
