# NASA POWER seasonal weather summary

Distinct weather series across 31 wells: **2** (coarse grid; see README).

Windows: pre-monsoon = 1 Nov (previous year) – 31 May; post-monsoon = 1 Jun – 31 Oct.

```
season             post    pre
rain_mm    count  589.0  558.0
           mean   449.5  141.6
           std    115.5   67.6
           min    206.3   28.3
           25%    358.8   88.9
           50%    464.2  132.2
           75%    520.7  183.0
           max    659.5  325.5
t2m_mean_c count  589.0  558.0
           mean    32.0   21.1
           std      0.7    1.0
           min     30.7   19.1
           25%     31.4   20.5
           50%     32.1   21.1
           75%     32.5   22.0
           max     33.9   23.0
```

Parameter metadata returned by the API:

```json
{
  "PRECTOTCORR": {
    "units": "mm/day",
    "longname": "Precipitation Corrected"
  },
  "T2M": {
    "units": "C",
    "longname": "Temperature at 2 Meters"
  },
  "T2M_MAX": {
    "units": "C",
    "longname": "Temperature at 2 Meters Maximum"
  }
}
```
