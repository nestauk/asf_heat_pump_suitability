# asf_heat_pump_suitability/config

## Data sources

### Features

| Config key                               | S3 file                         | Source                                                                                     | Date        | Description                                                                                                           |
| :--------------------------------------- | :------------------------------ | :----------------------------------------------------------------------------------------- | :---------- | :-------------------------------------------------------------------------------------------------------------------- |
| `["data_source"]["UK_ons_postcode_dir"]` | `source_data/ONSPD_AUG_2023_UK` | [ONS](https://geoportal.statistics.gov.uk/datasets/487a5ba62c8b4da08f01eb3c08e304f6/about) | August 2023 | ONS postcode directory for UK as at August 2023. Source of output area information and rural-urban classification.^1^ |

_Footnotes:_

1. Rural-urban classification codes are mapped to text descriptions according to the sources listed below:
   - Scotland 8-fold code to text classification mapping: [Table 2.2, Scottish Government Urban Rural Classification](https://www.gov.scot/publications/scottish-government-urban-rural-classification-2020/pages/2/)
   - Scotland 8-fold to 2-fold classification mapping: [Table 2.3, Scottish Government Urban Rural Classification](https://www.gov.scot/publications/scottish-government-urban-rural-classification-2020/pages/2/)
   - England & Wales 10- and 2-fold code to text classification mapping: [page 19, ONS Postcode Directory (August 2023) User Guide](https://geoportal.statistics.gov.uk/datasets/a8db59f77e7542d092458426dbacfb90/about)

### Validation datasets

|                       Config key                       | S3 file                                                                 | Source                                                                                                                                 | Date                          | Description                                                                                                                            |
| :----------------------------------------------------: | :---------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------- | :------------------------------------------------------------------------------------------------------------------------------------- |
| `["data_source"]["EW_census_housing_characteristics"]` | `source_data/2021census_Mar2023update_housing_characteristics_E_W.xlsx` | [ONS](https://www.ons.gov.uk/peoplepopulationandcommunity/housing/datasets/numberofdwellingsbyhousingcharacteristicsinenglandandwales) | Census 2021, updated Mar 2023 | Number of dwellings by different property features for LSOA level <br/>and above for England and Wales. Note, contains censored data.  |
|         `["data_source"]["EW_census_tenure"]`          | `source_data/2021census_Mar2023update_tenure_E_W.csv`                   | [ONS](https://www.ons.gov.uk/datasets/TS054/editions/2021/versions/4)                                                                  | Census 2021, updated Mar 2023 | Census 2021 estimates that classify all households in England and Wales by tenure, per LSOA.                                           |
|     `["data_source"]["EW_census_number_of_rooms"]`     | `source_data/2021census_Mar2023update_number_of_rooms_E_W.csv`          | [ONS](https://www.ons.gov.uk/datasets/TS051/editions/2021/versions/4)                                                                  | Census 2021, updated Mar 2023 | Census 2021 estimates that classify all households in England and Wales by number of rooms, per LSOA.                                  |
|       `["data_source"]["EW_cdrc_dwelling_age"]`        | `source_data/2015cdrc_dwelling_ages_E_W.csv`                            | [CDRC](https://data.cdrc.ac.uk/dataset/dwelling-ages-and-prices/resource/dwelling-age-band-counts-lsoa-2015)                           | 2015                          | Residential dwelling ages, grouped into approximately 10-year age bands from pre-1900 to 2015, with counts of each age group per LSOA. |
