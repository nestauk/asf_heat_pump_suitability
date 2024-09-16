# asf_heat_pump_suitability/config

## Data sources

### Features

See additional attribution statements listed below.
| Config key | S3 file | Source | Date | Description |
| :------------------------------------------------------------- | :--------------------------------------------------------------------------------------------- |:----------------------------------------------------------------------------------------------------------------------------------------------------------------| :---------- |:-------------------------------------------------------------------------------------------------------------------------------|
| `["data_source"]["UK_ons_postcode_dir"]` | `source_data/ONSPD_AUG_2023_UK.csv` | [ONS](https://geoportal.statistics.gov.uk/datasets/487a5ba62c8b4da08f01eb3c08e304f6/about)<sup>1,2,3</sup> | August 2023 | ONS postcode directory for UK as at August 2023. Source of output area information and rural-urban classification<sup>a</sup>. |
| `["data_source"]["GB_ons_garden_space_access"]` | `source_data/ONS_Apr2020_access_to_garden_space.xlsx` | [ONS](https://www.ons.gov.uk/economy/environmentalaccounts/datasets/accesstogardensandpublicgreenspaceingreatbritain)<sup>4</sup> | April 2020 | Access to garden space, Great Britain. Contains avg garden size per MSOA. |
| `["data_source"]["GB_osopen_uprn_latlon"]` | `source_data/osopenuprn_202405_csv.zip` | [OS Open](https://osdatahub.os.uk/downloads/open/OpenUPRN)<sup>4</sup> | May 2024 | UPRNs of GB with their latitude and longitude. |
| `["data_source"]["UK_ons_lad_bounds"]` | `source_data/Local_Authority_Districts_December_2023_Boundaries_UK_BFE_-2600600853110041429/*` | [ONS](https://www.data.gov.uk/dataset/288458f7-7789-47d0-80d4-ffdf746c6b75/local-authority-districts-december-2023-boundaries-uk-bfe)<sup>4</sup> | Dec 2023 | Local Authority District boundaries UK BFE. |
| `["data_source"]["EW_inspire_land_extent"]` | `source_data/inspire_gml/*` | [INSPIRE Land Registry](https://use-land-property-data.service.gov.uk/datasets/inspire)<sup>5,6</sup> | Nov 2023 | Registered land extent polygons for England and Wales by council (LAD). |
| `["data_source"]["global_microsoft_building_footprint_links"]` | `source_data/June2024_microsoft_building_footprint_dataset_links.csv` | [Microsoft Global ML Building Footprints ](https://github.com/microsoft/GlobalMLBuildingFootprints)<sup>7</sup> | June 2024 | Global building footprints and building height data. |
| `["data_source"]["EW_census_number_of_households"]` | `source_data/2021_vMar2023_census_numberofhouseholds_EW.csv` | [ONS](https://www.nomisweb.co.uk/datasets/c2021ts041)<sup>4</sup> | March 2023 | Census estimates on the number of households in England and Wales. |
| `["data_source"]["EW_census_land_area"]` | `source_data/2021_vMar2021_census_landareaKM_EW.csv` | [ONS](https://geoportal.statistics.gov.uk/datasets/a488cb8fc9a74accb63cb52961e456ef/about)<sup>3</sup> | March 2021 | Standard Area Measurements (SAM) for the 2021 Statistical Areas in England and Wales. |
| `["data_source"]["UK_spa_offgasgrid"]` | `source_data/2024_vMar2024_SPA_offgaspostcode_UK.xlsx` | [Xoserve](https://www.xoserve.com/help-centre/supply-points-metering/supply-point-administration-spa/) | March 2024 | Register of postcodes with no record of an active gas connection. |
| `["data_source"]["E_historicengland_listed_buildings"]` | `source_data/Jun2024_vJul2024_HistoricEngland_listedbuilding_E.gpkg` | [Historic England](https://opendata-historicengland.hub.arcgis.com/datasets/historicengland::national-heritage-list-for-england-nhle/about?layer=3)<sup>8</sup> | June 2024 | Geographic locations and boundaries of listed buildings in England. |
| `["data_source"]["W_cadw_listed_buildings"]` | `source_data/May2024_vMay2024_Cadw_listedbuilding_W.gpkg` | [Cadw](https://datamap.gov.wales/layers/inspire-wg:Cadw_ListedBuildings)<sup>9</sup> | May 2024 | Geographic locations and boundaries of listed buildings in Wales. |
| `["data_source"]["E_historic_england_conservation_areas"]` | `source_data/Aug2024_historic_england_conservation_areas_E.geojson` | [Historic England](https://www.planning.data.gov.uk/dataset/conservation-area)<sup>10</sup> | June 2024 | Building conservation area boundaries in England (places of special architectural and historic interest). |
| `["data_source"]["W_welsh_gov_conservation_areas"]` | `source_data/2022_welsh_gov_building_conservation_areas_W.gpkg` | [Welsh Government](https://datamap.gov.wales/layers/inspire-wg:conservation_areas)<sup>4</sup> | Sept 2022 | Building conservation area boundaries in Wales (places of special architectural and historic interest). |
| `["data_source"]["EW_ons_lsoa_lad_lookup"]` | `source_data/2021_vApr2023_ons_lsoa_to_lad_lookup_EW.csv` | [ONS](https://geoportal.statistics.gov.uk/datasets/ons::lsoa-2021-to-local-authority-districts-april-2023-best-fit-lookup-in-ew/explore)<sup>3</sup> | April 2023 | LSOA to LAD lookup. |

_Footnotes:_

a. Rural-urban classification codes are mapped to text descriptions according to the sources listed below:

- Scotland 8-fold code to text classification mapping: [Table 2.2, Scottish Government Urban Rural Classification](https://www.gov.scot/publications/scottish-government-urban-rural-classification-2020/pages/2/)
- Scotland 8-fold to 2-fold classification mapping: [Table 2.3, Scottish Government Urban Rural Classification](https://www.gov.scot/publications/scottish-government-urban-rural-classification-2020/pages/2/)
- England & Wales 10- and 2-fold code to text classification mapping: [page 19, ONS Postcode Directory (August 2023) User Guide](https://geoportal.statistics.gov.uk/datasets/a8db59f77e7542d092458426dbacfb90/about)

_Attributions:_

1. Contains OS data © Crown copyright and database right 2024.
2. Contains Royal Mail data © Royal Mail copyright and database right 2024.
3. Contains Office for National Statistics information licensed under the Open Government Licence v.3.0.
4. Contains public sector information licensed under the Open Government Licence v3.0.
5. This information is subject to Crown copyright and database rights 2024 and is reproduced with the permission of HM Land Registry. [See INSPIRE index polygons conditions of use](https://use-land-property-data.service.gov.uk/datasets/inspire#conditions).
6. The polygons (including the associated geometry, namely x, y co-ordinates) are subject to Crown copyright and database rights 2024 Ordnance Survey 100026316.
7. This data is licensed by Microsoft under the [Open Data Commons Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/).
8. © Historic England 2024. Contains Ordnance Survey data © Crown copyright and database right 2024.
9. Designated Historic Asset GIS Data, The Welsh Historic Environment Service (Cadw), DATE 2024, licensed under the [Open Government Licence](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
10. © Historic England 2024. Contains Ordnance Survey data © Crown copyright and database right 2024. The Historic England GIS Data contained in this material was obtained on August 2024. The most publicly available up to date Historic England GIS Data can be obtained from HistoricEngland.org.uk.
11. The data for this research have been provided by the Consumer Data Research Centre, an ESRC Data Investment.

### Validation datasets

|                       Config key                       | S3 file                                                                 | Source                                                                                                                                 | Date                          | Description                                                                                                                            |
| :----------------------------------------------------: | :---------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------- | :------------------------------------------------------------------------------------------------------------------------------------- |
| `["data_source"]["EW_census_housing_characteristics"]` | `source_data/2021census_Mar2023update_housing_characteristics_E_W.xlsx` | [ONS](https://www.ons.gov.uk/peoplepopulationandcommunity/housing/datasets/numberofdwellingsbyhousingcharacteristicsinenglandandwales) | Census 2021, updated Mar 2023 | Number of dwellings by different property features for LSOA level <br/>and above for England and Wales. Note, contains censored data.  |
|         `["data_source"]["EW_census_tenure"]`          | `source_data/2021census_Mar2023update_tenure_E_W.csv`                   | [ONS](https://www.ons.gov.uk/datasets/TS054/editions/2021/versions/4)                                                                  | Census 2021, updated Mar 2023 | Census 2021 estimates that classify all households in England and Wales by tenure, per LSOA.                                           |
|     `["data_source"]["EW_census_number_of_rooms"]`     | `source_data/2021census_Mar2023update_number_of_rooms_E_W.csv`          | [ONS](https://www.ons.gov.uk/datasets/TS051/editions/2021/versions/4)                                                                  | Census 2021, updated Mar 2023 | Census 2021 estimates that classify all households in England and Wales by number of rooms, per LSOA.                                  |
|       `["data_source"]["EW_cdrc_dwelling_age"]`        | `source_data/2015cdrc_dwelling_ages_E_W.csv`                            | [CDRC](https://data.cdrc.ac.uk/dataset/dwelling-ages-and-prices/resource/dwelling-age-band-counts-lsoa-2015)<sup>11</sup>              | 2015                          | Residential dwelling ages, grouped into approximately 10-year age bands from pre-1900 to 2015, with counts of each age group per LSOA. |
