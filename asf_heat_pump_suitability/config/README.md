# asf_heat_pump_suitability/config

## Data sources

The sources of data used in this project are listed in the tables below. Data is downloaded from these sources and saved
to S3 in Nesta's `asf-heat-pump-suitability` bucket (private) in the `source_data` directory. The files in S3 must be
manually updated when new versions of the source data are available. The appropriate value in
`asf_heat_pump_suitability/config/base.yaml` must then be updated with the new S3 URI.

In the tables below, the 'Config key' column indicates the
key used in `asf_heat_pump_suitability/config/base.yaml` under the primary `["data_source"]` key.

### EPC data

We have used domestic Energy Performance Certificate (EPC) records from the [Energy Performance of Buildings Register](https://epc.opendatacommunities.org/) for
England and Wales published by the Department for Levelling Up, Housing & Communities ([licensing information](https://epc.opendatacommunities.org/docs/copyright)),
and from the [Scottish Energy Performance Certificate Register](https://www.scottishepcregister.org.uk/) published by the Scottish Government
for Scotland ([licensing information](https://statistics.gov.scot/data/domestic-energy-performance-certificates)). The data has been combined
and preprocessed and deduplicated with our `asf-daps` [pipeline](https://github.com/nestauk/asf_daps).

### Features

This table shows the different datasets used to calculate each feature and their sources.

See additional data attribution statements listed below.
| Feature | Config key | Source | Date | Description |
| :---------------------------- | :--------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------- | :----------------------------------------------------------------------------------------------------------------------------- |
| LSOA, MSOA, LAD, RUC | `["UK_ons_postcode_dir"]` | [ONS](https://geoportal.statistics.gov.uk/datasets/487a5ba62c8b4da08f01eb3c08e304f6/about)<sup>1,2,3</sup> | August 2023 | ONS postcode directory for UK as at August 2023. Source of output area information and rural-urban classification<sup>a</sup>. |
| Average garden size | `["GB_ons_garden_space_access"]` | [ONS](https://www.ons.gov.uk/economy/environmentalaccounts/datasets/accesstogardensandpublicgreenspaceingreatbritain)<sup>4</sup> | April 2020 | Access to garden space, Great Britain. Contains avg garden size per MSOA. |
| Lat, lon and BNG coordinates | `["GB_osopen_uprn_latlon"]` | [OS Open](https://osdatahub.os.uk/downloads/open/OpenUPRN)<sup>4</sup> | May 2024 | UPRNs of GB with their latitude and longitude. |
| Garden size (& in building conservation area) | `["UK_ons_lad_bounds"]` | [ONS](https://www.data.gov.uk/dataset/288458f7-7789-47d0-80d4-ffdf746c6b75/local-authority-districts-december-2023-boundaries-uk-bfe)<sup>4</sup> | Dec 2023 | Local Authority District boundaries UK BFE. |
| Garden size | `["EW_inspire_land_extent_dir"]` | [INSPIRE Land Registry](https://use-land-property-data.service.gov.uk/datasets/inspire)<sup>5,6</sup> | Nov 2023 | Registered land extent polygons for England and Wales by council (LAD). |
| Garden size | `["S_inspire_land_extent_dir"]` | [Registers of Scotland](https://ros-inspire.themapcloud.com/)<sup>1,4,13</sup> | May 2024 | Registered land extent polygons for Scotland by Registration County. |
| Garden size | `["global_microsoft_building_footprint_links"]` | [Microsoft Global ML Building Footprints ](https://github.com/microsoft/GlobalMLBuildingFootprints)<sup>7</sup> | Regular updates (approx. monthly) | Global building footprints and building height data. |
| Households per km2 | `["EW_census_number_of_households"]` | [ONS](https://www.nomisweb.co.uk/datasets/c2021ts041)<sup>4</sup> | March 2023 | Census estimates on the number of households in England and Wales. |
| Households per km2 | `["EW_census_land_area"]` | [ONS](https://geoportal.statistics.gov.uk/datasets/a488cb8fc9a74accb63cb52961e456ef/about)<sup>3</sup> | March 2021 | Standard Area Measurements (SAM) for the 2021 Statistical Areas in England and Wales. |
| Households per km2 | `["S_NRScotland_households"]` | [National Records Scotland](https://www.nrscotland.gov.uk/publications/small-area-statistics-on-households-and-dwellings/)<sup>1,4,13</sup> | 2023, June 2024 update | Estimates of the numbers of households per Data Zone in Scotland. |
| Grid capacity | `["S_scottish_gov_DZ2011_boundaries"]` | [Scottish Government](https://www.data.gov.uk/dataset/ab9f1f20-3b7f-4efa-9bd2-239acf63b540/data-zone-boundaries-2011)<sup>4</sup> | June 2024 | 2011 Scottish Data Zone geospatial boundaries. |
| Off-gas status | `["UK_spa_offgasgrid"]` | [Xoserve](https://www.xoserve.com/help-centre/supply-points-metering/supply-point-administration-spa/) | March 2024 | Register of postcodes with no record of an active gas connection. |
| Listed building status | `["E_historicengland_listed_buildings"]` | [Historic England](https://opendata-historicengland.hub.arcgis.com/datasets/historicengland::national-heritage-list-for-england-nhle/about?layer=3)<sup>8</sup> | June 2024 | Geospatial polygons of listed buildings in England. |
| Listed building status | `["W_cadw_listed_buildings"]` | [Cadw](https://datamap.gov.wales/layers/inspire-wg:Cadw_ListedBuildings)<sup>9</sup> | May 2024 | Point geometries of listed buildings in Wales. |
| Listed building status | `["S_scottish_gov_listed_buildings"]` | [Scottish Government](https://www.data.gov.uk/dataset/722b93f3-75fd-47ce-9f06-0efcfa010ecf/listed-buildings)<sup>4</sup> | June 2024 | Point geometries of listed buildings in Scotland. |
| In building conservation area | `["E_historic_england_conservation_areas"]` | [Historic England](https://www.planning.data.gov.uk/dataset/conservation-area)<sup>10</sup> | Feb 2025 | Building conservation area boundaries in England (places of special architectural and historic interest). |
| In building conservation area | `["W_welsh_gov_conservation_areas"]` | [Welsh Government](https://datamap.gov.wales/layers/inspire-wg:conservation_areas)<sup>4</sup> | Sept 2022 | Building conservation area boundaries in Wales (places of special architectural and historic interest). |
| In World Heritage Site | `["S_historic_environment_scotland_world_heritage_sites"]` | [Scottish Government](https://www.data.gov.uk/dataset/eab6ee72-23e8-46df-b74b-c2a9cb3ee6e0/world-heritage-sites)<sup>4</sup> | June 2024 | Geospatial polygons of UNESCO World Heritage Sites in Scotland. |
| Anchor properties | `["UK_poi_locations"]` | [CDRC](https://data.cdrc.ac.uk/dataset/point-interest-data-united-kingdom)<sup>3</sup> | Sept 2024 | Points of interest geographic locations and classifications covering UK |
| Grid capacity | `["E_ENW_dfes_primaries"]` | [ENW DFES Primary Data](https://electricitynorthwest.opendatasoft.com/explore/dataset/dfes-2023-primary-data0/information/), Electricity North West Ltd<sup>14</sup> - [reference](https://www.enwl.co.uk/get-connected/network-information/dfes/) | 2023 | Distribution Future Energy Scenarios (DFES) primary substation demand forecasts for the North West region. |
| Grid capacity | `["E_ENW_ndp_headroom"]` | [ENW Network Development Plan](https://electricitynorthwest.opendatasoft.com/explore/dataset/ndp-pry-bsp-headroom/information/), Electricity North West Ltd<sup>14</sup> - [reference](https://www.enwl.co.uk/get-connected/network-information/network-development-plan/) | 2024 | Network Development Plan (NDP) primary substation headroom data for the North West region. |
| Grid capacity | `["E_ENW_ndp_voronoi"]` | [ENW Primary Substation Areas](https://electricitynorthwest.opendatasoft.com/explore/dataset/ndp-pry-voronoi/information/), Electricity North West Ltd<sup>14</sup> | 2024 | Primary substation service area boundary polygons for the North West region. |
| Grid capacity | `["E_NPg_heatmap"]` | [NPg Heat Map Data](https://northernpowergrid.opendatasoft.com/explore/dataset/heatmapdemanddata/table/), Northern Powergrid<sup>15</sup> | 2024 | Primary substation demand heat map data for the North East/Yorkshire region. |
| Grid capacity | `["E_NPg_ndp_demand"]` | [NPg Network Development Plan](https://northernpowergrid.opendatasoft.com/explore/dataset/npg_ndp_demand_headroom/information/), Northern Powergrid<sup>15</sup> | 2024 | Network Development Plan demand and headroom data for primary substations in the North East/Yorkshire region. |
| Grid capacity | `["S_SPEN_spd_substations"]` | [SP Distribution Primary Substations](https://spenergynetworks.opendatasoft.com/explore/dataset/distributed-generation-sp-distribution-heat-maps-spd-primary-substations/information/), SP Energy Networks SC389555<sup>14</sup> | Oct 2024 | Primary substation data for SP Distribution license area (South Scotland). |
| Grid capacity | `["W_SPEN_spm_substations"]` | [SP Manweb Primary Substations](https://spenergynetworks.opendatasoft.com/explore/dataset/distributed-generation-sp-manweb-heat-maps-spm-primary-substations/information/), SP Energy Networks SC389555<sup>14</sup> | Sep 2024 | Primary substation data for SP Manweb license area (North Wales). |
| Grid capacity | `["S_SPEN_spd_polygons"]` | [SP Distribution Network Areas](https://spenergynetworks.opendatasoft.com/explore/dataset/ndp-spd-primary-substation-polygons/information/), SP Energy Networks SC389555<sup>14</sup> | Aug 2024 | Primary substation service area boundary polygons for SP Distribution license area (South Scotland). |
| Grid capacity | `["W_SPEN_spm_polygons"]` | [SP Manweb Network Areas](https://spenergynetworks.opendatasoft.com/explore/dataset/ndp-spm-primary-group-polygons/information/), SP Energy Networks SC389555<sup>14</sup> | Apr 2024 | Primary substation service area boundary polygons for SP Manweb license area (North Wales). |
| Grid capacity | `["E_SSEN_demand"]` | [SSEN Heat Maps](https://data.ssen.co.uk/@ssen-distribution/generation-availability-and-network-capacity), SSEN Distribution<sup>14</sup> | Feb 2024 | Primary substation demand heat map data for SEPD (South England) license areas. |
| Grid capacity | `["S_SSEN_demand"]` | [SSEN Heat Maps](https://data.ssen.co.uk/@ssen-distribution/generation-availability-and-network-capacity), SSEN Distribution<sup>14</sup> | Aug 2023 | Primary substation demand heat map data for North Scotland license areas. |
| Grid capacity | `["SHET_SSEN_demand"]` | [SSEN Heat Maps](https://data.ssen.co.uk/@ssen-distribution/generation-availability-and-network-capacity), SSEN Distribution<sup>14</sup> | Aug 2023 | Primary substation demand heat map data for Shetland license areas. |
| Grid capacity | `["S_SSEN_sepd_bounds"]` | [SEPD Network Areas](https://data.ssen.co.uk/@ssen-distribution/primary-substation-boundaries), SSEN Distribution<sup>14</sup> | Nov 2023 | Primary substation service area boundary polygons for SEPD license area (South England). |
| Grid capacity | `["E_SSEN_shepd_bounds"]` | [SHEPD Network Areas](https://data.ssen.co.uk/@ssen-distribution/primary-substation-boundaries), SSEN Distribution<sup>14</sup> | Nov 2023 | Primary substation service area boundary polygons for SHEPD license area (North Scotland). |
| Grid capacity | `["E_UKPN_primaries"]` | [UKPN Network Capacity Map](https://ukpowernetworks.opendatasoft.com/explore/dataset/ukpn_primary_postcode_area/table/), UK Power Networks, Company number 3870728<sup>14</sup> | Oct 2024 | Primary substation data and service area mappings for South East England. |
| Grid capacity | `["EW_WPD_capacity"]` | [WPD Network Capacity Map](https://connecteddata.nationalgrid.co.uk/dataset/spatial-datasets), National Grid Electricity Distribution<sup>16</sup> | 2024 | Network capacity data for primary substations across WPD's regions. |
| Grid capacity | `["E_WPD_east_midlands_bounds"]` | [WPD East Midlands Network Areas](https://connecteddata.nationalgrid.co.uk/dataset/spatial-datasets), National Grid Electricity Distribution<sup>16</sup> | Jun 2024 | Primary substation service area boundary polygons for East Midlands region. |
| Grid capacity | `["W_WPD_south_wales_bounds"]` | [WPD South Wales Network Areas](https://connecteddata.nationalgrid.co.uk/dataset/spatial-datasets), National Grid Electricity Distribution<sup>16</sup> | Jun 2024 | Primary substation service area boundary polygons for South Wales region. |
| Grid capacity | `["E_WPD_south_west_bounds"]` | [WPD South West Network Areas](https://connecteddata.nationalgrid.co.uk/dataset/spatial-datasets), National Grid Electricity Distribution<sup>16</sup> | Jun 2024 | Primary substation service area boundary polygons for South West England region. |
| Grid capacity | `["E_WPD_west_midlands_bounds"]` | [WPD West Midlands Network Areas](https://connecteddata.nationalgrid.co.uk/dataset/spatial-datasets), National Grid Electricity Distribution<sup>16</sup> | Jun 2024 | Primary substation service area boundary polygons for West Midlands region. |
| Grid capacity | `["EW_lsoa_bounds"]` | [ONS](https://geoportal.statistics.gov.uk/datasets/ons::output-areas-december-2021-boundaries-ew-bfe-v9/about)<sup>3</sup> | December 2021 | Lower Level Super Output Area boundaries EW BFE. |

_Footnotes:_

a. Rural-urban classification codes are mapped to text descriptions according to the sources listed below:

- Scotland 8-fold code to text classification mapping: [Table 2.2, Scottish Government Urban Rural Classification](https://www.gov.scot/publications/scottish-government-urban-rural-classification-2020/pages/2/)
- Scotland 8-fold to 2-fold classification mapping: [Table 2.3, Scottish Government Urban Rural Classification](https://www.gov.scot/publications/scottish-government-urban-rural-classification-2020/pages/2/)
- England & Wales 10- and 2-fold code to text classification mapping: [page 19, ONS Postcode Directory (August 2023) User Guide](https://geoportal.statistics.gov.uk/datasets/a8db59f77e7542d092458426dbacfb90/about)

### Reweighting datasets

These datasets provide the target marginals from the census (and other sources) for reweighting the EPC data.

|             Config key             | Source                                                                                                                                                                         | Date                          | Description                                                                                                                            |
| :--------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------- | :------------------------------------------------------------------------------------------------------------------------------------- |
| `["EW_census_accommodation_type"]` | [ONS](https://www.ons.gov.uk/datasets/TS044/editions/2021/versions/4)<sup>4</sup>                                                                                              | Census 2021, updated Mar 2023 | Census 2021 household counts by property type per LSOA in England and Wales.                                                           |
| `["S_census_accommodation_type"]`  | [Scotland's Census](https://www.scotlandscensus.gov.uk/search-the-census#/search-by)<sup>4</sup> - [National Records of Scotland](https://www.nrscotland.gov.uk/)<sup>13</sup> | Census 2022                   | Census 2022 household counts by property type per Data Zone in Scotland.                                                               |
|       `["EW_census_tenure"]`       | [ONS](https://www.ons.gov.uk/datasets/TS054/editions/2021/versions/4)<sup>4</sup>                                                                                              | Census 2021, updated Mar 2023 | Census 2021 household counts by tenure per LSOA in England and Wales.                                                                  |
|       `["S_census_tenure"]`        | [Scotland's Census](https://www.scotlandscensus.gov.uk/search-the-census#/search-by)<sup>4</sup> - [National Records of Scotland](https://www.nrscotland.gov.uk/)<sup>13</sup> | Census 2022                   | Census 2022 household counts by tenure per Data Zone in Scotland.                                                                      |
|     `["EW_cdrc_dwelling_age"]`     | [CDRC](https://data.cdrc.ac.uk/dataset/dwelling-ages-and-prices/resource/data-dwelling-age-band-counts-2015)<sup>11</sup>                                                      | 2015                          | Residential dwelling ages, grouped into approximately 10-year age bands from pre-1900 to 2015, with counts of each age group per LSOA. |
|  `["EW_census_number_of_rooms"]`   | [ONS](https://www.ons.gov.uk/datasets/TS051/editions/2021/versions/4)<sup>4</sup>                                                                                              | Census 2021, updated Mar 2023 | Census 2021 household counts by number of rooms per LSOA in England and Wales.                                                         |

### Other datasets

This table contains information about other datasets used in the pipeline.

|          Config key          |                                                                              Source                                                                              |  Date of access  |                                                                                    Description                                                                                    |
| :--------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|      `["S_dz_lookup"]`       |                           [Scottish Government](https://opendata.scot/datasets/scottish+government-2011+data+zone+lookup/)<sup>4</sup>                           |   17 Mar 2025    |                                     Scottish geography lookup tables used for aggregation, from 2011 data zones to higher level geographies.                                      |
| `["EW_ons_lsoa_lad_lookup"]` |       [ONS](https://geoportal.statistics.gov.uk/datasets/ons::lsoa-2021-to-local-authority-districts-april-2023-best-fit-lookup-in-ew/explore)<sup>3</sup>       | 6 September 2024 |                                                            2021 LSOA to LAD lookup (April 2023) for England and Wales.                                                            |
|     `["S_data_zone_LA"]`     |                                    [Scottish Government](https://statistics.gov.scot/data/data-zone-lookup-2022)<sup>3</sup>                                     | 28 February 2025 |                                                            2022 Scottish data zone to higher level geographies lookup.                                                            |
|       `["EW_LSOA_LA"]`       |        [ONS](https://geoportal.statistics.gov.uk/datasets/ons::lsoa-2021-to-local-authority-districts-april-2023-best-fit-lookup-in-ew/about)<sup>3</sup>        | 28 February 2025 |                                                    LSOA (2021) to Local Authority Districts (April 2023) Best Fit Lookup in EW                                                    |
|     `["EW_LSOA_region"]`     | [ONS](https://www.data.gov.uk/dataset/c43641d8-710c-48e6-9139-1302953cf16c/lsoa-2021-to-bua-to-lad-to-region-december-2022-best-fit-lookup-in-ew-v2)<sup>3</sup> | 28 February 2025 | A best fit lookup between Lower layer Super Output Areas (LSOA), Built-up Areas (2022) to local authority districts (LAD) to regions as at 31 December 2022 in England and Wales. |

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
12. © Crown copyright. Reproduced with the permission of Registers of Scotland.
13. © Crown copyright. Data supplied by National Records of Scotland.
14. [Creative Commons Attribution](https://creativecommons.org/licenses/by/4.0/)
15. Supported by Northern Powergrid Open Data. [License](https://northernpowergrid.opendatasoft.com/p/opendatalicence/)
16. Supported by NGED Open Data. [License](https://www.nationalgrid.co.uk/open-data-licence)
