# NGWMN
Scripts and files used to help manage data for the [National Groundwater Monitoring Network](https://cida.usgs.gov/ngwmn/)
* If you can't find data or a station or see a bad site location, please submit a descriptive issue

## Data Sources
* The UGS NGWMN Project compiles data from three different data sources
* These sources are connected to each other by a monitoring location id.  If this id is not present or messed up, connecting the data is a pain in the butt.

### Field Data
* Field data entry should be done on the Google Sheet `FieldDataEntry` in the shared drive under `Projects\NGWMN\Field_Data_and_Field_Sheets`

### State Lab Data
* State Lab data is received as raw text files downloaded from the State Lab Keith Henderson (khenderson@utah.gov) at the State Lab.
* Raw text files are kept in the directory `Projects\NGWMN\Chemistry_Results_Data\StateLab\`

### EPA Data
* These data are sent as spreadsheets from the EPA lab in Denver
* All bottles collected by the UGS and sent to the EPA are tracked using COCs, stored in the shared drive under `Projects\NGWMN\COCs_lab_forms` and sorted by year

## Stations
* Site locations and metadata are entered through the [NGWMN Location Registry](https://www.usgs.gov/apps/location-registry/)

## Data Archives
* Data are stored in the UGS SDE, then submitted to the EPA WQX using the [CDX portal](https://cdx.epa.gov/)
* The USGS NGWMN Portal hosts the EPA WQX data
* Site data are stored in teh UGS SDE, and submitted and updated on the [NGWMN Portal](https://cida.usgs.gov/ngwmn/)
