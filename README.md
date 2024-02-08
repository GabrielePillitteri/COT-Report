# COT Report
This code utilizes the API provided by the CFTC to automatically import historical data from the Commitment of Traders (COT) report. Currently, data retrieval is limited to the Futures Legacy report, with plans to incorporate the disaggregated report and the Futures+Option report soon.

## Which Commodities?
The code allows extraction of information for any commodity covered by the COT. Upon execution, you will be prompted to specify from which market you want to obtain the data.

## Which Data?
The key data points extracted include open interest, short, long, and net positions of Non-commercial and Commercial entities. Additionally, for each variable, the Z-Score value is calculated to provide insight into operators' positioning compared to the previous year.

## Date
By defining "start" and "end" in the format of YYYY-MM-DD, you can choose your interval window. If no dates are specified, all data from 2010-01-01 will be selected by default.

## Table
Following the graphs, a table will be provided, reporting the most recent data available for Non-commercial Net and Commercial positioning, along with their respective Z-Scores and the latest reported open interest.
