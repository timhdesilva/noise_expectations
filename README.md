# noise_expectations
Replication Code for "Noise in Expectations: Evidence from Analyst Forecasts", by Tim de Silva and David Thesmar

## Instructions to run code

Before the code can be run, each file needs to be opened. At the top of each code file, there is a section titled "TO BE ADJUSTED BY USER". These parameters need to be adjusted as described in order for the code to run. If a file does not have this section, then nothing needs to be adjusted to be run.

The order in which the files must be run is indicator by the number that starts each file name. Files with the same first number can be run in any order. The first file is `1_datacollection.sas`. After this file is run, the remaining files can be run from a Linux/Unix/Bash terminal by running `./run_py.sh`.

There are several requirements to run these files. Running `1_datacollection.sas` requires an active WRDS account and internet access to access the WRDS remote server. Running the files that start with `3_` are very computationally-intensive because they perform rolling estimations of the models described in the text with cross-validation. These files should be run on a computing cluster. By default, they will parallelize the cross-validation across all available threads.