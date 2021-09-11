                      Please use the branch for_bats_dataset and not the master


### Setup
The following files need to be adapted in order to run the code on your own machine:
- Change the file paths to the datasets in `utils/mypath.py`, e.g. `/path/to/dataset`.
- Specify the output directory in `configs/env.yml`. All results and model checkpoints will be stored under this directory. 

### Train model
The configuration files can be found in the `configs/` directory. The training procedure consists of the following steps:
- __STEP 1__: Solve the pretext task i.e. `simclr.py`. The configuration is given in `configs/pretext/simclr_batsnet.yml`
- __STEP 2__: Perform the clustering step i.e. `scan.py`. The configuration is given in `configs/pretext/scan_batsnet.yml`
- __STEP 3__: Perform the self-labeling step i.e. `selflabel.py`. The configuration is given in `configs/pretext/selflabel_batsnet.yml`

For example, run the following commands sequentially to perform clustering on our dataset:
```shell
python simclr.py --config_env configs/env.yml --config_exp configs/pretext/simclr_batsnet.yml
python scan.py --config_env configs/env.yml --config_exp configs/scan/scan_batsnet.yml
python selflabel.py --config_env configs/env.yml --config_exp configs/selflabel/selflabel_batsnet.yml
```
### Running via Notebook (on Google Colab)
A notebook has already been setup which will fetch the required data, install all the required packages, fetch the code, run the algorithm and output results in a `csv` format. Please use `SCAN colab.ipynb`. The details are given below:
* Clone the code from github repository. Or the code folder (provided) could also be uploaded manually in the notebook directory. 
* Checkout to the required branch using: `git checkout for_bats_dataset`
* Setup environment and instsall packages using `conda env create -f '/content/SCAN-algorithm/requirements_[algo].yml'`
* Connect to Google Drive and copy all the required data which needs to be clustered. You can also upload the data manually.
* Run the pre-text stage.
* Run the SCAN stage.
* Run the self-label stage (if applicable).
* The output is provided in `csv` format in the directory: `/content/results_scan.csv`

## Citation
This Algorithms is taken from:
```bibtex
@inproceedings{wvangansbeke2020scan,
  title={SCAN: Learning to Classify Images without Labels},
  author={Van Gansbeke, Wouter and Vandenhende, Simon and Georgoulis, Stamatios and Proesmans, Marc and Van Gool, Luc},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020}
}
```
