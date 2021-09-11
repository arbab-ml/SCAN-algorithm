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

For example, run the following commands sequentially to perform our method on CIFAR10:
```shell
python simclr.py --config_env configs/env.yml --config_exp configs/pretext/simclr_batsnet.yml
python scan.py --config_env configs/env.yml --config_exp configs/scan/scan_batsnet.yml
python selflabel.py --config_env configs/env.yml --config_exp configs/selflabel/selflabel_batsnet.yml
```
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
