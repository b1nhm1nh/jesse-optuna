# Jesse optuna for Jesse-CLI using Jesse-tk

Only works with the CLI version / branch of jesse.

The config.yml should be self-explainatory.

# Installation

```sh
# install from git
pip install git+https://github.com/cryptocoinserver/jesse-optuna.git

# cd in your Jesse project directory

# create the config file
jesse-optuna create-config

# create the database for optuna 
jesse-optuna create-db optuna_db

# edit the created yml file in your project directory 

# run optimize
jesse-optuna run --config optuna-config.yml

# run walkforward optimize

jesse-optuna walkforward [start-date] [end-date] [inc months] [training months] [walkforward length months] --config optuna-config.yml

# 4 months period: 3 months for training, 1 months for testing. Walk 2 months after that
jesse-optuna walkforward 2020-01-01 2021-01-01 2 3 4

# 6 months period: 4 months for training, 2 months for testing. Walk 3 months after that
jesse-optuna walkforward 2020-01-01 2021-01-01 3 4 6
```


## Disclaimer
This software is for educational purposes only. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS. Do not risk money which you are afraid to lose. There might be bugs in the code - this software DOES NOT come with ANY warranty.
