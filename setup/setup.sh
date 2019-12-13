#!/bin/bash
# a bash script to setup the environment for the project

# install pip3 and virtual
sudo apt-get install python3-pip -y

# create the virutual environment in the project root
pip3 install virtualenv

# activate the project environment and install the packages
virtualenv --no-site-packages -p python3 capetownai_env
source capetownai_env/bin/activate
pip install -r setup/requirements.txt

pip install ipykernel
python -m ipykernel install --user --name CapeTownAI_kernel --display-name "CAPETOWN AI"