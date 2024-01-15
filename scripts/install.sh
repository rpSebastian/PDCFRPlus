git submodule update --init --recursive

. "${HOME}/miniconda3/etc/profile.d/conda.sh"
sudo apt-get install graphviz libgraphviz-dev gcc

conda create --name PDCFRPlus python=3.8.0 -y
conda activate PDCFRPlus
pip install -e .

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$SCRIPT_DIR/../

cd $ROOT_DIR/third_party/PokerRL
tar -xzvf texas_lookup.tar.gz
pip install -e .
