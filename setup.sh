wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p miniconda3 -s
rm Miniconda3-latest-Linux-x86_64.sh
conda=miniconda3/bin/conda
$conda create -y -n pytorch python=3.8 pip
$conda activate pytorch
$pip install numpy
$pip install opencv-python
$pip install soundfile
$conda deactivate pytorch
$conda install -n pytorch -y pytorch torchvision cudatoolkit=10.1 -c pytorch
