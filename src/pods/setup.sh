apt update
apt install vim -y
# ln -s /workspace/model_cache ~/.cache
# ln -s /workspace/work ~/work
# python3 -m pip install --upgrade pip && python3 -m pip install --upgrade -r ~/work/requirements.txt --no-cache-dir
python3 -m pip install --upgrade pip && python3 -m pip install --upgrade -r requirements.txt --no-cache-dir

# for kokoro tts synthesizer fallback
apt-get -qq -y install espeak-ng > /dev/null 2>&1

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.10/dist-packages/torch/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib/
# need libcudnn_ops_infer.so.8.
# apt-get -y install cudnn-cuda-11
# apt-get install libcudnn9-cuda-11

apt-get install -y imagemagick
sudo mv /etc/ImageMagick-6/policy.xml /etc/ImageMagick-6/policy.xml.backup