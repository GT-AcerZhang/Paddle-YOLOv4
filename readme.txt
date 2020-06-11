
cp ../data/data39638/yolov4.zip ./yolov4.zip

unzip yolov4.zip






# 最新代码
rm -f a.zip
zip -r a.zip ./conf* ./ppde* ./tool* ./test_im*

rm -f o.zip
zip -r o.zip ./output/ou*.jpg ./output/pa*.jpg ./output/up*.jpg ./output/sk*.jpg


pip install pycocotools
cd d*
cd d*
unzip ann*.zip
unzip val*.zip
unzip tes*.zip
unzip image_info*.zip
unzip train*.zip
cd ~/w*


为了让test_dev不报错，编辑下面的文件：
vim /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/pycocotools/__init__.py

from .mask import *
esc + :wq

--------------------------训练--------------------------
python train.py


rm -f train.txt
nohup python train.py>> train.txt 2>&1 &


--------------------------预测--------------------------
python demo.py


--------------------------eval--------------------------
python eval.py


rm -f eval.txt
nohup python eval.py>> eval.txt 2>&1 &





















