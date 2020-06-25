

# 在本地windows运行1_pytorch2paddle.py，会生成一个yolov4文件夹，就是我们的预训练模型了。
# 把yolov4文件夹打包成zip，通过AIStudio的“创建数据集”将zip包上传。
# 仓库使用这个数据集和COCO2017数据集，就可以完成预训练模型上传了。
# 进入AIStudio，把上传的预训练模型解压：
cd ~/w*
cp ../data/data39638/yolov4.zip ./yolov4.zip
unzip yolov4.zip


# 安装依赖、解压COCO2017数据集
cd ~
pip install pycocotools
cd data
cd data7122
unzip ann*.zip
unzip val*.zip
unzip tes*.zip
unzip image_info*.zip
unzip train*.zip
cd ~/w*



--------------------------训练--------------------------
rm -f train.txt
nohup python train.py>> train.txt 2>&1 &


--------------------------预测--------------------------
python demo.py


--------------------------eval--------------------------
rm -f eval.txt
nohup python eval.py>> eval.txt 2>&1 &



--------------------------test-dev--------------------------
rm -f test_dev.txt
nohup python test_dev.py>> test_dev.txt 2>&1 &



--------------------------导出--------------------------
python export_model.py




python deploy_infer.py --model_dir inference_model --image_file images/test/000000013659.jpg
















