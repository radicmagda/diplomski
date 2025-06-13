#pip install -r requirements.txt unnecessary since added to Dockerfile

mkdir ./datasets/GoPro
gdown --id 1zgALzrLCC_tcXKu_iHQTHukKUVT1aodI -O train_dataset.zip
unzip train_dataset.zip -d train_dataset/
rm train_dataset.zip

mv train_dataset/train datasets/GoPro 
rm -rf train_dataset

gdown --id 1abXSfeRGrzj2mQ2n2vIBHtObU6vXvr7C -O test_dataset.zip
unzip test_dataset.zip -d test_dataset/
rm test_dataset.zip

mv test_dataset/GoPro/test datasets/GoPro 
rm -rf test_dataset

python scripts/data_preparation/make_crops.py --dataset GoPro
#rm -rf datasets/GoPro/train/input
#rm -rf datasets/GoPro/train/target