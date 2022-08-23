# GMIM-GAN

## Requirements
...

## Usages

To run this project with high performance, please following the below commands. You can also load the models from `./save_models/`

- DLBP dataset

```shell
python3 train_rec.py --data_dir dataset/dblp/ --id dblp --struct_rate 1e-05 --GNN 2 --margin 0.1 --lambda 0.05 --num_epoch 500 --batch_size 256 --lr 0.001 --topn 10
```

- ML-100K dataset
```shell
python3 train_rec.py --data_dir dataset/movie/ml-100k/1/ --id ml100k --struct_rate 0.0001 --GNN 2 --margin 0.1 --lambda 0.2 --num_epoch 1000 --batch_size 256 --lr 0.0005 --topn 10
```


- Last-fm dataset
```shell
python3 train_rec.py --data_dir dataset/music/ --id ml100k --struct_rate 0.001 --GNN 2 --margin 0.1 --lambda 0.2 --num_epoch 1000 --batch_size 256 --lr 0.0005 --topn 20
```