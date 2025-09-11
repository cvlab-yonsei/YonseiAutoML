#bash SuperNet/scripts/train.sh cifar10 0 0 0
#bash SuperNet/scripts/train.sh cifar10 0 0 118
#bash SuperNet/scripts/train.sh cifar10 0 0 1217
#bash SuperNet/scripts/train.sh cifar10 0 0 5349 
#bash SuperNet/scripts/train.sh cifar10 0 0 5983 

#bash SuperNet/scripts/train_decom.sh cifar10 0 0 0
#bash SuperNet/scripts/train_decom.sh cifar10 0 0 5349 
#bash SuperNet/scripts/train_decom.sh cifar10 0 0 5983 
#bash SuperNet/scripts/train_decom.sh cifar10 0 0 118
#bash SuperNet/scripts/train_decom.sh cifar10 0 0 1217

#bash SuperNet/scripts/train_darts_ss.sh 0 0
bash SuperNet/scripts/train_darts_ss.sh 0 5349 
bash SuperNet/scripts/train_darts_ss.sh 0 5983

#bash SuperNet/scripts/search_all.sh cifar10 0 4 0
#bash SuperNet/scripts/search_all.sh cifar10 0 1 0
#bash SuperNet/scripts/search_all.sh cifar10 0 0 0 
