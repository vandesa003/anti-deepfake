CUDA_VISIBLE_DEVICES=8 python train_net.py --model_saving_dir ../saved_models/Resnet_batch_0322 \
--log_file Resnet_batch_0322.log --batch_size 32 --model Resnet --n_epochs 30 \
--use_checkpoint 0 --base_lr 0.0001 --acc_steps 1 --quick_test 0