CUDA_VISIBLE_DEVICES=2 python train_net.py --model_saving_dir ../saved_models/Xception_batch_0322 \
--log_file Xception_batch_0322.log --batch_size 32 --model Xception --n_epochs 30 \
--use_checkpoint 0 --base_lr 0.01 --acc_steps 1 --quick_test 0