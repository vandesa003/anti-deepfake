CUDA_VISIBLE_DEVICES=3 python train_net_patch_level.py --model_saving_dir ../saved_models/Xception_patch_concat_0322 \
--log_file Xception_patch_concat_0322.log --batch_size 64 --model Xception --n_epochs 30 \
--use_checkpoint 0 --base_lr 0.0001 --acc_steps 8 --quick_test 0