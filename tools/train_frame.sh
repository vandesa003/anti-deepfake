CUDA_VISIBLE_DEVICES=3 python train_net_frame_level.py --model_saving_dir ../saved_models \
--log_file training_patches_ffhq_ResNext.log --batch_size 64 --model ResNext --n_epochs 30 \
--use_checkpoint 0 --base_lr 0.02 --acc_steps 8 --quick_test 0