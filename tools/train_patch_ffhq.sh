CUDA_VISIBLE_DEVICES=3 python train_net_patch_level_ffhq.py --model_saving_dir ../saved_models/ResNext_patch_ffhq_shuffle_no_blur_0315 \
--log_file ResNext_patch_ffhq_shuffle_no_blur_0315.log --batch_size 64 --model ResNext --n_epochs 30 \
--use_checkpoint 0 --base_lr 0.01 --acc_steps 8 --quick_test 0