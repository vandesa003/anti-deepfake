CUDA_VISIBLE_DEVICES=2 python train_net.py --model_saving_dir ../saved_models/Xception_3d_concat_0320 \
--log_file Xception_3d_concat_0320.log --batch_size 64 --model video --n_epochs 30 \
--use_checkpoint 0 --base_lr 0.01 --acc_steps 1 --quick_test 0