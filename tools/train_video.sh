CUDA_VISIBLE_DEVICES=1 python -i train_net.py --model_saving_dir ../saved_models/CRNN_embedding_0329 \
--log_file CRNN_embedding_0329.log --batch_size 2 --model lstm --n_epochs 30 \
--use_checkpoint 0 --base_lr 0.0001 --acc_steps 8 --quick_test 0