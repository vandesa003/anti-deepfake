
def train(model_name, image_size):
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    header = ['Epoch', 'Learning rate', 'Time', 'Train Loss', 'Val Loss']

    if not os.path.isfile(snapshot_path + '/log.csv'):
        with open(snapshot_path + '/log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    df_all = pd.read_csv(csv_path)

    kfold_path_train = '../data/fold_5_by_study/'
    kfold_path_val = '../data/fold_5_by_study_image/'

    for num_fold in range(5):
        print('fold_num:',num_fold)

        with open(snapshot_path + '/log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([num_fold])

        f_train = open(kfold_path_train + 'fold' + str(num_fold) + '/train.txt', 'r')
        f_val = open(kfold_path_val + 'fold' + str(num_fold) + '/val.txt', 'r')
        c_train = f_train.readlines()
        c_val = f_val.readlines()
        f_train.close()
        f_val.close()
        c_train = [s.replace('\n', '') for s in c_train]
        c_val = [s.replace('\n', '') for s in c_val]

        # for debug
        # c_train = c_train[0:1000]
        # c_val = c_val[0:4000]

        print('train dataset study num:', len(c_train), '  val dataset image num:', len(c_val))
        with open(snapshot_path + '/log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['train dataset:', len(c_train), '  val dataset:', len(c_val)])
            writer.writerow(['train_batch_size:', train_batch_size, 'val_batch_size:', val_batch_size])

        train_transform, val_transform = generate_transforms(image_size)
        train_loader, val_loader = generate_dataset_loader(df_all, c_train, train_transform, train_batch_size, c_val, val_transform, val_batch_size, workers)

        model = eval(model_name+'()')
        model = model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00002)
        scheduler = WarmRestart(optimizer, T_max=5, T_mult=1, eta_min=1e-5)
        model = torch.nn.DataParallel(model)
        loss_cls = torch.nn.BCEWithLogitsLoss(pos_weight = torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).cuda())



if __name__ == '__main__':
    csv_path = '../dataset/metadata_train.json'
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-backbone", "--backbone", type=str, default='xception_change_header', help='backbone')
    parser.add_argument("-img_size", "--Image_size", type=int, default=256, help='image_size')
    parser.add_argument("-tbs", "--train_batch_size", type=int, default=32, help='train_batch_size')
    parser.add_argument("-vbs", "--val_batch_size", type=int, default=32, help='val_batch_size')
    parser.add_argument("-save_path", "--model_save_path", type=str,
                        default='xception_change_header', help='epoch')
    args = parser.parse_args()

    Image_size = args.Image_size
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    workers = 24
    backbone = args.backbone
    print(backbone)
    print('image size:', Image_size)
    print('train batch size:', train_batch_size)
    print('val batch size:', val_batch_size)
    snapshot_path = 'data_test/' + args.model_save_path.replace('\n', '').replace('\r', '')
    train(backbone, Image_size)
    # valid_snapshot(backbone, Image_size)

