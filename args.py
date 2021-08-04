def get_arguments():
    import argparse
    from viclassifier import cfgs

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_directory',
                        action='store',
                        default=cfgs.data_directory,
                        dest='data_directory',
                        type=str,
                        help='training data directory')
    # parser.add_argument('--num_labels',
    #                     action='store',
    #                     dest='num_labels',
    #                     default=cfgs.num_labels,
    #                     type=int,
    #                     help='Number of categories')
    parser.add_argument('--arch',
                        action='store',
                        dest='arch',
                        choices=cfgs.archs,
                        default=cfgs.arch,
                        type=str,
                        help='Base pre-trained network')
    parser.add_argument('--num_epochs',
                        action='store',
                        dest='num_epochs',
                        default=cfgs.num_epochs,
                        type=int,
                        help='Iterations for gradient descent')
    parser.add_argument('--batch_size',
                        action='store',
                        default=cfgs.batch_size,
                        dest='batch_size',
                        type=int,
                        help='Batch size for gradient descent')
    parser.add_argument('--learning_rate',
                        action='store',
                        default=cfgs.learning_rate,
                        dest='learning_rate',
                        type=float,
                        help='Learning rate for gradient descent')
    return parser

