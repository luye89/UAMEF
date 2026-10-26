import argparse

def args_parser():
    parser = argparse.ArgumentParser(description='arguments')

    parser.add_argument('--data_dir', type=str, help='Path to the data',
                        default='data124')
    parser.add_argument('--normalizer_state', type=str, default=None, help='Path to a state file of a normalizer. Leave none if you want to use one of the provided ones.')
    parser.add_argument('--resize', default=256, type=int, help='number of epochs to train')
    parser.add_argument('--crop', default=224, type=int, help='number of epochs to train')

    parser.add_argument('--task', type=str, default='kcat,km,ki')
    parser.add_argument('--epochs', type=int, default=100, help='number of chunks to train')
    parser.add_argument('--device', type=str, default="0", help='cuda:number or cpu')
    parser.add_argument('--num_workers', type=int, default=8, help='num_workers for dataloader')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--model', type=str, default='our', help='our model or baselines')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=128)

    # args = argParser.parse_args()
    return parser
