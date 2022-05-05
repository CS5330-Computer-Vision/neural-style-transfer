from training import train
from application import apply


def main():
    # for padding_type in ['reflection', 'zero', 'replication']:
    #     print('*' * 25 + f' Testing the {padding_type} type of the padding method ' + '*' * 25)
    #     train(testing=True, padding_type=padding_type)
    #
    # for upsampling_type in ['nearest', 'linear', 'bilinear', 'bicubic']:
    #     print('*' * 25 + f' Testing the {upsampling_type} type of the upsampling method ' + '*' * 25)
    #     train(testing=True, padding_type=padding_type)
    #
    # train()

    apply()


if __name__ == '__main__':
    main()