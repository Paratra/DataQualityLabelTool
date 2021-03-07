import numpy as np
import glob

def main():
    data_path_list = glob.glob(r'../data/labled_*.npy')

    for ind, each_data_path in enumerate(data_path_list):
        each_data = np.load(each_data_path)
        if ind == 0:
            result_data = each_data.copy()
        else:
            result_data = np.concatenate((result_data, each_data),0)

    np.save('../data/classifier_train_data', result_data)
    print('Data combined!')
    # import pdb; pdb.set_trace()

# Run the program
if __name__ == "__main__":
    main()
