import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

obs_true = np.load('/home/siddhant/vikash/pytorch-a2c-ppo-acktr-gail/obs_test_.npy')[0]
obs_pred = np.load('model_states_new2.npy')

x = np.arange(101)

fig, ax = plt.subplots(nrows = 4, ncols =5, figsize=(24, 10))

for i in range(4):
    for j in range(5):
        if 5*i + j > 16:
            print('here')
            fig.delaxes(ax[3, 2])
            fig.delaxes(ax[3, 3])
            fig.delaxes(ax[3, 4])
            plt.tight_layout()
            plt.savefig('states.png')
            exit()

        ax[i, j].plot(x, obs_true[:, 5*i+j], color='blue', label='ground truth')
        ax[i, j].plot(x, obs_pred[:, 5*i+j], color='yellow', label='predicted')
        ax[i, j].set_title('state[' + str(5*i+j) + ']')
        ax[i, j].legend(loc='best')

