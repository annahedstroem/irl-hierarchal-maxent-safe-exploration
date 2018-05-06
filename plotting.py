import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
def plot_policyquiver():
    file = os.path.join('plots', 'risky', 'worker.pi.npy')
    data = np.load(file)

    grid_size = np.sqrt(data.shape[0]).astype('int')
    direction = np.argsort(-data, axis=1)[:,0]
    placeholder = np.zeros(data.shape)
    placeholder[np.arange(0, data.shape[0]), direction] = 0.1
    # left right up down
    placeholder *= [-1, 1, 1, -1]
    #reshaped = data.reshape((grid_size, grid_size, data.shape[1]))
    arrow = placeholder.reshape((grid_size, grid_size, placeholder.shape[1]))

    # secondary arrow
    grid_size = np.sqrt(data.shape[0]).astype('int')
    direction = np.argsort(-data, axis=1)[:,1]
    placeholder = np.zeros(data.shape)
    placeholder[np.arange(0, data.shape[0]), direction] = 0.1
    # left right up down
    placeholder *= [-1, 1, 1, -1]
    #reshaped = data.reshape((grid_size, grid_size, data.shape[1]))
    arrow_secondary = placeholder.reshape((grid_size, grid_size, placeholder.shape[1]))

    X, Y = np.meshgrid(np.arange(0.0, grid_size,1.0), np.arange(grid_size,0.0,-1.0))
    X += 0.5
    Y += 0.5

    U = np.sum(arrow[:,:, [0, 1]],axis=-1)
    V = np.sum(arrow[:,:, [2, 3]], axis=-1)

    # secondary arrow
    U2 = np.sum(arrow_secondary[:,:, [0, 1]],axis=-1)
    V2 = np.sum(arrow_secondary[:,:, [2, 3]], axis=-1)
    fig = plt.figure(figsize=(5,5))
    plt.quiver(X,Y,U,V,  color='yellow', units='dots')
    plt.quiver(X, Y, U2, V2, color='#FFFF0044', linewidths=0.1)
    ax = plt.gca()
    ax.set_facecolor('black')
    ###  Wall
    ax.add_patch(
        patches.Rectangle(
            (4, 5),  # (x,y)
            1,  # width
            1,  # height
            color='red'
        )
    )
    #plt.ylim((1,10))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color='white', linestyle=':', linewidth=.5)
    plt.savefig('plots/risky_returnhome_gradient.png',
                bbox_inches='tight',
               pad_inches=0)
    plt.show()





    print('hello world')


plot_policyquiver()