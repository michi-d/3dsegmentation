from seg3d.data import Fake3DDataset


def main():
    
    L = 3
    seed = 0
    save_name = 'vol_train_set.h5'
    
    data = Fake3DDataset(L=L, seed=seed, h5path=save_name,
                         L_xy=128, L_z=128, NF_size=45, cube_kernel_size=2,
                         simplify_factor=1.25)

    

if __name__ == '__main__':

    main()
