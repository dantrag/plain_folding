"""
Test on real shirt images and produce density estimate
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import pickle
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset_f = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    #webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    #laod the real image dataset:
    file = open("../datasets/cgan_dataset_real_test.pkl",'rb')
    dataset = pickle.load(file)
    file.close()
    print("Loaded test dataset")
    #for data in dataset_f:
    #    print(data)
    #    print(data['A'].shape)
    #    a=1/0
    titles = ["real", "real masked", "generated", "heat", "overlay"]
    for i, entry in enumerate(dataset):
        #if i >= opt.num_test:  # only apply our model to opt.num_test images.
        #    break
        real_img=entry[0]
        real_masked=entry[1]
        #real masked needs to be converted to the set input format
        real_masked=real_masked[None, :, :] * np.ones(3, dtype=int)[:, None, None]
        real_masked=np.expand_dims(real_masked,axis=0)
            
        data = {
                    "A": torch.from_numpy(real_masked).float(),
                    "B": torch.from_numpy(real_masked).float(),
                    "A_paths": "dummy",
                    "B_paths": "dummy"

                }
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        #canabalise the visual container
        visuals['real_A']=torch.from_numpy(real_img).float()
        #img_path = model.get_image_paths()     # get image paths
        fake_B_np = torch.squeeze(visuals['fake_B']).permute(1, 2, 0).detach().cpu().numpy()
        #norm 0-1
        fake_B_np=(fake_B_np+1)/2
        #make jet heat
        gray_img = cv2.cvtColor(fake_B_np, cv2.COLOR_BGR2GRAY)
        heatmap_img = cv2.applyColorMap((gray_img*255).astype('uint8'), cv2.COLORMAP_JET)
        heatmap_img=cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(heatmap_img, 0.5, real_img, 0.5, 0)

        images = (real_img, (entry[1][:, :, None] * np.ones(3, dtype=int)[ None, None, :]*255).astype('uint8'), fake_B_np,heatmap_img,overlay) 
        for j in range(len(images)):
            plt.subplot(1,5,1+j)
            plt.axis("off")
            plt.imshow(images[j])
            plt.title(titles[j])
        print(web_dir+ str(i).zfill(3)+ '.png')
        plt.tight_layout()
        plt.savefig(web_dir+ str(i).zfill(3)+ '.png')
        #plt.show()
        plt.close()
        plt.clf()

        #if i % 5 == 0:  # save images to an HTML file
        #    print('processing (%04d)-th image... %s' % (i, img_path))
        #save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    #webpage.save()  # save the HTML
