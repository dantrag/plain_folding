import os
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import pickle
import os, os.path
import pickle
import sys
import re
"""
 Util functions for pkl files
--------------------------------------------------------------
"""

models=["pix2pix_unfolding_color_1000_fc_3_af_0p2",
		"cycle_gan_unfolding_1000_fc_3_af_0p2",
		"cycle_gan_horse2zebra"]


for model in models:
	source= "checkpoints/"+model + "/loss_log.txt"
	#read in logfile
	loss_logs= open(source,"r") 
	loss_logs_lines=loss_logs.readlines() 

	if "pix2pix" in model:
		g_gan=[]
		g_l1=[]
		d_real=[]
		d_fake=[]
		for lll in loss_logs_lines[1:]:
			g_gan.append(float(lll[lll.find("G_GAN:"):].split(" ")[1]))			
			g_l1.append(float(lll[lll.find("G_L1:"):].split(" ")[1]))
			d_real.append(float(lll[lll.find("D_real:"):].split(" ")[1]))
			d_fake.append(float(lll[lll.find("D_fake:"):].split(" ")[1].rstrip()))



		#plot result
		plt.plot(g_gan,label="g_gan")
		plt.plot(g_l1,label="g_l1")
		plt.plot(d_real,label="d_real")
		plt.plot(d_fake,label="d_fake")
		plt.xlabel("epochs/iters")
		plt.ylabel("Loss")
		plt.title(model)
		plt.legend(loc="upper right")
		plt.savefig("checkpoints/"+model + "/loss_log.png")
		plt.show()

	if "cycle" in model:
		d_a=[]
		g_a=[]
		cycle_a=[]
		idt_a=[]
		d_b=[]
		g_b=[]
		cycle_b=[]
		idt_b=[]
		for lll in loss_logs_lines[1:]:
			d_a.append(float(lll[lll.find("D_A:"):].split(" ")[1]))			
			g_a.append(float(lll[lll.find("G_A:"):].split(" ")[1]))
			cycle_a.append(float(lll[lll.find("cycle_A:"):].split(" ")[1]))
			idt_a.append(float(lll[lll.find("idt_A:"):].split(" ")[1]))
			d_b.append(float(lll[lll.find("D_B:"):].split(" ")[1]))			
			g_b.append(float(lll[lll.find("G_B:"):].split(" ")[1]))
			cycle_b.append(float(lll[lll.find("cycle_B:"):].split(" ")[1]))
			idt_b.append(float(lll[lll.find("idt_B:"):].split(" ")[1].rstrip()))



		#plot result
		plt.plot(d_a,label="d_a")
		plt.plot(g_a,label="g_a")
		plt.plot(cycle_a,label="cycle_a")
		plt.plot(idt_a,label="idt_a")
		plt.plot(d_b,label="d_b")
		plt.plot(g_b,label="g_b")
		plt.plot(cycle_b,label="cycle_b")
		plt.plot(idt_b,label="idt_b")
		plt.xlabel("epochs/iters")
		plt.ylabel("Loss")
		plt.title(model)
		plt.legend(loc="upper right")
		plt.savefig("checkpoints/"+model + "/loss_log.png")
		plt.show()
	


