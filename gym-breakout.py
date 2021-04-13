# from gym_minigrid.wrappers import *
import matplotlib.pyplot
import cv2
import gym
import sys
import numpy as np
import random as rand
from gym.wrappers.pixel_observation import PixelObservationWrapper

class Breakout(object):
	def __init__(self, fluent):
		self.env = gym.make("Breakout-v0")
		self.env.reset()
		# self.env = RGBImgObsWrapper(self.env)
		self.env = PixelObservationWrapper(self.env)
		self.dir = "./breakout_data/"+fluent+"/"
		self.c_neg = 0
		self.c_pos = 0

		## BRICK COLORS ##
		red = [200, 72, 72]
		orange = [198, 108, 58]
		clay = [180, 122, 48]
		yellow = [162, 162, 42]
		green = [72, 160, 72]
		blue = [66, 72, 200]
		self.sum_array = [sum(red), sum(orange), sum(clay), sum(yellow), sum(green), sum(blue)]
		
		self.row_begin = -1
		self.row_end = -1
		self.col_begin = -1
		self.col_end = -1

		self.n_bricks_per_row = 16
		self.n_brick_rows = 6
		self.brick_width = -1
		self.brick_height = -1


	def synthesize_env(self, img):
		rows = img.shape[0]
		cols = img.shape[1]

		## Find the first and last column index where bricks exist ##
		
		for row in range(30, rows):
			if sum(img[row][int(cols/2)]) in self.sum_array:
				self.row_begin = row
				break

		
		for row in range(rows-1, 30, -1):
			if sum(img[row][int(cols/2)]) in self.sum_array:
				self.row_end = row
				break

		
		for col in range(cols):
			if sum(img[self.row_begin][col]) in self.sum_array:
				self.col_begin = col
				break
		
		
		for col in range(cols-1, 0, -1):
			if sum(img[self.row_begin][col]) in self.sum_array:
				self.col_end = col
				break

		## Brick dimensions ##
		self.brick_width = int((self.col_end - self.col_begin + 1)/self.n_bricks_per_row)
		self.brick_height = int((self.row_end - self.row_begin + 1)/self.n_brick_rows)

		# print(row_begin, row_end, col_begin, col_end, brick_width, brick_height)


	def start(self, fluent):
		frame = self.env.render(mode='rgb_array')
		self.synthesize_env(frame)
		# cv2.imshow('someshit', img)
		# print(img.shape)
		# cv2.waitKey(0)

		if fluent == "is_clear_left":
			label = self.is_clear_left(frame)
		else:
			print("NO ARGUEMENT PASSED")

		# action = self.env.action_space.sample()
		# obs,reward,done,info = self.env.step(action)
		# print(obs['pixels'].shape)
		# cv2.imwrite(dir+"sample.png",cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
		# self.env.ale.saveScreenPNG(dir+'test_image.png')
		# img = self.env.ale.getScreenRGB2(a)

	def generate_samples(self, original_image, label):
		image = original_image.copy()
		for row_s in range(self.row_end, self.row_begin-1, -1*self.brick_height):
			random_bricks = np.random.randint(self.n_bricks_per_row, size=10)
			## Clear bricks row wise
			combined_sample = image.copy()
			for brick in random_bricks:
				sample = image.copy()
				## Clear a single brick
				for r in range(row_s, row_s-self.brick_height, -1):
					col_s = self.col_begin+brick*self.brick_width 
					for c in range(col_s,col_s+self.brick_width):	
						sample[r][c] = np.array([0, 0, 0]) 
						combined_sample[r][c] = np.array([0, 0, 0]) 
				if label == 'pos':
					self.c_pos += 1
					name = self.dir+label+"/sample"+str(self.c_pos)+".png"
				else:
					self.c_neg += 1
					name = self.dir+label+"/sample"+str(self.c_neg)+".png"
				# cv2.imwrite(name, cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))
				cv2.imwrite(name, cv2.cvtColor(combined_sample, cv2.COLOR_RGB2BGR))	


			## Clear all bricks in the row bottom up in the image
			for r in range(row_s, row_s-self.brick_height, -1):
				for c in range(self.col_begin, self.col_end):
					image[r][c] = np.array([0, 0, 0]) 

		return

	def is_clear_left(self,img):
		for stride in range(1, int(self.n_bricks_per_row/3)+1):
			self.generate_samples(img, 'neg')
			for col in range(self.col_begin, self.col_begin+self.brick_width*stride):
				for row in range(self.row_begin, self.row_end+1):
					if sum(img[row][col]) in self.sum_array:
						img[row][col] = np.array([0, 0, 0])

			# cv2.imwrite("./samples/test.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))	
		self.generate_samples(img, 'pos')
				

			# cv2.imshow('someshit', img)
			# cv2.waitKey(0)

		


if __name__ == "__main__":
	fluent = sys.argv[1]
	bo = Breakout(fluent)
	bo.start(fluent)
	pass



