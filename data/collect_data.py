from gym_minigrid.wrappers import *
import tqdm
import cv2

class OnEdgeDataCollector(object):
    def __init__(self,n,number_of_images=100):
        self.n  = n
        self.number_of_images = number_of_images
        self.env_name = "MiniGrid-Empty-{}x{}-v0".format(n+2,n+2)
        self.env = gym.make(self.env_name)
        self.env = RGBImgObsWrapper(self.env)
        self.print = True
        self.env.reset()

    def start(self):
        i = 0
        c_pos = 0
        c_neg = 0
        while c_pos < self.number_of_images or c_neg < self.number_of_images:
            done = False
            self.env.reset()
            while not done:
                a = self.env.render()
                label = self.is_on_edge(a)
                if label == "pos":
                    dir = "./Data/is_on_edge/pos/"
                    c_pos += 1
                    name = str(c_pos) + ".png"
                else:
                    dir = "./Data/is_on_edge/neg/"
                    c_neg += 1
                    name = str(c_neg) + ".png"
                noisy_a = self.noisy(a)
                i += 1
                cv2.imwrite(dir+name,cv2.cvtColor(obs["image"], cv2.COLOR_RGB2BGR))
                action = self.env.action_space.sample()
                obs,reward,done,info = self.env.step(action)
                if c_pos > self.number_of_images and c_neg > self.number_of_images:
                    break

    def noisy(self, image):
        # row, col, ch = image.shape
        # mean = 0
        # var = 2
        # sigma = var ** 0.5
        # gauss = np.random.normal(mean, sigma, (row, col, ch))
        # gauss = gauss.reshape((row, col, ch))
        # noisy = image + gauss
        # return noisy
        return image

    def is_on_edge(self,img):
        row = range(1,self.n+1)
        col = range(1,self.n+1)
        each_grid = int(img.shape[0] / (self.n+2))
        offset = int(each_grid / 2)
        flag = False
        for i in row:
            for j in col:
                check = False
                if i == 1 or i == self.n:
                    check = True
                else:
                    if j == 1 or j == self.n:
                        check = True
                if check:
                    pixel_index_row = each_grid * i + offset
                    pixel_index_col = each_grid * j + offset
                    if self.print:
                        print (pixel_index_row, pixel_index_col, 1)
                        print (img[pixel_index_row, pixel_index_col, 1])
                    if img[pixel_index_row, pixel_index_col, 0] == 255:
                        flag = True
                        break
            if flag:
                break
        self.print = False
        if flag:
            return "pos"
        else:
            return "neg"

if __name__ == "__main__":
    dc = OnEdgeDataCollector(n=6,number_of_images=500)
    dc.start()
    pass
