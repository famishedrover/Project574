import pickle 
import numpy as np
import random 

np.random.seed(1234)

random.seed(1234)

# load the model from disk
filename = "final_model.sav"
loaded_model = pickle.load(open("./concepts/"+filename, 'rb'))


x = np.random.random((1,128))
print (loaded_model.predict(x))
