import numpy as np
import matplotlib.pyplot as plot
import seaborn as sns

bd = [112, 62, 23, 34, 74, 33, 18, 17, 82, 47, 81, 19, 33, 111, 35, 66, 42]


#Kernel distribution of observed values
sns.distplot(bd)
plot.title("Kernel density of observed values")
plot.show()

#RANDOM NUMBER DRAWING FOR MODEL
alpha = 2.5
beta = 6
rand = np.random.beta(alpha, beta, 17)
rand_max = np.max(rand)
rand_min = np.min(rand)
scaled_rand = ((rand - rand_min)/(rand_max - rand_min)) * (np.max(bd) - np.min(bd)) + np.min(bd)
sns.distplot(rand)
plot.title("Kernel density estimation of Beta distribution")
plot.show()
print(rand)
print(np.sort(scaled_rand))
###########


avg = np.sum(bd)/len(bd)
variance = np.var(bd)
std = np.sqrt(variance)

print(np.sort(bd))
print("Average: " + str(avg))
print("Variance: " + str(variance))
print("Std: " + str(std))


bu = [20,11,7,3,6,4,7,3,13,8,15,7,19,3,6,8]
avg_bu = np.sum(bu)/len(bu)
var_bu = np.var(bu)
std_bu = np.sqrt(var_bu)

print(np.sort(bu))
print("Average: " + str(avg_bu))
print("Variance: " + str(var_bu))
print("Std: " + str(std_bu))
