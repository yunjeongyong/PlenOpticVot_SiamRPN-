import numpy as np
import matplotlib.pyplot as plt

iou1 = open('data/2D 객체추적 영상1(GT 안보이게) Video3/result_iou.txt', 'r')
iou2 = open('data/3D 객체추적 영상4(GT 보이게) Video3/result_iou.txt', 'r')

output1 = iou1.readline()
output1 = output1.split(',')
output1 = output1[:len(output1)-1]

output2 = iou2.readline()
output2 = output2.split(',')
output2 = output2[:len(output2)-1]

output1 = list(map(float, output1))
output2 = list(map(float, output2))

output1 = np.asarray(output1)
output2 = np.asarray(output2)

iou1.close()
iou2.close()

x_values = list(range(1, len(output1) + 1))

print(output1)
print(output2)
print(len(x_values))

plt.title('2D vs 3D_choice IOU(Video3)')
plt.plot(x_values, output1, color='black', label='2D')
plt.plot(x_values, output2, color='red', label='3D_all')
plt.legend(loc='lower left')
plt.grid()
plt.show()
