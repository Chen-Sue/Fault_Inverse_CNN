import numpy as np
from math import cos, sin, atan2, sqrt, pi ,radians, degrees
 
print("\n  Begin station.py ... \n")

t = time.time()

def center_geolocation(data):
    x, y = 0, 0
    lenth = len(data)
    for i in range(lenth):
        x += data[i, 0]
        y += data[i, 1]
    x = float(x / lenth)
    y = float(y / lenth)
    return (x, y)

def relative_coordinates(x, y, center_x, center_y):
    x_new = (x - center_x) * 111320   # x:纬度
    import math
    y_new = (y - center_y) * 111320  # y:经度
    return x_new, y_new

if __name__ == '__main__':
    data =np.array([[ 30.104, 105.33 ], [ 31.74, 105.925 ], [ 30.639, 104.065 ], [ 29.905, 107.232 ], [ 31.208, 107.507 ], \
                    [ 30.754, 107.187 ], [ 28.959, 102.77 ], [ 34.02, 102.058 ], [ 34.43, 104.023 ], [ 33.423, 104.815 ], \
                    [ 27.981, 107.726 ], [ 26.473, 106.669 ], [ 32.792, 102.548 ], [ 32.205, 105.56 ], [ 30.389, 104.545 ], \
                    [ 29.565, 103.755 ], [ 30.343, 106.939 ], [ 28.872, 105.414 ], [ 31.899, 102.229 ], [ 28.335, 103.134 ], \
                    [ 31.44, 104.726 ], [ 31.344, 106.065 ], [ 32.356, 106.833 ], [ 29.62, 105.119 ], [ 30.91, 103.757 ], \
                    [ 32.932, 100.741 ], [ 34.478, 100.249 ], [ 30.354, 103.306 ], [ 30.2, 104.103 ], [ 29.458, 104.434 ], \
                    [ 31.841, 106.745 ], [ 30.978, 101.123 ], [ 32.439, 105.852 ], [ 31.61, 100.019 ], [ 29.349, 102.632 ], \
                    [ 29.008, 101.5 ], [ 28.179, 104.516 ], [ 31.388, 100.672 ], [ 28.84, 103.534 ], [ 27.929, 101.275 ], \
                    [ 28.333, 102.174 ], [ 31.671, 103.85 ], [ 30.979, 105.882 ], [ 27.057, 102.716 ], [ 26.503, 101.744 ], \
                    [ 29.229, 102.354 ], [ 30.508, 105.562 ], [ 32.648, 103.582 ], [ 30.074, 102.765 ], [ 28.937, 99.803 ],  \
                    [ 31, 102.372 ], [ 28.651, 102.512 ], [ 27.432, 101.513 ], [ 33.126, 106.688 ], [ 31.928, 107.217 ], \
                    [ 32.075, 108.032 ], [ 29.981, 103.011 ], [ 31.221, 105.386 ], [ 28.798, 104.596 ], [ 31.076, 106.559 ], \
                    [ 26.108, 103.181 ], [ 26.408, 103.292 ], [ 26.696, 100.029 ], [ 25.724, 101.327 ], [ 25.689, 101.861 ], \
                    [ 26.683, 100.754 ], [ 27.823, 99.698 ], [ 31.006, 104.546 ], [ 31.651, 105.169 ], [ 30.11, 103.387 ], \
                    [ 30.673, 103.272 ], [ 30.976, 102.688 ], [ 29.843, 103.291 ], [ 30.16, 102.924 ], [ 30.302, 102.816 ], \
                    [ 30.447, 102.716 ], [ 30.71, 102.743 ], [ 30.949, 101.866 ], [ 30.066, 102.151 ], [ 30.108, 101.009 ], \
                    [ 29.846, 101.558 ], [ 27.719, 101.961 ]])      #  X : 纬度 南北方向       Y： 经度 东西方向
    # data[:, [0, 1]] = data[:, [1, 0]] # data垂直翻转     
    print("  0-max:{}, 0-min:{}, 1-max:{}, 1-min:{} ".format(max(data[:, 0]), min(data[:, 0]), max(data[:, 1]), min(data[:, 1])))    
    # 确定中心位置
    center = center_geolocation(data)
    print("  center: ", center)
    # 确定相对位置
    for i in range(len(data)):
        data[i, ] = relative_coordinates(data[i, 0], data[i, 1], center[0], center[1])
        if i == 1:
            print("  data: ", data[i, ])
    print("  0-max:{:.3f}, 0-min:{:.3f}, 1-max:{:.3f}, 1-min:{:.3f}".format(max(data[:, 0]), min(data[:, 0]), max(data[:, 1]), min(data[:, 1])))
    print("  data: ", data)    
    # data垂直翻转
    # data[:, [0, 1]] = data[:, [1, 0]] 
    f = open('staion_relation.txt','w')
    for i in range(len(data)):     
        if i !=  len(data)-1:
            f.write(" {0:.1f}  {1:.1f} ".format(data[i, 0], data[i, 1]))
        else:
            f.write(" {0:.1f}  {1:.1f} ".format(data[i, 0], data[i, 1]))
        if (i+1) % 1 == 0:
             f.write("\n")
    f.close()
    f1 = open('staion_relation1.txt','w')
    for i in range(len(data)):     
        if i !=  len(data)-1:
            f1.write(" ({0:.1f}, {1:.1f}), ".format(data[i, 0], data[i, 1]))
        else:
            f1.write(" ({0:.1f}, {1:.1f}) ".format(data[i, 0], data[i, 1]))
        if (i+1) % 1 == 0:
             f1.write("\n")
    f1.close()

print("\n  It takes {:.2f} minutes. ".format((time.time() - t)/60))
print("\n  End ... \n")

