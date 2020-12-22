                                                                                        #                              LOGISTIC REGRESSION



**1. GIỚI THIỆU**

Trong khi hai mô hình tuyến tính Linear Regression và Perceptron Learning Algorithm  biểu diễn chung một dạng 

![image](https://user-images.githubusercontent.com/42260182/102713092-2e3cef00-42f8-11eb-984a-44d0482d7adb.png)

mô hình Logistic Regression đầu ra của mô hình này lại được thể hiện bằng xác suất và thường được sử dụng hơn trong bài toán classification

* Đầu ra dự đoán của Linear Regression:

  ![image](C:\Users\TOAN\Documents\102713124-72c88a80-42f8-11eb-9a40-9bab270a6bd0.png)

* Đầu ra dự đoán của  Perceptron Learning Algorithm:

  ![image-20201220192135353](C:\Users\TOAN\AppData\Roaming\Typora\typora-user-images\image-20201220192135353.png)

* Đầu ra dự đoán của Logistic Regression 

  ![image-20201220192214857](C:\Users\TOAN\AppData\Roaming\Typora\typora-user-images\image-20201220192214857.png)



**2. HÀM MẤT MÁT VÀ TỐI ƯU**

![image](https://user-images.githubusercontent.com/42260182/102841963-82e78380-4438-11eb-8004-fbe757d66b76.png)



![image](https://user-images.githubusercontent.com/42260182/102842021-a14d7f00-4438-11eb-88a8-9f5183842e7c.png)





![image](https://user-images.githubusercontent.com/42260182/102842043-aad6e700-4438-11eb-9821-12bdce499be4.png)

![image](https://user-images.githubusercontent.com/42260182/102842105-c8a44c00-4438-11eb-8d6f-6d77581637dc.png)





**3. VÍ DỤ VỚI PYTHON**

* Ví dụ 1 chiều

![image-20201222103439833](C:\Users\TOAN\AppData\Roaming\Typora\typora-user-images\image-20201222103439833.png)

Với kết quả tìm được, đầu ra yy có thể được dự đoán theo công thức: `y = sigmoid(-4.1 + 1.55*x)`. Với dữ liệu trong tập training, kết quả là:



```python
print(sigmoid(np.dot(w[-1].T, X)))
```



```
[[ 0.03281144  0.04694533  0.06674738  0.09407764  0.13102736  0.17961209
   0.17961209  0.24121129  0.31580406  0.40126557  0.49318368  0.58556493
   0.67229611  0.74866712  0.86263755  0.90117058  0.92977426  0.95055357
   0.96541314  0.98329067]]
```



```python
X0 = X[1, np.where(y == 0)][0]
y0 = y[np.where(y == 0)]
X1 = X[1, np.where(y == 1)][0]
y1 = y[np.where(y == 1)]

plt.plot(X0, y0, 'ro', markersize = 8)
plt.plot(X1, y1, 'bs', markersize = 8)

xx = np.linspace(0, 6, 1000)
w0 = w[-1][0][0]
w1 = w[-1][1][0]
threshold = -w0/w1
yy = sigmoid(w0 + w1*xx)
plt.axis([-2, 8, -1, 2])
plt.plot(xx, yy, 'g-', linewidth = 2)
plt.plot(threshold, .5, 'y^', markersize = 8)
plt.xlabel('studying hours')
plt.ylabel('predicted probability of pass')
plt.show()
```



![image](https://user-images.githubusercontent.com/42260182/102846012-79aee480-4441-11eb-8176-a39392d197b0.png)



* Ví dụ với dữ liệu 2 chiều 

-Được lưu trữ trong folder github



**4. MỘT VÀI TÍNH CHẤT CỦA LOGISTIC REGRESSION**



LOGISTIC REGRESSION thực ra được sử dụng nhiều trong cái bài toán classification

* BOUNDARY tạo bởi Logistic Regression có dạng tuyến tính

![image](https://user-images.githubusercontent.com/42260182/102846355-5a648700-4442-11eb-811f-1a097645b9f5.png)



**5. MỘT VÀI LƯU Ý**



![image](https://user-images.githubusercontent.com/42260182/102846418-841dae00-4442-11eb-8bc7-db850b58a40a.png)





