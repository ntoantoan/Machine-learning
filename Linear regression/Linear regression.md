#                                         Linear Regression
**1. GIỚI THIỆU**

Trong phần này tôi sẽ giới thiệu một thuật toán học máy cơ bản nhất đó là Linear Regression (Hồi quy tuyến tính)

-Chúng ta đi thẳng vào vấn đề

-Gỉa sử có một căn nhà rộng x1 mét vuông, có x2 phòng ngủ,  cách trung tâm thành phố x3 km, có x4 con đường đi qua hỏi giá của căn phòng đó là bao nhiêu?

-Gỉa sử ta có 1000 căn nhà như vậy ta đã biết các thông số về số mét vuông, số phòng ngủ, cách trung tâm thành phố bao nhiêu km, và có bao nhiêu con đường đi qua liệu ta có đoán được giá của căn nhà này không?

-Các giải quyết:

Ta sẽ đi tìm một hàm số thỏa mãn đi qua các điểm bên trên ở đây với x = [*x1,x2,x3,x4*] là một vector chứa thông tin input và y = f(x) là một giá trị đầu ra

+Gọi hàm số đi qua các điểm dữ liệu trên là y = f(x) 

![image](https://user-images.githubusercontent.com/42260182/100987116-ac6f6680-3580-11eb-8046-14bb77cc7c2a.png)

trong đó *w1, w2, w3, w4* là các hệ số còn *w5* là bias nhiệm vụ của chúng ta là đi tìm các hệ số *w1, w2, w3, w4* tối ưu



**2. PHÂN TÍCH TOÁN HỌC**



![image](https://user-images.githubusercontent.com/42260182/100986765-39fe8680-3580-11eb-84a6-626f12b14029.png)





![image](https://user-images.githubusercontent.com/42260182/100986942-6fa36f80-3580-11eb-8d15-3c68352fddcc.png)



![image](https://user-images.githubusercontent.com/42260182/100987054-91045b80-3580-11eb-946f-17088cbe0630.png)



**3. VÍ DỤ TRỰC TIẾP**

Thực nghiệm với ví dụ về chiều cao và cân nặng:

![image](https://user-images.githubusercontent.com/42260182/100992913-50f4a700-3587-11eb-9143-6a276127d19c.png)

* CODE

![image](https://user-images.githubusercontent.com/42260182/100992297-9795d180-3586-11eb-8b28-e88f81294f3f.png)

**Ví Dụ 2**

Ở ví dụ này tôi sẽ lấy một tập data ngẫu nhiên do mình tự tạo nó giống với ví dụ đầu tiên khi tôi giới thiệu về Linear Regression

![image](https://user-images.githubusercontent.com/42260182/100996514-b64a9700-358b-11eb-9432-e1a1ef0d42b6.png)

với value là giá trị output nó khá giống với giá nhà khi biết 4 tham số phía trước

Code

```python
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[2,1,2,1,2,3,1,1],
              [3,4,5,3,3,3,3,3],
              [5,6,6,6,3,5,5,6],
              [7,9,8,7,6,7,7,10]]).T
y = np.array([15,28,26,18,17,19,17,30])
one = np.ones((X.shape[0],1))
Xbar = np.concatenate((one,X),axis = 1)
A = np.dot(Xbar.T,Xbar)
b = np.dot(Xbar.T,y)
w = np.dot(np.linalg.pinv(A),b)
print(w)
y1 = w[1]*2 + w[2]*3 + w[3]*5+w[4]*7+w[0]
```

Kết quả:

```
[-14.20542636   0.07364341   2.53100775  -1.31395349   4.41860465]
17.895348837209553
```

Tôi đã thử với hàng đầu tiên trong bảng và kết quả dự đoán là 17.89 thay vì 15, điều này cũng rất dễ lý giải bởi vì tập data tôi lấy nó chứa rất nhiều nhiễu sẽ gây ảnh hưởng đến quá trình tính toán, tôi sẽ đề cập cách giải quyết khi gặp nhiễu



**4. NHẬN XÉT**

Thuật toán Linear Regression là một thuật toán đơn giản nhưng rất hiệu quả và được ứng dụng nhiều nhất trong thực tế với chi phí tính toán thấp và độ hiệu quả tốt với dữ liệu chuẩn.

Nhược điểm của thuật toán này là rất nhạy cảm với nhiễu, chỉ cần có 1 vài điểm nhiễu là khả năng dự đoán có thể giảm đi đáng kể trong trường hợp này ta sẽ có thể bỏ đi một vài thuộc tính đặc biệt gây ảnh hưởng đến kết quả, ví dụ dữ liệu quá lớn hoặc quá nhỏ nó ta có thể drop chúng hoặc thay thế chúng bằng một giá trị khác phù hợp hơn, Ngoài ra Linear Regression sẽ không biểu thị được với dự liệu phức tạp để giải quyết các dữ liệu phức tạp ta có thể xem các mô hình tôi sẽ trình bày ở các bài tiếp theo

Ta thấy Linear Regression là một mô hình đơn giản, lời giải cho phương trình đạo hàm bằng 0 cũng khá đơn giản. *Trong hầu hết các trường hợp, chúng ta không thể giải được phương trình đạo hàm bằng 0.*

Nhưng có một điều chúng ta nên nhớ, **còn tính được đạo hàm là còn có hy vọng**.