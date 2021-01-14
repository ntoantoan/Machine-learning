​                                                                      

#                                         NAIVE BAYES CLASSIFIER



**1. NAIVE BAYES CLASSIFIER**

Xét các bài toán phân lớp C Class khác nhau. Thay vì tìm ra chính xác label của mỗi điểm dữ liệu ![image](https://user-images.githubusercontent.com/42260182/104576762-33892180-568b-11eb-948a-a5fecc58ae60.png) ta có thể đi tìm xác suất để đầu ra đó rơi vào mỗi class C được kí hiệu là![image](https://user-images.githubusercontent.com/42260182/104577063-9f6b8a00-568b-11eb-87b4-c1d0c94d1d49.png) biểu thức này được hiểu là xác suất để đầu ra là class c biết rằng đầu vào là vector x. Biểu thức này nếu được tính có thể giúp xác định class của mỗi điểm dữ liệu bằng cách chọn ra class có xác suất rơi vào cao nhất 



![image](https://user-images.githubusercontent.com/42260182/104577787-7a2b4b80-568c-11eb-9a93-aab5c07beff4.png)



Biểu thức trên thường khó sử dụng để tính trực tiếp. Thay vào đó quy tắc Bayes thường được sử dụng:



![image](https://user-images.githubusercontent.com/42260182/104578027-c1194100-568c-11eb-9acc-da086887ef1f.png)



Giả thiết các chiều của dữ liệu độc lập với nhau nếu biết c, là quá chặt chẽ và ít khi tìm được dữ liệu mà các thành phần hoàn toàn độc lập với nhau. Tuy nhiên giả thiết ngây thơ này lại mang đến kết quả bất ngờ. Giả thiết về sự độc lập của các chiều dữ liệu này được gọi là Naive Bayes các xác định class của dữ liệu dựa trên giả thiết này có tên là Naive Bayes Classifier (NBC)

NBC nhờ vào tính đơn giản một cách ngây thơ, có tốc độ training và test rất nhanh việc này giúp nó mang lại hiệu quả cao trong các bài toán large-scale



+ Ở bước training, các phân phối ![image](https://user-images.githubusercontent.com/42260182/104578970-f4100480-568d-11eb-8985-6e77554a458f.png)  được xác định dựa vào traning data. Việc xác định các giá trị này có thể dựa vào Maximum Likelihood Estimation hoặc Maximum A Posteriori tôi đã trình bày ở phía trước.

+ Ở bước test, với mỗi điểm dữ liệu mới x, class của nó sẽ được xác định bởi:

  ![image](https://user-images.githubusercontent.com/42260182/104579219-46e9bc00-568e-11eb-990a-621d574dd4fc.png)

  với d lớn và các xác suất nhỏ, biểu thước này có thể được tính toán bằng cách lấy log của về phải

  ![image-20210114103218561](C:\Users\ntoan\AppData\Roaming\Typora\typora-user-images\image-20210114103218561.png)

  Việc lấy log này không ảnh hưởng đến kết quả vì log là một hàm số đồng biến trên tập các số dương





Mặc dù giả thiết Naive Bayes Classifiers sử dụng là quá phi thực tế, chúng vẫn hoạt động khá hiệu quả trong nhiều bài toán thực tế, đặc biệt là trong các bài toàn phân loại văn bản, ví dụ như lọc tin nhắn rác hay lọc email spam





**2. CÁC PHÂN PHỐI THƯỜNG DÙNG**



2.1 Gaussian Naive Bayes

Mô hình này được sử dụng chủ yếu trong loại dữ liệu mà các thành phần của nó là các biến liên tục.

![image](https://user-images.githubusercontent.com/42260182/104579909-2ec66c80-568f-11eb-82f5-858198897f67.png)



2.2 Multinomial Naive Bayes

Mô hình này chủ yếu được sử dụng trong phân loại văn bản mà feature vectors được tính bằng Bags of Words. Lúc này mỗi văn bản được biểu diễn bởi một vector có độ dài d chính là số từ trong từ điển. Giá trị của thành phần thứ i trong mỗi vector chính là số lần xuất hiện từ thứ i trong văn bản đó



![image](https://user-images.githubusercontent.com/42260182/104586096-94b6f200-5697-11eb-9b5b-c18631ae8af5.png)

* Hạn chế của cách tính này chính là nếu có một từ mới chưa bao giờ xuất hiện trong class c thì biểu thức trên sẽ bằng 0 dẫn đến kết quả sẽ trả về 0 bất kể các giá trị khác có lớn đến đâu để giải quyết vấn đề này ta sẽ sử dụng một kỹ thuật gọi là Laplace smoothing



![image](https://user-images.githubusercontent.com/42260182/104586717-6b4a9600-5698-11eb-9e96-efa42b80948d.png)

2.3 Bernoulli Naive Bayes



Mô hình này được áp dụng cho các loại dữ liệu mà mỗi thành phần là một giá trị binary bằng 0 hoặc bằng 1 công thức:



![image](https://user-images.githubusercontent.com/42260182/104586917-b82e6c80-5698-11eb-922c-f50e807a6478.png)

3. Ví dụ

![image](https://user-images.githubusercontent.com/42260182/104616864-60eec300-56bd-11eb-9929-b431e486e787.png)



![image](https://user-images.githubusercontent.com/42260182/104617001-87146300-56bd-11eb-9b6e-5c6995247fa2.png)



![image](https://user-images.githubusercontent.com/42260182/104617103-a01d1400-56bd-11eb-9f6f-086daa7650f2.png)





4. VÍ DỤ SỬ DỤNG THƯ VIỆN SKLEARN



```python
from __future__ import print_function
from sklearn.naive_bayes import MultinomialNB
import numpy as np 

# train data
d1 = [2, 1, 1, 0, 0, 0, 0, 0, 0]
d2 = [1, 1, 0, 1, 1, 0, 0, 0, 0]
d3 = [0, 1, 0, 0, 1, 1, 0, 0, 0]
d4 = [0, 1, 0, 0, 0, 0, 1, 1, 1]

train_data = np.array([d1, d2, d3, d4])
label = np.array(['B', 'B', 'B', 'N']) 

# test data
d5 = np.array([[2, 0, 0, 1, 0, 0, 0, 1, 0]])
d6 = np.array([[0, 1, 0, 0, 0, 0, 0, 1, 1]])

## call MultinomialNB
clf = MultinomialNB()
# training 
clf.fit(train_data, label)

# test
print('Predicting class of d5:', str(clf.predict(d5)[0]))
print('Probability of d6 in each class:', clf.predict_proba(d6))
```



Kết quả:

```
Predicting class of d5: B
Probability of d6 in each class: [[ 0.29175335  0.70824665]]
```





Ví dụ với Bernoulli Naive Bayes

```python
from __future__ import print_function
from sklearn.naive_bayes import BernoulliNB
import numpy as np 

# train data
d1 = [1, 1, 1, 0, 0, 0, 0, 0, 0]
d2 = [1, 1, 0, 1, 1, 0, 0, 0, 0]
d3 = [0, 1, 0, 0, 1, 1, 0, 0, 0]
d4 = [0, 1, 0, 0, 0, 0, 1, 1, 1]

train_data = np.array([d1, d2, d3, d4])
label = np.array(['B', 'B', 'B', 'N']) # 0 - B, 1 - N 

# test data
d5 = np.array([[1, 0, 0, 1, 0, 0, 0, 1, 0]])
d6 = np.array([[0, 1, 0, 0, 0, 0, 0, 1, 1]])

## call MultinomialNB
clf = BernoulliNB()
# training 
clf.fit(train_data, label)

# test
print('Predicting class of d5:', str(clf.predict(d5)[0]))
print('Probability of d6 in each class:', clf.predict_proba(d6))
```



```
Predicting class of d5: B
Probability of d6 in each class: [[ 0.16948581  0.83051419]]
```



