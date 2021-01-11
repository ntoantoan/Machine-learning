                        #                                      FEATURE ENGINEERING



**1.GIỚI THIỆU**

Feature Engineering là một tập các phương pháp tiền xử lý dữ liệu trước khi cho vào các mô hình học máy

Trong thực tế có 3 phương pháp chính

* Trích lọc feature 

   Trích lọc các thông tin chính có tác động đến quá trình phân loại

* Biến đổi feature

  Thêm hoặc bớt vào dữ liệu gốc để phù hợp với bài toán đặt ra

* Lựa chọn feature

  Thường là sẽ chọn m feature từ n feature gốc

Feature Engineering được ứng dụng trong các bài toán NLP, CV, Classification

Mô hình chung cho các bài toán Machine-Learning



**2. FEATURE EXTRACTION**

Trong thực tế các nguồn dữ liệu thường ở dạng thô, đến từ nhiều nguồn khác nhau như văn bản, hình ảnh, các phiếu thu thập, các hệ thống lưu trữ, các hệ thông web, app. Các nguồn dữ liệu này trước khi tiến hành xây dựng mô hình đều phải qua một quá trình trích lọc đặc trưng để biến dữ liệu thô thành các dữ liệu hữu ích cho bài toán

**2.1 FEATURE EXTRACTION cho Văn bản**

* Bags Of Words sẽ đếm số từ trong văn bản, đếm số từ xuất hiện trong câu và sau đó sắp xếp chúng theo một vị trí phù hợp trong vector

![image](https://user-images.githubusercontent.com/42260182/104146983-2a842000-53ff-11eb-8ad8-5ff222599511.png)



* TF-IDF

  Là một kỹ thuật sử dụng trong khai phá dữ liệu văn bản. Trọng số này được sử dụng để đánh giá tầm quan trọng của một từ trong một văn bản. Giá trị cao thể hiện độ quan trọng cao và nó phụ thuộc vào số lần từ xất hiện trong văn bản

  * TF: Term Frequency (Tần suất xuất hiện của từ) là số lần từ xuất hiện trong văn bản.

    ![image](https://user-images.githubusercontent.com/42260182/104147776-49d07c80-5402-11eb-817c-bfa7edc95117.png)

  * IDF: Inverse Doucument Frequency (Nghịch đảo tần suất của văn bản), giúp đánh giá tầm quan trọng của một từ. Hàm log có mục đích giảm mức độ ảnh hưởng của các từ xuất hiện nhiều lần mà đôi khi các từ đó không mang ý nghĩa trong văn bản.

  ![image](https://user-images.githubusercontent.com/42260182/104148584-69b56f80-5405-11eb-91d4-7412a2a93d9f.png)



Thực hiện tính toán:

![image-20210111121111482](C:\Users\TOAN\AppData\Roaming\Typora\typora-user-images\image-20210111121111482.png)



![image](https://user-images.githubusercontent.com/42260182/104149047-628f6100-5407-11eb-936a-85ac5a2c404d.png)



Một số Các phương pháp bỏ túi có thể tìm được ở các link như [Catch me if you can competition](https://www.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking), [bag of app](https://www.kaggle.com/xiaoml/bag-of-app-id-python-2-27392), [bag of event](http://www.interdigital.com/download/58540a46e3b9659c9f000372):

**2.2 TRÍCH LỌC ĐẶC TRƯNG TRONG XỬ LÝ ẢNH**

Với sự bùng nổ của deep-learning và hàng loại các model khủng đã được huấn luyện ta có thể dễ dàng sử dụng khiến cho những người tiếp cận với DL rất dễ dàng. Nhưng để xây dựng được một network cho riêng mình thì điều này không hề dễ chút nào!





**3. BIẾN ĐỔI DỮ LIỆU (FEATURE TRANSFORMATION)**

Các dữ liệu thường có sự khác biệt về đơn vị (scale), phân phối (distribution)... tác động không nhỏ đến khả năng hội tụ của các thuật toán Gradient. Để giải quyết vấn đề này ta có thể biến đổi dữ liệu về các chuẩn trong thống kê.

* Phần này xin phép được viết sau



**4. LỰA CHỌN TÍNH NĂNG (FEATURE SELECTION)**

Để xây dựng một mô hình Machine-learning hay Deep-learning thì điều quan trong nhất chính là dữ liệu những không hẳn toàn bộ dữ liệu đều cần thiết, đôi khi có những dữ liệu nhiễu gây ảnh hưởng ít nhiều đến mô hình của chúng ta. Khi có quá nhiều feature sẽ có một số hạn chế như sau

* 1. Tăng chi phí tính toán
  2. Qúa nhiều biến giải thích có thể dẫn đến overfiting. Tức là mô hình chạy tốt trên ở tập train nhưng không hiệu quả ở tập test
  3. Trong số các biến sẽ có những biến gây nhiễu và làm giảm chất lượng mô hình
  4. Rối loạn thông tin do không thể kiểm soát và hiểu hết các biến

Để khắc phục các vấn đề trên ta sẽ sử dụng một số phương pháp sau:

**4.1 Phương pháp thống kê**

Phần này xin phép được viết sau





**4.2 Sử dụng mô hình**

Đây là phương pháp thường xuyên được áp dụng trong các cuộc thi phân tích dữ liệu. Chúng ta sẽ dựa trên một số mô hình cơ sở để đánh giá mức độ quan trọng của các biến, 2 mô hình thường được dùng là Random Forest hoặc Linear Regression



```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

# Hồi qui theo RandomForest
rdFrt = RandomForestClassifier(n_estimators = 10, random_state = 1)
# Hồi qui theo LinearSVC
lnSVC = LinearSVC(C=0.01, penalty="l1", dual=False)
# Tạo một pipeline thực hiện lựa chọn biến từ RandomForest model và hồi qui theo logit
pipe1 = make_pipeline(StandardScaler(), SelectFromModel(estimator = rdFrt), logit)
# Tạo một pipeline thực hiện lựa chọn biến từ Linear SVC model và hồi qui theo logit
pipe2 = make_pipeline(StandardScaler(), SelectFromModel(estimator = lnSVC), logit)
# Cross validate đối với 
# 1. Mô hình logit
acc_log = cross_val_score(logit, X, y, scoring = 'accuracy', cv = 5).mean()
# 2. Mô hình RandomForest
acc_rdf = cross_val_score(rdFrt, X, y, scoring = 'accuracy', cv = 5).mean()
# 3. Mô hình pipe1
acc_pip1 = cross_val_score(pipe1, X, y, scoring = 'accuracy', cv = 5).mean()
# 3. Mô hình pipe2
acc_pip2 = cross_val_score(pipe2, X, y, scoring = 'accuracy', cv = 5).mean()

print('Accuracy theo logit:', acc_log)
print('Accuracy theo random forest:', acc_rdf)
print('Accuracy theo pipeline 1:', acc_pip1)
print('Accuracy theo pipeline 2:', acc_pip2)
```





**4.3 Sử dụng grid search**

Đây là phương pháp có thể coi là đáng tin cậy nhất trong việc lựa chọn biến quan trọng, Ý tưởng chính của phương pháo này đó là huấn luyện mô hình trên một tập dữ liệu con, lưu lại kết quả sau train, lập lại quá trình huấn luyện trên những mẫu con khác, so sánh chất lượng các mô hình dự báo để tìm ra một tập các biến tốt nhất. Phương pháp này còn được gọi là Exhaustive Feature Selection

* Sequential Feature Selection 

  Ta sẽ lấy N biễn dữ liệu trong M biến dữ liệu, ta đi qua toàn bộ N biến đó để lấy ra một bộ weight tốt nhất, sau đó ta cập nhật từ N+1 đến M biến cho đến khi đến M biến dữ liệu hoặc hàm loss function đạt giá trị nhỏ nhất



```python
!{sys.executable} -m pip install mlxtend
from mlxtend.feature_selection import SequentialFeatureSelector

selector = SequentialFeatureSelector(logit, scoring = 'accuracy', 
                                     verbose = 2, 
                                     k_features = 3,
                                     forward = False,
                                     n_jobs = -1)

selector.fit(X, y)
```





**4.4 Sử dụng Entropy trong lý thuyết thông tin**

Entropy là một đại lượng đo sự "có ảnh hưởng" giữa các biến và nhãn

![image](https://user-images.githubusercontent.com/42260182/104160549-e7887380-5423-11eb-8feb-91c183bf849c.png)

Thứ tự entropy càng lớn thì độ ảnh hưởng của feature đó càng lớn so với nhãn