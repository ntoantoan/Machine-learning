#                                        Gradient Descent 

1. Giới thiệu 

   Hiểu đơn giản Gradient Descent là phương pháp tìm cực trị của một hàm số. Trong thực tế ta hay dùng đạo hàm để tìm điểm cực trị, tuy nhiên không phải phương trình đạo hàm nào cũng có thể giải được. Cách thức hoạt động của Gradient Desent là chúng sẽ tăng hoặc giảm các giá trị nghiệm để đạt được gần điểm cực tiểu nhất có thể.

   ![image](https://user-images.githubusercontent.com/42260182/102703658-4c78ff80-42a4-11eb-94ff-3906e166ca1d.png)



2. Gradient Descent cho hàm 1 biến

![image](https://user-images.githubusercontent.com/42260182/102703889-73850080-42a7-11eb-8023-09d3d1bf3ccd.png)



3. Ví dụ với python

![image-20201220094224439](C:\Users\TOAN\AppData\Roaming\Typora\typora-user-images\image-20201220094224439.png)



![image](https://user-images.githubusercontent.com/42260182/102703998-cf9c5480-42a8-11eb-9217-748e677533a3.png)



với cùng learning rate với x0 = 5 sau 11 iterations sẽ hội tụ còn với x0 = -5 sau 29 iterations hội tụ



Với learning rate khác nhau:

![image](https://user-images.githubusercontent.com/42260182/102704281-0aec5280-42ac-11eb-948b-e2e02907adbb.png)



với learning rate = .2 thì tốc độ học sẽ nhanh hơn so với learning rate = .1



4. Gradient Descent với hàm nhiều biến

Sự khác biệt giữa Gradient Descent với hàm nhiều biến chính là ta phải tối ưu một tập các tham số

công thức cập nhật là:

![image](https://user-images.githubusercontent.com/42260182/102704359-06746980-42ad-11eb-9cd6-123645750118.png)





* Giải bài toán Linear Regression với Gradient Descent



![image](https://user-images.githubusercontent.com/42260182/102704410-784cb300-42ad-11eb-8992-9aeddb2f249c.png)

![image](https://user-images.githubusercontent.com/42260182/102704420-9b776280-42ad-11eb-98f3-0d391fb4bed3.png)



Ta tiến hành code với python

giả sử ta tạo 1000 điểm dữ liệu được chọn gần với đường thẳng y =  4x+3 áp dụng theo Linear Regression tìm lại được phương đường thẳng là nghiệm của bài toán 

![image](https://user-images.githubusercontent.com/42260182/102704545-75eb5880-42af-11eb-8255-70b40538a8f9.png)



* Việc kiểm tra đạo hàm là việc vô cùng quan trọng vì việc tính đạo hàm thông thường khá phức tạp và rất dễ mắc lỗi, nếu chúng ta tính sai đạo hàm thì thuật toán GD sẽ không thể chạy đúng được. Trong thực nghiệm ta thường dùng 2 công thức dưới đây để tính đạo hàm 

![image](https://user-images.githubusercontent.com/42260182/102706134-fe72f480-42c1-11eb-846b-f78afe58f0b3.png)





Tiến hành code



![image](https://user-images.githubusercontent.com/42260182/102706377-f74ce600-42c3-11eb-8f98-6cacb2fa6269.png)



sau 47 vòng lặp thì thuật toán hội tụ khá gần với nghiệm được tìm theo công thức ở trên





* Biến thể của Gradient Descent



1. Batch Gradient Descent 

   Thuật toán gradient Descent ta nói từ đầu đến giờ còn đươc gọi là Batch Gradient Descent. Tức là ta phải dùng tất cả các điểm dữ liệu để tiến hành cập nhật tham số cách làm này có rất nhiều hạn chế và tốn kém rất nhiều thời gian nếu dữ liệu lớn

2. Stochastic Gradient Descent

   Sự khác biệt ở chỗ ta chỉ tính đạo hàm của hàm mất mát dựa trên một điểm dữ liệu Xi rồi cập nhật θ dựa trên đạo hàm này làm như vậy cho toàn bộ các điểm khác chứ không phải tính lại đạo hàm nữa

3. Mini-batch Gradient Descent

   mini-batch dùng một lượng lớn dữ liệu n, với n nhỏ hơn tập dữ liệu và được lấy ngẫu nhiên, mỗi lần cập nhật thuật toán lấy ra một mini-batch để tính đạo hàm rồi cập nhật 





* Một số chú ý khi thực hiện thuật toán trong thực nghiệm
  1. Giới hạn số vòng lặp: đây là phương pháp phổ biến nhất và cũng để đảm bảo rằng chương trình chạy không quá lâu. Tuy nhiên, một nhược điểm của cách làm này là có thể thuật toán dừng lại trước khi đủ gần với nghiệm.
  2. So sánh gradient của nghiệm tại hai lần cập nhật liên tiếp, khi nào giá trị này đủ nhỏ thì dừng lại. Phương pháp này cũng có một nhược điểm lớn là việc tính đạo hàm đôi khi trở nên quá phức tạp (ví dụ như khi có quá nhiều dữ liệu), nếu áp dụng phương pháp này thì coi như ta không được lợi khi sử dụng SGD và mini-batch GD.
  3. So sánh giá trị của hàm mất mát của nghiệm tại hai lần cập nhật liên tiếp, khi nào giá trị này đủ nhỏ thì dừng lại. Nhược điểm của phương pháp này là nếu tại một thời điểm, đồ thị hàm số có dạng *bẳng phẳng* tại một khu vực nhưng khu vực đó không chứa điểm local minimum (khu vực này thường được gọi là saddle points), thuật toán cũng dừng lại trước khi đạt giá trị mong muốn.
  4. Trong SGD và mini-batch GD, cách thường dùng là so sánh nghiệm sau một vài lần cập nhật. Trong đoạn code Python phía trên về SGD, tôi áp dụng việc so sánh này mỗi khi nghiệm được cập nhật 10 lần. Việc làm này cũng tỏ ra khá hiệu quả.



