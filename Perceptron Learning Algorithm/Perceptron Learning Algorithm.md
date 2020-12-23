#                      Perceptron Learning Algorithm





**1. GIỚI THIỆU**

Perceptron là một thuật toán phân loại cơ bản chỉ có 2 class cũng có thể gọi là binary classification 

* Phát biểu bài toán:

  *Cho hai class được gán nhãn, hãy tìm một đường phẳng sao cho toàn bộ các điểm thuộc class 1 nằm về 1 phía, toàn bộ các điểm thuộc class 2 nằm về phía còn lại của đường phẳng đó. Với giả định rằng tồn tại một đường phẳng như thế.*

  Nếu tồn tại một đường phẳng phân chia hai class thì ta gọi hai class đó là *linearly separable*. Các thuật toán classification tạo ra các boundary là các đường phẳng được gọi chung là Linear Classifier.

  

**2. THUẬT TOÁN PERCEPTRON (PLA)**

Cũng giống như các thuật toán lặp trong K-means Clustering và Gradient Descent, ý tưởng cơ bản của PLA là xuất phát từ một nghiệm dự đoán nào đó, qua mỗi vòng lặp nghiệm sẽ được cập nhật tới một vị trí tốt hơn. Việc cập nhật này dựa trên việc giảm giá trị của một hàm mất mát nào đó



![image](https://user-images.githubusercontent.com/42260182/102975759-23ba6980-4533-11eb-80ad-a5b444e94ba2.png)



![image](https://user-images.githubusercontent.com/42260182/102975822-3d5bb100-4533-11eb-8719-8a57b0c9182e.png)



**3. VÍ DỤ PYTHON**

Các ví dụ trong folder github