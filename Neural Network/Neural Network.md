                                                                                                 #                                             NEURAL NETWORK	



**1. GIỚI THIỆU**

Trải qua nhiều "Thăng trầm" thì những năm trở lại đây (>2012) neural network được phát triển vô cùng mạnh chúng có thể giải quyết các bài toán mà trước đây con người nghĩ sẽ không bao giờ giải quyết được, với sự nổi lên như cồn của deep-learning thì bài viết này của tôi sẽ đi sâu vào neural network nó là kiến trúc cội nguồn của deep-learning



Neural network là gì?

Con người khi sinh ra sau một thời gian có thể phân biệt được các loại con vật, phân biệt được giữa người này với người khác,  những việc tưởng trừng hiển nhiên như vậy nhưng thực ra trải qua hàng triệu năm tiến hóa bộ não con người đã học được cách để phân biệt, nhưng nó lại là nhiệm vụ rất khó đổi với máy tính. Neural network ra đời được lấy cảm hứng từ chính bộ não của con người, ở hiện tại bộ não của con người vẫn là một kiến trúc tuyệt vời nhất

Hoạt động của các nơ ron 

Neural là tính từ của neuron (nơ ron), network chỉ cấu trúc. Nơ ron là đơn vị cấu tạo của hệ thần kinh và là thành phần quan trọng nhất của não. Bộ não chúng ta khoảng 10 triệu nơ ron và mỗi nơ-ron lại liên kết khoảng 10.000 nơ ron khác



Ở mỗi nơ-ron có phần thân (soma) chứa nhân, các tín hiệu đầu vào qua sợi nhánh (dendrites) và các tín hiệu đầu ra qua sợi trục (axon) kết nối với các nơ-ron khác. Hiểu đơn giản mỗi nơ-ron nhận dữ liệu đầu vào qua sợi nhánh và truyền dữ liệu đầu ra qua sợi trục, đến các sợi nhánh của các nơ-ron khác. Mỗi nơ-ron nhận xung điện từ các nơ-ron khác qua sợi nhánh. Nếu các xung điện này đủ lớn để kích hoạt nơ-ron, thì tín hiệu này đi qua sợi trục đến các sợi nhánh của các nơ-ron khác. => Ở mỗi nơ-ron cần quyết định có kích hoạt nơ-ron đấy hay không. Tuy nhiên NN chỉ là lấy cảm hứng từ não bộ và cách nó hoạt động, chứ không phải bắt chước toàn bộ các chức năng của nó. Việc chính của chúng ta là dùng mô hình đấy đi giải quyết các bài toán chúng ta cần.



![image](https://user-images.githubusercontent.com/42260182/103162118-7f7c3f80-481e-11eb-8bef-832dd384eb8e.png)



**2. MÔ HÌNH NEURAL NETWORK**



**2.1 Mô hình tổng quát**

Layer đầu tiên chính là input layer, các layer ở giữa gọi là hidden layer, layer cuối cùng được gọi là output layer. Các hình tròn được gọi là node

Mỗi node trong hidden layer và output layer : 

• Liên kết với tất cả các node ở layer trước đó với các hệ số w riêng. 

• Mỗi node có 1 hệ số bias b riêng. 

• Diễn ra 2 bước: tính tổng linear và áp dụng activation function

![image-20201227083809290](C:\Users\TOAN\AppData\Roaming\Typora\typora-user-images\image-20201227083809290.png)



**2.2 Kí hiệu**

![image](https://user-images.githubusercontent.com/42260182/103162385-99b81c80-4822-11eb-80cf-77f0f08a360c.png)





![image](https://user-images.githubusercontent.com/42260182/103167827-2255ad80-4861-11eb-99dd-b9b78fa8603b.png)



**3. FEED FORWARD**



![image](https://user-images.githubusercontent.com/42260182/103167888-9ee88c00-4861-11eb-99b4-7670de1076d4.png)



![image](https://user-images.githubusercontent.com/42260182/103167908-c0497800-4861-11eb-92be-fb6be64f6de9.png)



**4. BACK-PROPAGATION**



![image](https://user-images.githubusercontent.com/42260182/103167929-e838db80-4861-11eb-9ddb-166fdf892116.png)

![image](https://user-images.githubusercontent.com/42260182/103167945-0acaf480-4862-11eb-8a2d-cb5be9faa62b.png)



![image](https://user-images.githubusercontent.com/42260182/103167953-19b1a700-4862-11eb-9258-cf63bd1d6df3.png)

![image](https://user-images.githubusercontent.com/42260182/103167967-351cb200-4862-11eb-8c0d-0597ffec6dc4.png)

![image](https://user-images.githubusercontent.com/42260182/103167977-4c5b9f80-4862-11eb-8802-672b8942c509.png)







**4.1 BỔ XUNG QUY TẮC CHUỐI + BACKPROPAGATION**

![image](https://user-images.githubusercontent.com/42260182/103168022-8f1d7780-4862-11eb-94e2-33078f0a7c88.png)





**4.2 VỚI NHIỀU LAYER**

![image](https://user-images.githubusercontent.com/42260182/103168223-f12aac80-4863-11eb-940a-9aae51d9b738.png)

![image-20201227165358715](C:\Users\TOAN\AppData\Roaming\Typora\typora-user-images\image-20201227165358715.png)



**5. TÓM TẮT LẠI MÔ HÌNH**

![image](https://user-images.githubusercontent.com/42260182/103168297-67c7aa00-4864-11eb-8176-5a219917019c.png)



![image](https://user-images.githubusercontent.com/42260182/103168386-6945a200-4865-11eb-8e11-33e3980e33f7.png)





**6. VÍ DỤ PYTHON**

Trong forlder neural-netword ở github

