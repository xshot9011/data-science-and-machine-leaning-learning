# Image Classification

เริ่มมาด้วยเื้อเรื่องของปัญหาเลย มีชายญี่ปุ่นชื่อมาโกโตะ เขาช่วยทำงานอยู่ที่ฟามแตงกว่าของพ่อแม่ สิ่งที่เขาก็รู้ก็คือพ่อแม่ของเราทำการจัดเรียงแตงกวาโดยคุณภาพของมัน
แล้วการจัดเรียงคุณภาพของแตงกวาเนี่ยแม่งเสียเวลามากถ้าใช้คนทำ เขาก็เลยอยากใช้ image classification ช่วยในการจัดกลุ่ม
ตอนแรกก็ถ่ายรูปแปะ label ให้แต่ละเกรดของแตงกวา
แล้วก็ย่อขนาด pixel ลง เสร็จแล้วเขาก็เอาไปเทรน
แล้วก็เอาไปเตรียมใช้งาน

เราจะทำการ classification แค่ 10 รุปนั้นก็คือ เรือ ม้า กบ หมา กวาง แมว นก รถบรรทุก รถ เครื่องบิน

แล้ว model ที่เราจะใช้ก็คือ artificial neural network ที่เรียกว่า multi layer perceptron

## Installing Tensorflow and Keras for Jupyter Notebook On Local Machine

จริงไม่อยากให้ลงที่ local เพราะมันยุ่งยากด้วย และก็ neural network มันซับซ้อน ถ้าบน colab จะมีสิ่งที่เรียกว่า tensor board ที่ชอบเราในการ monitor และเปรียบเทียบ model ของเรา

Anaconda ที่เราลงอะมันไม่ได้แถม Tensorflow กับ Keras เราต้องลงอีก

พิมพ์ในรูปแบบเดิม

```bash
conda install -c [channel_name] [package_name]
```

เริ่มการติดตั้ง
```bash
conda install -c conda-forge tensorflow
conda install -c conda-forge keras
```

## gather data

เราใช้ใช้ dataset ที่ื่อว่า CIFAR-10 คือ มีรุปที่ classify เป็น 10 type แล้วเรียบร้อย

```python
from keras.datasets import cifar10
(x_train_all, y_train_all), (x_test, y_test) = cifar10.load_data()  # from the documentation
```

## clean data & explore data

dataset เนี่ยอะเราไม่ได้ไปหามาจากที่ไม่แน่นอน คือแบบ dataset ของ lib ดังนั้นก็ทำการเช็คหาที่หายไปก็ไม่น่าพบ 

จากเดิมเรามี dataset ที่ถูกแบ่งเป็น 2 ชุดนั้นก็คือ train, test

|-----------------------data----------------------|

|-------------train-------------||------test------|

ตอนนี้เราจะทำการแบ่งส่วนที่เราเรียกว่า validation เพิ่มอีกส่วนจาก train dataset

|-----------------------data----------------------|

|------train-------||-validation-|------test------|

validation ใช้สำหรับทดสอบหา Metrics หลังจากเทรนเสร็จว่าโมเดลทำงานได้ดีแค่ไหน และหลังจากจูนแต่ละครั้งโมเดลไหนทำงานได้ดีกว่ากัน

test ก็หลังจากที่เราได้ model ที่ดีที่สุดมาแล้ว ว่าโมเดลจะทำงานได้ดีแค่ไหนกับข้อมูลที่ไม่เคยเห็นมาก่อน

แล้วขนาดที่เหมาะสมสำหรับ validation คือเท่าไหร

60% training, 20% validation, 20% test 

ถ้าใหญ่มากก็ 1% สำหรับแต่ละ validation, test

## Build Model

ทุกครั้งที่เราทำงานร่วมกับ tensorflow ในกรณีนี้เราใช้ keras ที่ใช้ tensorflow เป็นเบื้องหลังในการคำนวณ

มันก็จะมีขั้นตอน 3 ขั้นตอนมาเกี่ยวข้องในการสร้าง model

1.) Define Model
กำหนดโครงสร้างของ model

2.) Compile Model
เป็นขั้นตอนที่บอก tensorflow ว่าอยากวัดค่า loss ยังไง
จะปรับ weights ยังไง

3.) Trainning Model (Fit)
tensorflow จะเริ่มทำงานกับ data คำนวณแล้วววว

เราใช้ model ของ neural network ที่ชื่อ mulilayer percoptron

### 1.) define model

specify the number of layers, neuronsm, type of activation functions in side those neurons

เราจะตั้งค่าสถาปัตยกรรมของ model นี้
ที่เราจะกำหนดนั้นก็คือ layer ต่างๆ 

แต่ที่เราจะทำการกำหนดอันแรกไม่ใช่ input layer แต่เป็น first hidden layer ต่างหาก

keras ต้องการเรามาช่วยในการปรับแต่ง input layer เพียงนิ้ดเดียวเท่านั้น

ตอนนี้เราก็ต้องบอก keras ว่ามีกี่ input ใน first hidden layer

ถ้าเกิดเราทำงานเกี่ยวกับรูปภาพ จำนวนของ input จะขึ้นอยู่กับ resolution, color space

ในที่นี้ภาพของเราคือ 32*32 และ 3 channel

ทำให้ input ของรูปภาพเรา = 32 * 32 * 3 = 3072 นี่คือข้อมูลที่เราจะให้เมื่อเราทำการสร้าง first hidden layer ใส่ไปใน parameter input_dim

```python
from keras.models import Sequential
from keras.layers import Dense, Activation


Dense(units=[number of output that we want],  # the number of neuron in that layer
      input_dim=[input dim]),  # we need to specific the input dim because we made it the first layer
      activation='relu',  # how our neurons going to behave
```

ตอนนี้เรามี node ใน first hidden layer ทั้งหมด 128 อัน
ใน second hidden layer ทั้งหมด 64 อัน
thrid hidden layer 16 อัน
output layer 10 อัน

```python
model = Sequential([
    Dense(units=128, input_dim=TOTAL_INPUTS, activation='relu'),  # first hidden layer
    Dense(units=64, activation='relu'),  # second hidden layer
    Dense(units=16, activation='relu'),  # third hidden layer
    Dense(units=10, activation='softmax'),  # output layer
])
```

#### Activation Function

ยิงสัญญาณออกแรงแค่ไหน

สามารถดูเพิ่มเติมได้ที่ [doc](https://keras.io/api/layers/activations/)

การอ่านกราฟก็คือ input ของ layer ก่อนหน้านี้คือ แกน x แกน y คือ output ที่จะปล่อยสัญญาณต่อไป

[relu function picture](https://medium.com/@sonish.sivarajkumar/relu-most-popular-activation-function-for-deep-neural-networks-10160af37dda)

[soft max](http://krisbolton.com/a-quick-introduction-to-artificial-neural-networks-part-2)

ที่เราใช้ soft max ก็เพราะว่า มันจะเปลี่ยน output เป็น probability ทำให้ model เราสามารถบอกได้ว่า ภาพนี้มีโอกาศที่จะมีแมวอยู่ 87 เปอร์เซ็นอะไรประมาณนี้ 

soft max มี output ออกระหว่าง 0-1 และทั้งหมดจะรวมกันได้เพียง 1 เท่านั้น นี่คือสาเหตุที่ทำไมเรามักเห็น soft max เป็น activation function ของ output layer

### 2.) Compile Model

compile the model หมายความว่า การที่เราบอก tensorflow ว่าชนิดของการคำนวณหลังจากนี้คืออะไร

ทำไมเราต้องทำแบบนี้ เพราะว่า เบื้องหลังของ tensorflow คือการสร้าง graph

the graph is important because tensorflow needs to know how to organize its calculations

calculations >> การคำนวณ loss, how far away from true value, update the weight as the model beign train, track accuracy of the model during the training process

เราต้องทำการระบุ loss หรือ cost function ที่จะใช้

เช่นใน regression >> mse คือ cost dunction

และใน loss หรือ cost function ที่เหมาะสำหรับงานแบบที่เราจำก็คือ Categorical Cross Entropy

[graph](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html)

CE = -sum(y_real_i+log(y_predicted_i))  # i is number of category

การทำงานของ cost function นี้ ขอยกตัวอย่างเช่น
สมุติเรามีแมวเพื่อมาเช็คว่ารูปนี้มีแมวอยู่หรือไม่ >> 2 category มี (1) กับ ไม่มี (0)

แกน y คือ cost (infinity - 0)
แกน x คือ predicted probability (0 - 1)

CE = -(y_1 * log(y_predicted_1) + y_0 * log(y_predicted_0))
   = -(1 * log(1) + [0 * log(0)])  # ทำนายว่ามีแมว
   = -(1 * 0 + 0 * 1)
   = 0

มันมีหลายวิธีในการปรับค่า cost
วิธีในการปรับเรียงว่า [Optimizer](https://keras.io/api/optimizers/) >> algorithm that calculate the loss and adjust the weight

มันมีเยอะแยะเลย แล้วเราจะเลือกอันไหนดีหล่ะ เพราะแต่ละอันก็มีความแตกต่างกันเล็กน้อย

ถ้าอันที่นิยมสุดก็คือ Adam >> ประสิทธิภาพดี, ใช้เมมอรี่น้อย

เมื่อเราจะทำการ compile เราต้องกำหนดให้มันสามอย่าง optimizer, loss function, metric to calculate

```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

#### parameter

เราสามารถดู parameter ทั้งหมดได้โดย model.summary()
แล้วเราสามารถคำนวณไม่ได้ด้วยมือไหม ?

จำตอนแรกได้ป่ะ ที่เราคำนวณมันด้วยจำนวนของ neuron ในช่วงนั้นๆ ได้ 90 connection

แต่ 90 connection ไม่ได้หมายความว่ามันจะมี 90 parameter เพราะว่าในแต่ละ neuron มี bias ของมันอยู่

ถ้าพูกถึงแต่ละ neuron ก็หมายถึง activation function ด้วยที่บอกว่า neuron นั้นจะยิงสัญญาณแรงเท่าไหร

และเมื่อเราพูดถึง เกี่ยวกับ learning และ adjust weights สิ่งที่มันเกิดขึ้นจริงๆก็คือการเปลี่ยน activation function ทำให้กว้างขึ้นหรือแคบลงก็ได้หรือ shift กราฟไปทางซ้ายหรือทางขวาหรือทางไหนก็ได้ (Bias)

การเปลี่ยนรุปร่างของ activation function นั้นเอง

แน่นอนเมื่อมีการเปลี่ยนแปลงเกี่ยวกับ activation function นั้นก็หมายความว่า ความแรงของสัญญาณที่ส่งออกก็จะเปลี่ยนแปลงตามไปด้วย

เช่นใน jupyter notebook

parameter = [(32 * 32 * 3) * 128 + {128}] + [128 * 64 + {64}] + [64 * 16 + {16}] + [16 * 10 + {10}]

### 3.) Training Model

แล้วจะทำยังไงหล่ะ ก็ต้องเข้าไปดู doc เขา 

[keras api doc](https://keras.io/api/models/model/)

ดูที่ fit method

แล้วหลังจากเราทำการ fit ไปแล้วอะ เราจะรู้ได้ไงว่า model เราดีแค่ไหน ก็ต้องใช้ tensorboard 

ถ้าใน doc fit method จะมีชื่อ parameter หนึ่งที่เรียกว่า callbacks=None

```python
from keras.callbacks import TensorBoard  # ตัวเดียวกันแน่นออนนนน

model_1.fit(x_train_xs, y_train_xs, callbacks=[get_tensorboard('model_1')])
```

เมื่อเรา train model เสร็จจะดูผลลัพธ์ที่ tensorboard ทำยังไงหล่ะ ก็ ใช้ anaconda prom ไม่ชัวว่า cmd ได้ไหม

```bash
tensorboard --logdir=[path]
# http://localhost:6006/
```

ถ้าเราทำแค่นี่จะเห็นได้ว่า accuracy ของ model นั้นต่ำมากใน tensorboard ก็ต้องย้อนกลับไปดูที่ doc แล้วว่าเราพลาดอะไรไปหรือป่าว

จะเห็นได้ว่ามี parameter ที่ชื่อ batch_size, epoch

#### Epoch

epoch is when the entire dataset has been passed through the neural network a single time 

จากการสังเกตุ default ของ epoch นั้นมีค่า 1 มันก็เมคเซนต์ที่เห็น data point จุดเดียวบน tensorboard

การใส่ dataset ลงไปทีเดียวทั้งหมดเหมือนจะเป็นเรื่องที่ดี แต่ก็ไม่ใช่นะ

จำตอนที่เราเรียน gradient descent algorithm ได้ป่ะ ว่า optimization process นั้นคือการวนซ้ำเรื่อยๆ

นั้นก็คือ weight จะถูกอัพเดทเมื่อเรารัน fit method หนึ่งครั้ง

แสดงว่าที่เราทำเมื่อกี้ก็คือการโยน dataset ทั้งหมดเข้าไปทำงานเพียงรอบเดียวเท่านั้น

แล้วลองคิดดูถ้า dataset นั้นใหญ่มากๆๆๆๆๆๆๆๆ จะเกิดไรขึ้นถ้าเราโยนเข้าไปแบบตูมตามมม

ถ้าคอมแรกสามารถจัดการได้ก็ทำไป โยนเข้าไปรอบเดียวเพียวๆ ถ้าคอมไม่แรงก็ต้องแบ่ง dataset ออกเป็น set ย่อยๆที่เราเรียกว่า batch

#### batch_size

iteratino = number of training smaple / batch size

ก็ตามปกติ  batch size คือบอกว่าทำรอบละกี่ point

#### Let's do it again

```python
%%time
epoch_number = 20
batch_size = 1000
model_1.fit(x_train_xs, y_train_xs, batch_size=batch_size, epochs=epoch_number,
            callbacks=[get_tensorboard('model_1')])
```

ถ้าเกิด model ของเราไม่เกิดการเรียนรู้เลยแสดงว่าบางทีมันอาจจะมาจากจุดเริ่มต้นที่ผิดก็ได้ ก็ recompile ตอนเราสร้าง model ใหม่อีกรอบ แล้วค่อย train มัน

จาก tensboard จะรู้ได้เลยว่าเกิดการเรียนรู้ขึ้นแล้ววว ลองเพิ่มขนาด epoch ก็ได้ แล้วสั่งเกตุการเปลี่ยนแปลง

รองรันมันดูหลายรอบ จะเห็นได้ว่ามันไม่เหมือนกันเลยสักรอบ ทำไมหล่ะ ทั้งๆที่มาจาก dataset อันเดียวกันแท้ๆ

นั้นก็เพราะว่า optimizer ของเรามีการ random นิ้ดหน่อย อาจจะเกิดโชคดี หรือ โชคร้ายก็เป็นได้

```python
%%time
# with epoch & batch_size and validation
epoch_number = 1200
batch_size = 1000
model_1.fit(x_train_xs, y_train_xs, batch_size=batch_size, epochs=epoch_number,
            callbacks=[get_tensorboard('model_1')],  # which got higher accuracy
            verbose=0,  # not to display the output
            validation_data=(x_validation, y_yalidation))
```

ถ้าเราเพิ่ม calidation_data เข้าไป เราจะสั่งเกตุว่า tensorboard ของเรามี ช่องเพิ่มขึ้นมาสองช่องคือ cal_acc, val_loss

คุณไม่ต้องตกใจที่ validation accuracy มันต่ำซึ่งนั้นเป็นเรื่องปกติอยู่แล้ว

แต่ที่น่าตกใจนั้นก็คือค่า loss ของมัน คือตอนแรกมาก แลเว น้อย น้อย น้อย น้อย แล้วเริ่มมากขึ้นในช่วงท้าย นี่แหละที่เกิดปัญหา over fitting
