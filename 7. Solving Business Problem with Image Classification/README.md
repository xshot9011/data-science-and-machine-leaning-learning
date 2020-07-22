# Image Classification

เริ่มมาด้วยเื้อเรื่องของปัญหาเลย มีชายญี่ปุ่นชื่อมาโกโตะ เขาช่วยทำงานอยู่ที่ฟามแตงกว่าของพ่อแม่ สิ่งที่เขาก็รู้ก็คือพ่อแม่ของเราทำการจัดเรียงแตงกวาโดยคุณภาพของมัน
แล้วการจัดเรียงคุณภาพของแตงกวาเนี่ยแม่งเสียเวลามากถ้าใช้คนทำ เขาก็เลยอยากใช้ image classification ช่วยในการจัดกลุ่ม
ตอนแรกก็ถ่ายรูปแปะ label ให้แต่ละเกรดของแตงกวา
แล้วก็ย่อขนาด pixel ลง เสร็จแล้วเขาก็เอาไปเทรน
แล้วก็เอาไปเตรียมใช้งาน

เราจะทำการ classification แค่ 10 รุปนั้นก็คือ เรือ ม้า กบ หมา กวาง แมว นก รถบรรทุก รถ เครื่องบิน

แล้ว model ที่เราจะใช้ก็คือ artificial neural network ที่เรียกว่า multilayer perceptron

เนื้อหาก่อนหน้านี้เขาใช้ model Inception ResNet ซึ่งมีความแม่นยำมากกว่า model นี้มาก 55555

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
from keras.layers import Dense, Activation, Dropout


Dense(units=[number of output that we want],  # the number of neuron in that layer
      input_dim=[input dim]),  # we need to specific the input dim because we made it the first layer
      activation='relu',  # how our neurons going to behave
```

ตอนนี้เรามี node ใน first hidden layer ทั้งหมด 128 อัน
ใน second hidden layer ทั้งหมด 64 อัน
thrid hidden layer 16 อัน
output layer 10 อัน

```python
model_1 = Sequential([
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
tensorboard --logdir [path]
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

แต่ที่น่าตกใจนั้นก็คือค่า loss ของมัน คือตอนแรกมาก แลเว น้อย น้อย น้อย น้อย แล้วเริ่มมากขึ้นในช่วงท้าย นี่แหละที่เกิดปัญหา overfitting

#### Overfitting & Regularisation

##### Overfitting

ค่า validation value ถ้าดูจากกราฟ มันค่อยๆลดลงแล้วก็เพิ่มตอนท้าย 

เกิดขึ้นจาก overfitting = over(เกินไป) + fitting(พอดี)

เกิดขึ้นเมื่อ model เรียนรู้ data ได้ดีเกินไป ก็คือ model เรียนรู้ลักษณนิสัยทั้งหมดของ dataset นั้นได้ แบบทั้งลักษณะเล็กลักษณะใหญ๋ ถ้าเป็นคนก็คือสัมผัสมาทั้งภายนอกและภายใน

ทำให้ model เนี่ยมันเหมาะสมมากๆๆๆ สำหรับ training dataset นี้

model ของเราไม่ได้เรียนรู้แค่ความสัมพันธ์ที่อยู่ในข้อมูล แต่ยังเรียนรู้ noise ต่างๆที่อยู่ในข้อมูลนั้นๆด้วย

ดังนั้น model becomes unable to generalize well.

นั้นก็คือ model นี้ไม่สามารถที่จำใช้ในการทำนายข้อมูลนอก training dataset ได้

ตอนเราทำ regression คิดดูถ้าเส้นสมการที่เราสร้างอะลากเชื่อมทุกชุดได้ ตัวหนอนแน่นอน แลัวแน่นอนทำนายได้แม่นยำมาก แต่ก็ได้แค่ใน dataset นั้นที่เอามาใช้ train

ปัญหา overfitting เนี่ยมีอยู่ทุกที่ทั่ว machine learning technique ไม่ใช่แค่ neural netowrk 

แต่ใน neural networks มักจะมีปัญหานี้อยู่ด้วยเท่านั้นเอง

เพราะอะไร ?

ืneural network นั้นมีแนวโน้วที่จะมี parameter จำนวนมาก แค่ model เราตอนนี้มี 400000 กว่าแล้ว

จริงแล้วๆยิ่ง parameter เยอะเท่าไหร model ของเราก็จะยิ่งมีแนวโน้มที่จะมี (prone) overfitting มากเท่านั้น

เมื่อเรามี over แน่นอนเราก็ต้องมี under

Underfitting

ก็ตรงข้ามกันอันนั้นเหมาะสมเกินไป อันนี้ก็คือไม่เหมาะสม หรือนั้นก็คือการที่เราเทรนมันด้วย epochs=1 นั้นเอง

[graph](https://en.wikipedia.org/wiki/Overfitting)

ลองเข้าไปดูเส้นสีเขียวคือ overfitting การพยามแบ่งแยกข้อมูลมากเกินไป ทำให้แบ่งโดน noise ไปด้วย

###### Detect Over Fitting

เราก็สามารถดูได้จาก validation loss ใช่มะเมื่อมันเริ่มเพิ่มขึ้นแบบแปลกๆ นั้นก็คือมันเริ่มไม่เหมาะแก่การทำนายแล้ว

อาการที่เหมาะแก่การสังเกตุ คือมันลดลงเรื่อยๆ หรืออยู่เท่าเดิม หรือเพิ่มขึ้นเรื่อยๆ

แล้วเราจะแก้ไขปัญหา overfitting นี้ได้อย่างไร

วิธีที่ใช้ในการแก้ปัญหานี้ก็คือ Regularisation

##### Regularisation

มันคือเทคนิคที่ช่วยในการแก้ปัญหาด้านบน

###### Early Stop

ถ้าเกิดเราทำการ train model เราด้วย epoch ที่มากเกินไป เราก็ลดจำนวน epoch สิ ในเมื่อ validation accuaracy นั้นไม่เพิ่มขึ้นเท่าไหร และ validation loss นั้นก็ไม่ลดลงแล้ว

```python

```

###### Dropout

มีงานวิจัยพบว่า 

If you randomly ignore some of the nerons during the training, then you can reduce overfitting.

In other words during each training step some random neuron either in input layer of in the hidddn layer is not considered.

If we apply drop out technique to the input layer, we can specify a chance for every single one of these neurons to not be considered during the training.

If there's a 20 percent chance that each of these neurons can drop out, during the first training maybe the first neuron and all of its connections will be ignored.

If this neuron and all of its connections drop out then the network shrinks, it becomes a less complex network because there are fewer connections.

during the next training step a different neuron might not drop out with X% probability, the neuron that's drop out the first coming step will come back.

why this work ????

If some neuron is dropped out, it mean that all the connected downstream neurons in the first hidden layer don't want to rely to heavily on any single input.

If a random nueron drop out during the training step every time all of the connected nuerons will try to hedge(ป้องกันความเสี่ยง) themselves in order not to weigh any particular input too heavily.

This will help prevent Overfitting.

[drop out doc](https://keras.io/api/layers/regularization_layers/dropout/)

```python
model_2 = Sequential([
    Dropout(0.2, seed=42, input_shape=(TOTAL_INPUTS,)),  # dropout on input layer
    Dense(units=128, activation='relu', name='m2_fh'),  # first hidden layer
    Dense(units=64, activation='relu', name='m2_sh'),  # second hidden layer
    Dense(units=16, activation='relu', name='m2_th'),  # third hidden layer
    Dense(units=10, activation='softmax', name='m2_o'),  # output layer
])

model_2.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])
```

รอเราทำการ train ปุ๊ปเราจะสังเกตุได้ว่า อาการมันดีขึ้นนะจากกราฟที่แสดงออกมาเมื่อเทียบกับอันเดิม

ถ้าเราลองปรับปรุงมันมากกว่าเดิมโดยการผสมผสานเทคนิคหล่ะ

เอา dropout ไปแปะใน input, first hidden layer

```python
model_3 = Sequential([
    Dropout(0.2, seed=42, input_shape=(TOTAL_INPUTS,)),  # dropout on input layer
    Dense(units=128, activation='relu', name='m3_fh'),  # first hidden layer
    Dropout(0.25, seed=42),  # dropout on first hidden layer
    Dense(units=64, activation='relu', name='m3_sh'),  # second hidden layer
    Dense(units=16, activation='relu', name='m3_th'),  # third hidden layer
    Dense(units=10, activation='softmax', name='m3_o'),  # output layer
])

model_3.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])
```

###### Look More Closly

เปิด tensorboard ไปที่ๆผมเก็บ log ไว้ให้เลย มี xs คือ train ด้วย dataset ที่มีขนาดเล็ก

ตอนนี้เราก็จะมี model ทั้งหมด 3 แบบด้วยกัน

1. แบบที่ไม่มี dropout เลย
2. แบบที่มี dropout ที่ input layer
3. แบบที่มี dropout ที่ input, first hidden layer

จากการสังเกตุทั้ง 3 model กับ ขนาดของ dataset ที่มีขนาดเล็กและขนาดใหญ๋

ถ้าดูที่ค่า accuracy ของ dataset ที่มีขนาดใหญ่นั้นก็มีอัตราการเปลี่ยนแปลงไปในทางที่ดีมากกว่า dataset ที่มีขนาดเล็ก ถึงแม้ว่าจะมี dataset ขนาดเล็กที่แซงขึ้นมาได้อันเดียวแต่นั้นก็อาจจะเป็นเพราะการสุ่มในการปรับน้ำหนัก

ส่วนค่า loss ของทั้งสองแบบลดลงแบบปกติ ไม่ค่อยมีอะไรน่าสนใจในกรณีนี้

ในช่วงแรง validation accuracy ของ dataset ขนาดใหญ่นั้นมีความแม่นยำที่เพิ่มขึ้นอย่างรวดเร็วมากในช่วงแรก
กลับกันใน dataset ขนากเล็ดนั้นมีความแม่นยำเพิ่มขึ้นแบบช้ามากๆในช่วงแรก 

แต่ค่า validation accuracy ของ model ทั้งสาม ที่มี dataset ขนาดเท่ากันนั้นค่าก็ไม่เกาะกลุ่มกันมากไม่ต่างกันเท่าไหร

ลองดูค่า max ของ validation accuracy ดูจะสังเกตุได้ว่า dataset ขนาดใหญ่นั้นมีความแม่นยำสูงกว่า dataset ขนาดเล็กมาก

ถึงแม้ว่าช่วงท้ายของ epoch นั้นจะมีอัตราการเปลี่ยนแปลงที่ต่ำ แต่การไปถึงค่า max ที่ควรเป็นไปได้นั้น dataset ขนาดใหญ่ใช้จำนวนครั้งน้อยกว่ามาก

ถ้าในเรื่องของเวลาการเทรน โมเดล 2, 3 นานกว่า 1 อยู่แล้วเนื่องจากมีการใช้ dropout มาช่วย

สรุป:::::::

ปริมาณ data ทำให้เกิดความแตกต่างที่เยอะมาก ทำให้ลด overfitting, ทำให้เพิ่ม accuracy

early stop ก็ดีเมื่อใช้หยุดไม่ต้องเสียเวลานานเกินควร

เราใช้ dropout ทำให้โอกาสเกิด overfitting นั้นน้อยลง แต่เิ่มเวลาในการเทรนนิ้ดหน่อย

การเพิ่มปริมาณ data นั้นมีผลมากกว่าการทำ dropout อีก

แต่ model ตอนแรกที่เราทำมันยังมี accuracy ต่ำอยู่นะเดี่ยวค่อยไปปรับกันทีหลัง

## Evaluate

จำ recall กับ precision กับ f-score ได้ไหม

และก็ที่มีเพิ่มขึ้นมาก็มี accuracy, loss

และที่เพิ่มขึ้นมาก็คือ Confusion Matrix

ไอเดียของ confusion matrix นั้นเรียบง่าย จริงๆมันคือตาราง crosstabs ขนาด n×n ทั่วไป โดยแกนนอนคือ actual result ส่วนแกนตั้งคือ prediction result

ถ้าค่า actual == predicted นั้นก็คือค่าบนเส้นทะแยงมุมนั้นเอง 

แสดงว่าถ้าอยากรู้ accuracy วัดจากเส้นทแยงมุมได้

หรือนั้นก็คือข้อมูลที่อยู่ในเส้นทแยงมุมคือ true positive

ถ้ามองตาม column ต่างๆ เช่น เครื่องบินหล่ะ ไม่เอาเส้นทแยงมุมนะ ทำนายว่ามันคือเครื่องบิน แต่จริงๆไม่ใช่ นั้นก็คือ false positive
sum column > flase positive,

ถ้ามองตาม row ต่างๆ เช่น เครื่องบินเหมือนเดิมไม่มองเส้นทะแยงมุมเหมือนเดิม ก็คือ จริงๆแล้วมันเป็นเครื่องบินแต่ทำนายผิด นั้นก็คือ false negative
sum row > false negative

ลองมองใน matrix ดูก็ได้ว่า model เราสับสนในเรื่องไหนมากที่สุด

jupyter notebook ได้คำนวณเรื่องพวกนี้หมดแล้ว