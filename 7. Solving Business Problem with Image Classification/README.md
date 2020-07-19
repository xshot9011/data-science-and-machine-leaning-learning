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

มี first hidden layer 6 node
 second hidden layer 5 node
 output layer 4 node


### 1.) define model

เราจะตั้งค่าสถาปัตยกรรมของ model นี้
ที่เราจะกำหนดนั้นก็คือ layer ต่างๆ 

แต่ที่เราจำทการกำหนดอันแรกไม่ใช่ input layer แต่เป็น first hidden layer ต่างหาก

keras ต้องการเรามาช่วยในการปรับแต่ง input layer เพียงนิ้ดเดียวเท่านั้น

ตอนนี้เราก็ต้องบอก keras ว่ามีกี่ input ใน first hidden layer

ถ้าเกิดเราทำงานเกี่ยวกับรูปภาพ จำนวนของ input จะขึ้นอยู่กับ resolution, color space

ในที่นี้ภาพของเราคือ 32*32 และ 3 channel

ทำให้ input ของรูปภาพเรา = 32*32*3 = 3072 นี่คือข้อมูลที่เราจะให้เมื่อเราทำการสร้าง first hidden layer ใส่ไปใน parameter input_dim

```python
from keras.models import Sequential
from keras.layers import Dense, Activation


Dense(units=[number of output that we want],  # the number of neuron in that layer
      input_dim=[input dim]),  # we need to specific the input dim because we made it the first layer
      activation='relu',  # how our neurons going to behave
```

ตอนนี้เรามี node ใน first hidden layer ทั้งหมด 6 อัน

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