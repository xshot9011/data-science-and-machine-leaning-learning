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