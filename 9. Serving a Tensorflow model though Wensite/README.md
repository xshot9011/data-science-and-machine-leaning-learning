# What's coming up

เราสามารถเอา model ของเราออกไปทำงานกับโลกภายนอกได้ เพื่อให้เจอ real life data

turn machine learning model to actual product

## step

### 1. Saving & Loading Models

เพราะว่าจากการที่เราทำว่าอะ trianing ใช้เวลานานแล้วจะเทรนใหม่ทุกครั้งที่มันถูกเรียกใช้งานก็แปลกๆ 

### 2. Deploy Model to Browser

เราไม่สามารถรัน tensorflow บน browser ได้ตรงๆนะมันจะมีทริคเล็กน้อย

นั้นก็คือการใช้ tensorflow.js นั้นก็คือการ convert model เราไปเป็น tensorflow.js เพือที่จะให้เราสามารถเรียกใช้งาน model ของเราได้บน web browser

### 3. Build out Website

เขียนเว็ปไซต์ขึ้นมาเพื่อทำงานเกี่ยวกับมัน ทั้งหน้าบ้านหรือหลังบ้าน

### 4. Pre-Process Data

เช่นการที่เราได้ data มาอาจจะเป็นรูป 28*28 หรืออะไรก็ตามเราก็ต้องทำการ faltten data ที่เราได้มาด้วย ดูสีดุอะไรว่าตรงกับที่เราต้องการไหม อาจจะเป็นพวกสเกลรูปก็เป็นได้ มีมากมายในส่วนนี้

ในตอนนี้เป็นเวลาที่ดีที่เราจะได้เรียนเกี่ยวกับ lib OpenCV ของ intel

### 5. Predict Input

ลองใช้งานจริง

### 5. Public Website

## 1. Saving & Loading Models

### Saving Model

ช่วยลดเวลาในการทำงานได้อย่างมาก เช่นในการทำงานเราต้องเทรนก่อน 20 hr แล้วทำนาย ต่อมาเราก็ต้องทำแบบนั้นอีก แต่ถ้าเรา save & load มาเวลาในส่วนนั้นก็จะประหยัดไป

ทั้งในแง่ทรัพยากรต่างๆที่ไม่ใช่ เวลา นะ

ตอนนี้การ save model ของเรามีทั้งหมด 2 แบบด้วยกัน

checkpoint, savedmodel

แล้วมันต่างกันยังไง 

checkpoint > save variables
savedmodel > save variables, grapg & metadata (weight, name, ...)

แล้วเรื่องการใช้งานหล่ะ 

checkpoint > during training 
    ใช้ในระหว่างการ train เช่นอยากจะ save ทุกๆ 10000 step เหมือนทำ snapshot ไว้ที่จุดต่างๆ
savemodel > after training
    ใช้ในการ serve model
    [doc](hhttps://www.tensorflow.org/versions/r1.15/api_docs/python/tf/saved_model/simple_save)

```python
tf.saved_model.simple_save(
    session,  # session 
    export_dir,  # the dir to keep all model files
    inputs,
    outputs,
    legacy_init_op=None
)
```

แล้วอะไรคือ inputs, outputs หล่ะ ถ้าไปดูที่ tensorboard graph จะสังเกตุว่าเราได้ทำการแยกไว้ให้แล้ว

input ของเราก็คือ X ที่เก็บค่า features ไว้ทั้งหมด แล้วก็วิ่งไปที่ layer1 ไปต่อที่ dropout ไปต่อที่ layer2 ตามกราฟ

แล้วไอตัว input เนี่ยอยู่ส่วนไหนใน jupyter notebook 

อยู่ตรงส่วนของที่เราประกาศสร้าง placeholder ที่ชื่อตัวแปร X นั้นเองง

แล้ว output ของเราที่เป็น prediction หล่ะ ถ้าลองเข้าไปหาใน tensorboard ก็จะพบว่ามี name scope หนึ่งที่ชื่อว่า out อันนั้นใช่ output จริงๆที่เราควร save หรือป่าว

คำตอบคือ เกือบใช่แต่ก็ไม่ 

เพราะว่า ความจริงที่ว่าอันนี้คือ layer สุดท้ายของ model นี้นั้นก็จริงแต่ว่า prediction ของเราอะคือค่าที่ max ที่สุดของ output นั้นไม่ใช่หรอ

output จาก layer นั้นคือการที่เราเอาค่าผ่าน softmax activation function ซึ่งให้ prob ของ class ต่างๆมาอีกที

แต่เราต้องการค่าที่มากที่สุดในนั้น เราควรไปเอามันจากไหนดีหล่ะ

มี name scope หนึ่งที่เรียกว่า accuracy_calc เอาไว้คำนวณความถูกต้อง

```python
with tf.name_scope('accuracy_calc'):
    correct_pred = tf.equal(tf.argmax(output, axis=1), tf.argmax(Y, axis=1))  # if true value, equal to predicted value
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
```

เราก็ทำการปรับแต่งเพื่อหาค่า prob ที่มากที่สุดจาก softmax โดยการ

```python
with tf.name_scope('accuracy_calc'):
    model_prediction = tf.argmax(output, axis=1, name='prediction')
    correct_pred = tf.equal(model_prediction, tf.argmax(Y, axis=1))  # if true value, equal to predicted value
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
```

พอเราแก้มันปุ้ป แล้วลอง train model เราใหม่เราก็จะพบว่า ในส่วน name_scope accuracy_calc นี้จะมีก้อนทีชื่อว่า prediction ที่เป็น output จริงๆของการทำนายออกมา

นี่แหละคือที่เราะเอาไปใส่ใน output นั้นตอนเรา savedmodel

ฉะนั้นก็ทำการ copy มาเลย >> accuracy_calc/prediction บนมุมขวาของกราฟ

```python
output = {'accuracy_calc/prediction': model_prediction}
inputs = {'X': X}
tf.compat.v1.saved_model.simple_save(session, 'SavedModel', inputs, output)
```

อย่าลืมการที่เราจะ save model ได้ session นั้นต้องทำงานอยู่ก็รันใหม่ไป

จะมี folder ใหม่ปรากฏขึ้นชื่อว่า 'SavedModel'

### Loading Model

ขั้นตอนส่วนมากถูกเขียนไว้ใน jupyter 

[doc](https://www.tensorflow.org/api_docs/python/tf/compat/v1/saved_model/load)

## 2. Deploy Model to Browser

ตอนนี้เราจะทำการแปลง tensorflow model ของเราไปเป็น tensorflow.js format
[git](https://github.com/tensorflow/tfjs-converter)

ตอนนี้เราเล่นกับ module หลายๆส่วนและหลายๆ version ทำให้การทำงานเราอ้องได้ในบางครั้ง

ดังนั้นเราต้องทำการสร้าง environment

```bash
conda create --name [env_name] python=[versino].., ==[version...]
```

ถ้าเราทำ window ให้เราเปลี่ยน maximun name path ไปเป็น 1 ใน registry

```bash
conda actiavte [env_name]
pip install tensorflowjs==1.2.3
```

เริ่มการแปลงได้

```bash
tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model "[savedmodel path]" "where the output should store"
```

ไปสร้าง folder ใหม่เพื่อรอรับ output ที่ออกมาจากการรันคำสั่งด้านบนได้เลย

พอรันคำสั่งเสร็จจะได้ไฟล์มา 2 ไฟล์ model.json กับไฟล์ weights 
อีกไฟล์หนึ่งค่อนข้างจะอ่านยากเนื่องจากเป็น binary ไฟล์

## coming