# What's coming up?

เราจะมาจัดการเกี่ยวกับลายมือของตัวเลขกัน

jupyter notebook เรามีจะ 2 version นั้นก็คือ version ที่เป็นปกติมีคอมเมนต์ กับไม่มีคอมเมนต์ตั้งแต่เนื้อหา name scope เนื่องจากมีการจัดกลุ่ม code ไหม เลยขออนุญาติล้าง code ที่อยู่ใน jupyter นั้นๆ มาเริ่มไว้ที่อันไหม

# Tensor

ผู้คนให้คำนิยาม Tensor ต่างกันไป มันคืออะไรมาดูกัน

ตัวเลขปกติ 9 เราก็เรียกมัน scalar

ถ้ามีตัวเลขอยุ่ชุดหนึ่ง 0 8 16 11

เราอาจจะใช้คำนิยามมันว่า vector หรือ array

1 2 3 4
5 6 7 8
2 3 1 0

เราก็จะเริ่มเจอกับตัวเลขที่มากขึ้นก็นิยามมันว่า DataFrame หรือ Matrix

1 2 3 4 | 255 255 0 | 127   0 0
5 6 7 8 | 255   0 0 |   0 255 0
2 3 1 0 |   0   0 0 |   0   0 0

หรือมาเจอรูปภาพที่มี dimension ที่ 3

จริงๆจำนวนของ dimension นั้นสามารถเรียกได้ว่า rank

scalar => rank = 0
vector => rank = 1
matrix => rank = 2
stack matrix => rank = 3

เมื่อเราทำงานกับ TensorFlow ก็เหมือนเราทำงานกับ Data Structure ที่มี dimension เท่าไหรก็ได้

So a tensor is basically a container with N dimension

เช่น stack matrix ของ RGB นั้นก็อาจจะมีชื่อว่า 3 dimension tensor

tensor อาจจะกล่าวถึงตั้งแต่ 2 dimensionขึ้นไป ไปถึง N dimension เพราะว่าอันอื่นเราก็มีคำเรียกแทนอยู่แล้ว

ดังนั้นคุณมักจะเห็นผู้คนกล่าวถึงคำว่า tensor เมื่อเรียกสิ่งที่ของที่ 3 dimension ขึ้นไป

tensor เนี่ยมันเป็นมากกว่าdata structure นั้นก็คือ tensor นั้นเป็นไปตามกฏคณิตศาสตร์ สามารถคูณ บวก ได้ >> linear algebra

แล้ว linear algebra เนี่ยจะมาช่วยเราในการคำนวณ input กับ output ของ neuron ของเรา

เช่น เราจำทำการ คูณ output ด้วย connection weight เพื่อหาว่าควรเอาค่าอะไรไปเป็นค่าใส่ใน input layer ถัดไป

เช่น

input layer |||||  first hidden layer
 out1=>2.2                in1
 out2=>7.1                in2
 out3=>8.9                in3
 out4=>2.2                in4
 out5=>6.0

ถ้ามีค่า out1-5 ที่แน่นอนแล้ว เราจะคำนวณค่า input สำหรับ in1-4 ได้อย่างไร? เนื่องจากมีน้ำหนักเข้ามาเกี่ยวข้อง

จำได้ป่ะ out1-5 อะจะเข้ามาเป็น input หรือแกน x ใน acitvation function ของเราต่อ แต่มันมีขั้นตอนก่อนหน้านี้อยู่

นั้นก็คือ

w11 w12 w13 w14 w15     out1     in1
w21 w22 w23 w24 w25     out2     in2
w31 w32 w33 w34 w35  *  out3  =  in3
w41 w42 w43 w44 w45     out4     in4
w51 w52 w53 w54 w55     out5     in5

# TF setup & Architecture

ตอนนี้เราจะมาสร้าง tensor เหมือนกับเรา setting layer ใน nueral network

tensorflow เนี่ยต้องการให้เราสร้าง placholder เอาไว้ให้ tensorflow รู้ set up, how data should flow ด้วยกราฟของมัน

[doc](https://www.tensorflow.org/api_docs/python/tf/compat/v1/placeholder)

# Tensorboard Summaries and the Filewriter

วิธีการที่ tensorboard จะเก็บผลลัพท์การคำนวณทั้งหมด เรียกว่า Summaries

: The way that tensorboard gets hold of theses calculations is through something called a summary

Summary เนี่ยทำให้ tensorboard สามารถสร้าง chart สวยๆได้เหมือนกับที่เราได้เห็นใน keras callbacks และการจะสร้างไฟล์แบบนั้นต้องอาศัย FileWriter

# Tensorflow Graph

จะมาพูดถึง Tensorflow Graph แบบลงลึกนิ้ดหน่อย เพราะจากการที่เราไปเขียนโค็ดมาก็ งง พอสมควร

ก่อนจะมาอ่าน markdown ส่วนนี้ให้ไปอ่านโค๊ดใน jupyter ตั้งแต่ set up tensorflow นะจนถึง rerun 

แล้ว markdown ส่วนนี้จะเป็นการ review โค๊ดในส่วนนั้นๆแทน

กลับไปอ่านแล้วก็กลับมาที่นี้

ไปที่ tensorboard ของเรา ลองเปลี่ยนหัวข้อจาก scalar ไปเป็น graph ได้เลย คุณจะเห็นได้ว่า มันมีกราฟอะไรไม่รู้เต็มไปหมดเลย

ย้อนกลับมาที่ตอนแรกเริ่มเราจะแบ่งการทำงานของมันได้เป็น 2 step
นั้นก็คือขั้นตอนการ defininf all the calculation and variables แล้วขั้นตอนการ running and evaluate calculation

สาเหตุที่เราต้องมี step แรกนั้นก็คือ tensorflow จะเอาโค็ดเรามาแล้วก็ compile the graph

และเมื่อเราไปดูที่ graph

node - ก้อนกลม - mathemetical operation
edge - เส้น - data flow บอก size data ด้วย

ลองสังเกตุพวก relu, loss function ของเราดูที่เราทำการกำหนดจะอยู่ใน graph หมดเลย

การคำณวณต่างๆลองไล่จากล่างขึ้นบนจะเห็นได้เลยว่า ใช่เลยยยย

Variable - something that maintain the state of the graph

Variable เนี่ยสามารถถูกเปลี่ยนแปลงค่าได้ เช่นพวก weights, biases ที่เราทำการเปลี่ยนแปลงระหว่างการเทรนโมเดล

```python
initial_w_1 = tf.truncated_normal(shape=[TOTAL_INPUTS, hidden_1_number], stddev=0.1, seed=42)  # the number of input, number of nueron
w1 = tf.Variable(initial_value=initial_w_1)  # create tensorflow's variable to hold all the weight in the first hidden layers
```

นี้เป็นตอนแรกเริ่มที่เราสร้าง tensorflow variable

เมื่อเรามี Variable ที่เปลี่ยนแปลงได้ก็ต้องมีสิ่งที่เปลี่ยนแปลงไม่ได้เรียกว่า constant แต่ในนี้เราสร้างมันขึ้นมาเพื่อเป็น initial เฉยๆ

ด้านบนที่กล่าวมาคร่าวๆก็คือการ set up อะ แตถ้าอยากเริ่มการคำนวณหล่ะ

We wanted to launch this graph

```python
session = tf.Session()
# initialise all the variables
init = tf.global_variables_initializer() ##2##
session.run(init)  # the initializer is evaluated
```

##2## this line of code that evaluated all the initialzation operations, we had above

สาเหตุที่ต้องมีบรรทัดด้านบนก็เพราะว่า ถ้าบรรทัดนี้ไม่ถูกรันอะ บรรทัดอื่นก่อนหน้าอะไม่ได้เก็บค่าอะไรไว้เลย

tensorflow variable only get their values after the initializer is evaluated.

go to this tag in jupyter "to See the values inside the var that we created"

หลังจาก initializer ถูก evaluated แล้วเราก็สามารถดูค่าของตัวแปรได้

```python
b_2.eval(session)
```

แล้ว session หล่ะมันคือสิ่งใด

Tensorflow session is when our placeholder can start getting their values

The thing about placeholder is that place holders must be fed and they must be fed during the session.

นี่คือสาเหตุที่เราต้องใช้ session.run([......])

Brief: Placeholder are hungry and you have to feed them. Placeholder will do you work for food. We feed them with a feed_dict.

feed_dict เนี่ยไปดูที่สิ่งเรากำหนดค่ามัน

```python
feed_dictionary = {X:batch_x, Y:batch_y}
```

ก็คือมัน map ค่า placeholder: X ไปที่ data จริงๆที่เรามี data กับ placeholder ต้องมี shape ที่เข้ากัน

จำได้ป่ะเรา map มันในโค็ดสองรอบ แต่ขนาดของ row อะแตกต่างกันทำไม placeholder ไม่อ้อง

นั้นก็เพราะว่าตอนเรากำหนด shape ให้ placeholder เราใส่ None, TOTAL_INPUT ไง

การที่เรา set None ไว้เนี่ยแหละทำให้เราสามารถ feed มันด้วยจำนวนของ sample ที่หลากหลายได้

และเมื่อมีการเปิดตัวแปรอื่นๆไว้ด้านบนก็ต้องมีการปิดเพื่อคืนทรัพยากรต่างๆให้ระบบ .close()

ลองอ่านไปตามกราฟก็ได้ก็จะพอเข้าใจโครงสร้างของมันว่ามันทำงานยังไง

แล้วถ้าอยากจัดกลุ่มพวกนี้หล่ะ กราฟนี้มันดูรกๆยังไงไม่รุ้พ่ามมม ก็ต้องพูดถึงเรื่อง Name scope

## Name Scope

We can start grouping different types of operations together in out tensorflow grapg

We are going to work with something called "Context Manager"

ย้อนกลับไปดูใน graph มันจะมีส่วนของ first hidden layer อยู่ ถ้าเราสามารถรวมมันได้มันก็จะดูเป็นหมวดหมู่ขึ้นได้

ใน tensorflow ก็มี function ที่ชื่อ tf.name_scope แล้ว function เนี่ยก็ไปใช้ context manager อีกที

ดังนั้น all calcalations ใน first hidden layer นั้นก็ต้องอยู่ใน 1 context และทำนองเดียวกันก็แบ่งได้ตาม syntax เบื้องต้น

```python
with tf.name_scope('output_layer'):
    # default set up for output layer
    initial_w_output = tf.truncated_normal(shape=[hidden_2_number, NUMBER_CLASSES], stddev=0.1, seed=42)
    w_output = tf.Variable(initial_value=initial_w_output, name='w_output')
    initial_bias_output = tf.constant(value=0.0, shape=[NUMBER_CLASSES])
    b_output = tf.Variable(initial_value=initial_bias_output, name='b_output')

    o_input = tf.matmul(layer_2_output, w_output) + b_output
    output = tf.nn.softmax(o_input)
```

หลักจากเราทำ nmae_scope ให้ทุกๆส่วนของ code แล้วก็ลองเข้าไปดูที่ graph อีกรอบจะเห็นการจัดหมวดหมู่ที่ชัดเจนมากขึ้นกว่าเดิม

ลองอ่านกราฟดูอีกรอบจะเข้าใจการทำงานของ tensorflow มากขึ้น

# Hyperparameters & Dropout and Histogram summary

เนื้อหานี้จะอยู่ในส่วน jupyter notebook ที่ maintain แล้ว

ลองอ่านเทียบไปกับการเปิดกราฟ tensorboard

หัวข้อทั้งหมดด้านบนเนี่ยจะทำให้เราสามารถ train model ด้วย epoch ที่เหลากหลายได้ เพื่อดูการเปลี่ยนแปลง accuracy

```python
def setup_layer(input_before, weight_dim, bias_dim, name):  # with histogram summary
    with tf.name_scope(name):
        initial_w = tf.truncated_normal(shape=weight_dim, stddev=0.1, seed=42)
        w = tf.Variable(initial_value=initial_w, name='w')
        
        initial_bias = tf.constant(value=0.0, shape=bias_dim)
        b = tf.Variable(initial_value=initial_bias, name='b')

        layer_input = tf.matmul(input_before, w) + b
        if name == 'out':
            layer_output = tf.nn.softmax(layer_input)
        else:
            layer_output = tf.nn.relu(layer_input)
        
        ########################################################
        tf.summary.histogram('weights', w)
        tf.summary.histogram('biases', b)
        ########################################################

        return layer_output
```

แล้วลองทำการเทรนโมเดลเราใหม่ดูจะเห็นได้ว่ามี column เพิ่มขึ้นมาด้วยยว้าวๆ 

สองบรรทัดที่เราทำการ add เข้าไปตอนสร้าง layer อะ

## histogram/ layer1/ biases

ลองไปดูกราฟนี้ตาม path ด้านบนใน tensorboard สวยมากเลยนะ แต่มันหมายถึงอะไร ??

กราฟนี้แสดงถึง distribution of biases for the first hidden layer over time

ตอนเรากำหนดอะ bias มีค่าเป็น 0 ใช่ป่ะแสดงว่าค่าที่อยู่บนกราฟนี้คือค่าที่รันเสร็จในแต่ละ epoch แล้ว (y)

## histogram/ layer1/ weights

เราเริ่มด้วย truncated distribution

สิ่งที่น่าสนใจก็คือตอนท้ายมันก็ยังคงเหมือนกับ truncate distribution อยู่

สิ่งที่บอกได้เมื่อ biases มีการเปลี่ยนแปลงแต่ weight นั้นไม่มีการเปลี่ยนแปลงมีได้เพียงอย่างเดียวนั้นก็คือ การที่การเรียนรู้ของเราเกิดขึ้นที่ biases อย่างเดียว

## histogram/ layer2/ biases, weights

สิ่งที่น่าสนใจก็คือการที่ biases เปลี่ยนแปลงแบบรุนแรงมากๆ
และเหมือนเดิมใน weight นั้นแทบไม่มีการเปลี่ยนแปลงเลย

### histogram/ output/ biases, weights

ถ้าลองสังเกตุ biases ดูก็จะพบว่ามันมีทังหมด 10 ค่า (10 ภูเขา)

## distribution

เป็นการดูการกระจายของข้อมูล ที่สีเข้มๆนั้นก็คือที่ median อยู่ ที่ขอบด้านบนนั้นก็คือค่า max ขอบข้างล่านั้นคือคือค่า min ตามปกติ

distribution เนี่ยค่อยข้างออกไปเชิงสถิติ แต่ละเส้นบนกราฟบ่งชี้ถึง เปอร์เซ็นไท ต่างๆ

## ลองทดลองเปลี่ยน hyperparameter

เริ่มแรกเราตั้ง learning rate เอาไว้ที่ 0.0001 ก็จะมี accuracy แบบหนึ่ง

แต่พอเราได้ทำการเปลี่ยน learning rate ไปเป็น 0.001 จะเห็นได้อย่างจัดเจนเลยว่า accuracy เราเพิ่มขึ้นไวมากในตอนแรก และตอนท้ายก็มีค่ามากกว่า learning rate ที่ 0.0001

แต่ลองไปดูตอนท้าย validation_accuracy นั้นดูเหมือนว่าจะไม่ลดลงอีกและมีท่าทีกระตุกเพิ่มขึ้นด้วย

ทำให้รู้ว่าปัญหาตอนนี้อาจจะเกิด overfitting เราอาจจะต้องทำ regulariztion หรือ drop out ก็เป็นได้

ลองไปดูที่ weight เมื่อเทียบกันทั้งสองแบบ แบบที่ learning rate สูงกว่าจะมีการเกาะกลุ่มกันมากกว่าและค่าสูงมากเมื่อเทียบกับแบบเก่า

ไปดูที่ output layer ต่อจะเห็นว่า leanring rate ที่ต่ำนั้นการเปลี่ยนแปลงส่วนมากมักจะเกิดที่ output layer ไม่ใช่เกิดที่ first, second hidden layer มาดูอันที่ learning rate สูงถ้าเรามองจาก overlay มุมมอง เราจะเห็นได้ว่า การเปลี่ยนแปลงก็มีแต่ไม่ dramatically แบบอันครั้งเก่า

ย้อนกลับมาดูที่ biases ทั้งสองกรณีเลย bias มีการเปลี่ยนแปลงอย่างรุนแรงมาก แต่ก็เป็นตัวชี้วัดได้ดีว่า model ของเราเกิดการเรียนรู้ขึ้น

## Overfitting Problem

ถึงแม้ตอนนี้มันอาจจะไม่เกิดแต่ในอนาคตเราต้องได้พบเจอแน่นอน

```python
# model dropout
layer_1 = setup_layer(X, weight_dim=[TOTAL_INPUTS, hidden_1_number],
                     bias_dim=[hidden_1_number],
                     name='layer_1')
layer_dropout = tf.nn.dropout(layer_1, keep_prob=0.8, name='dropout_layer_1')  ######
# then change the input layer to layer 2 is dropout layer instead
layer_2 = setup_layer(layer_dropout, weight_dim=[hidden_1_number, hidden_2_number],
                     bias_dim=[hidden_2_number],
                     name='layer_2')
output = setup_layer(layer_2, weight_dim=[hidden_2_number, NUMBER_CLASSES],
                     bias_dim=[NUMBER_CLASSES],
                     name='output')
model_name = f'{hidden_1_number}-{hidden_2_number} LR;{learning_rate} E;{epoch_number}'
```

ก็การเพิ่ม dropout เข้าไปก็จะส่งผลคล้ายๆกับที่เราทำใน keras เช่นการเทรนช้าลงบ้าง ช่วยลด overfitting ได้