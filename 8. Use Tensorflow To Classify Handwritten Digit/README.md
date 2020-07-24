# What's coming up?

เราจะมาจัดการเกี่ยวกับลายมือของตัวเลขกัน

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