# Neural Networks

## Introducing & How brain learning

artificial neural networks were inspired by or desire to understand the human brain thinking

ถ้าเกิดคอมมีความสามารถเยอะขึ้นเรื่อยๆคุณก็จะมีคำถามว่าทำไมงานง่ายๆมันทำได้ไม่ดีเท่าไหร เช่นการให้บอกความแตกต่างของหมากับแมว

ตอนนี้เรามีทรานซิสเตอร์ ครอคสปีดที่เยอะ การประมวลผลที่เร็วขึ้นเรื่อยๆ ไวกว่าสมองมนุษย์อีก

แล้วสมองมนุษย์ของเรามีการเรียนรู้ได้อย่างไร ก็จะมีพวกเซลประสาทในสมองที่เชื่อมต่อกัน ที่มีอินพุทคือขาเดรนไดร์ทและเอาท์พุทคือขาแอคซอน

ถ้าขาที่รับเข้ามาจะส่งต่อหรือไม่ก็ขึ้นอยู่กับ threshould แล้วการเรียนรู้เกิดขึ้นได้อย่างไร

มีการเขียนในงานวิจัยว่า

Neural pathways are strengthened every time that they are used.
If 2 neurons fire ay the same time the connection between them is enhanced

ยกตัวอย่างการเรียนรู้ในภาษาอื่นๆ เซลประสาทก็จะส่งสัญญานทุกๆครั้งที่เราทำการพูดทำให้เราสามารถจำได้ หรือนั้นก็คือการ activating nueron นั้นซ้ำๆเรื่อยๆ

artificial neural network based on สิ่งนี้

## Layers, Feature Generation and Learning

### Why are Neural Networks so Special

จำได้ไหวตอนที่เราทำ multiple regression เพื่อทำนายราคาบ้าน เราก็มานั่งเรื่องว่าจะเอา feature ไหนบ้างที่เข้าไปประมวลผลใน model ของเรา

ตอนที่เราทำ filter spam เหมือนกัน เราใช้ความน่าจะเป็นของแต่ละคำมาคูณๆกันใช่ป่ะเพื่อแยกว่าอีเมลนั้นเป็นสแปมหรือไม่เป็นสแปม

สองวิธีที่เราเคยเรียนมาอะ มาเรียกว่า shallow learning algorithm

เราในฐานะนัก data sci ก็จะมาเลือกว่า feature ไหนเหมาะสม(make sense) และ ช่วยให้การทำนายนั้นแม่นยำขึ้นได้

ทั้งหมดที่ algorithm จะทำก็คือการหา parameter สำหรับ model

the thing is deciding which features to use in model and how to use these features is very challenging thing

เช่นในพวก classification problem ก็จะมีการเลือกเส้นที่ใช้ในการจำแนกว่าเส้นไหนดีที่สุด บางครั้งความสัมพันธ์ของเส้น boundary นั้นอาจจะไม่ใช่ linear สะทีเดียว อาจจะเป็นเส้นโค้งหยึกยัก

เราในฐานะ programmer ด้วยก็ต้องหาวิธีการรับมือกับมัน เช่นการปรับ data เอามันไปใส่ log เอามันไปกำลังสอง หรือเอามันไปทำไรก็ได้ เพราะว่านี่คือ shallow learning algorithm

shalldow != deep อยู่แล้วตามความหมายในภาษาอังกฤษ

ืneural network จะมาทำหน้าที่ในการเลือก feature แทน programmer ก็คือ neural network จะเรียนรู้เกี่ยวกับ feature จาก data โดยตรงโดยไม่สนว่าคุณได้ความสัมพันธ์แบบ linear หรือ non-linear

neural network จะเรียนรู้เพื่อการผสมผสาน feature ให้เกิดประโยชน์สูงสุด และนี่ก็เป็นเหตุผลที่เราไม่ต้องไป programming เกี่ยวกับมันเยอะ

นี่คือสาเหตุว่าทำไมเราสามารถสอน รถ ให้สามารถขับตัวมันเองได้ โดยที่เราไม่ต้องเขียนโค๊ดเพื่อที่จะสร้างรถขับเคลื่อนตัวเอง

เช่นใน image recognition neural network สามารถเรียนรู้จนระบุได้ว่ามีอะไรอยู่ในรูปบ้าง

หลักการก็คล้ายๆของเดิมเราเตรียมข้อมูลให้เพื่อการเรียนรู้จะได้ model มาแล้วเอามันไป test โดนรูปที่ยังไม่เคยเจอมาก่อน

ส่วนข้อมูลที่ให้ไปอะก็แค่รูปแมวแล้วมี label แปะว่า cat ที่เหลือ newral network จะทำการสร้างสรร feature ขึ้นมาเองจาก dataset ที่เราเตรียมไว้ให้

Deep learning นั้นให้ความรู้สึกไม่เหมือนกับ programmer เมื่อเราเอามาเทียบกับ maching learning

Deep Learning จะเปรียบเสมือน black box ที่รู้ผลลัพธ์ แต่ไม่รู้ว่าอะไรคือเหตุของผลลัพธ์หรือกล่าวได้ว่าเราไม่รู้ถึงการได้มาของผลลัพธ์นั่นเอง

## How Newral Network Generate its own Features

เช่นตัวอย่างคือการที่เราตรวจสอบว่าภาพนั้นมีแมวอยู่หรือไม่

รูปแบบโครงสร้างการทำงานก็จะแบ่งเป็น

Input Layer ||  Hidden Layer  || Output Layer

ทุกๆ node ของ input layer จะเชื่อมต่อกับทุกๆ node ภายใน Hidden Layer เหมือนกันใน Hidden กับ Output Layer

แต่ละ nueron จะทำงานหรือไม่ทำงาน (ที่ส่งสัญญานประสาทตามแบบฉบับของคนอะ) ขึ้นอยู่กับ function แต่เราก็สามารถทำให้มัน advance ขึ้นได้

เช่น ปกติกราฟที่บอกเอ้าท์พุทแน่นอน 0 -- 1 ไม่มีค่าระหว่างนี้ เช่น เป็นแมวหรือไม่เป็นแมว นี่เป็นเคสที่เราไม่ใช่ stepwise function 

แต่บางครั้งเอาท์พุทเราจะอยู่ ระหว่าง! 0-1 มีตัวเลขระหว่างกลางได้  เช่น มันมีโอกาศ 10% ที่ในภาพนี้จะมีแมว

ทุกๆ neuron จะใช้ฟังก์ชั่นนั้นๆ เพื่อทำการปล่อย output 

ฟังก์ชั่นที่เป็นตัวบ่งชี้ว่าจะปล่อยความแรงของสัญญานออกมาเท่าไหร ฟังก์ชั่นนี้เรียกว่า "Activation Function "

สิ่งหนึ่งที่เราสามารถทำได้ก็คือการเปลี่ยนสถาปัตยกรรมของ network นี้ เราสามารถเพิ่มจำนวน Layer ที่มีอยุ่ได้

the deeper the network the more layers it has. That's where the deep learning come from.

ยิ่งอยากได้เยอะเท่าไหรก็ใส่ Layer เข้าไปเรื่อยๆ 

### What go on in the Hidden Layer

ก่อนจะไปเริ่มที่ Hidden Layer อะ เรียนรู้เกี่ยวกับ Input Layer ก่อนละกัน

#### First Hidden Layer

จากการที่บอกไปว่าทุกๆ input ใน Input Layer จะเชื่อมต่อกับทุกๆ node ใน First Hidden Layer นี้

How the overall goal of the nueral network will be to discover the optimal combination of features

ในความจริงที่ว่ามันเชื่อมต่อกันทุก node แสดงว่ามันจะต้องลองทุกๆรูปแบบที่เป็นไปได้

นี้คือหน้าที่ของ First Hidden Layer คือพยายามที่จะรวม feature ต่างๆเพื่อหาทางที่ดีที่สุดที่จะรวมมันเข้าด้วยกันได้

เช่นเมื่อตอนที่เราอยากรู้ว่า รูปภาพนี้มีแมวอยู่หรือไม่
 
รูปทุกๆรูปที่เราใช้ในการเทรน แต่ละรูป ทุกๆ pixel จะถูกใช้เป็น input ใน neural network นี้

สรุปก็คือทุกๆ input จะถูกส่งไปยังทุกๆ node ที่อยู่ใน First Hidden Layer

มันก็จะเริ่มรวมภาพแล้วสร้าง feature ขึ้นมาจาก pixel

neural network ตอนแรกเริ่มก็จะเริ่มที่จะ detect พวก line, edge, textures พวกนี้ก็คือ feature ที่ได้ถูกสร้างขึ้นมานั้นเอง

#### Second Hidden Layer

ก็ตามหลักการก็ใช้ output ของ First Hidden Layer

แต่ในที่นี้มันไม่ได้ทำงานเกี่ยวกับพวกเม้ดสีอะไรพวกนั้นแล้วนะ มันจะจัดการเกี่ยวกับ features ที่ first hidden layer นั้นส่งมาแทน

ตอนนี้ก็อาจจะเริ่ม detect รูปร่างสี่เหลี่ยมวงกลม

#### Third Hidden Layer

ก็รับ output (feature) มาจาก second heeden layer แล้วก็จะสร้าง feature ในแบบของ layer ตัวเอง 

อาจจะเป็นการ detect พวกแขนขาตาหู แล้วก็ส่งต่อไปยัง layer ถัดไป

นั้นก็คือ output layer เพื่อให้ระบุว่าในรูปนี้มีแมวหรือไม่มีแมวนั้นเอง

### เชิงเปรียบเทียบ

ถ้าเปรียบเทียบการทำงานก็คือ output layer คือ ceo ของบริษัทนั้นๆที่มีหน้าที่ตัดสินใจตามองค์ประกอบที่ได้รับมา 

เช่น last hidden layer บอกว่า 
>node1 ฉัน detect ขาได้
>node2 ฉัน detect ตาได้
ไปเรื่อยๆ

ตอนนี้ output layer ก็ต้องทำการตัดสินใจ 

การตัดสินใจใน output layer นั้นอาจจะบอกว่า "ฉันค่อนข้างชัวเลยที่ในภาพจะมีแมว 80%"

แต่เราในฐานะมนุษย์ที่รู้ว่าแมวคืออะไร เราก็บอก ceo คนนั้นว่าเสียใจด้วยนะ คุณแม่งมั่ว ไม่มีแมวเลยจ้าาา

ceo ก็จะต้องไปหาทางใหม่อาจจะเลือก weight น้ำหนักของเส้นประสาทนั้นๆลง (ผู้จัดการ) เช่นไอคนบอกเจอตา ตาที่เป็นแฟนยายป่าว หยอกก หรือเพิ่มน้ำหนักก็แล้วแต่

พอ ceo เริ่มทำการปรับน้ำหนักในการทำงาน ผู้จัดการ ก่อนหน้านี้ก็จะเริ่มรู้ตัวละว่าฉันมีอะไรผิดพลาดก็จะเริ่มตำหนิผู้ช่วยผู้จัดการต่อละ ก็เป็น chain ต่อไปเรื่อยๆ

ไปจนถึง input layer แต้ input layer อะไม่สามารถส่งอะไรกลับมาได้หรอกนอกจาก pixel มันอาจจะเป็นแค่รูปโง่ๆ

ดังนั้น first hidden layer ก็ต้องทำการปรับ weight ของตัวเองในที่สุด เพื่อสร้าง feature ที่แตกต่าง

process ทั้งหมดนี้ถูกเรียกว่า "Backpropagation"

ลองคิดดูก็จะเกิดคำถามว่าแล้วมันปรับ weight ยังไงเพิ่มหรือลด

สิ่งนี้ขึ้นอยู่กับ cost function ของเราค่าความชันอะ

แต่จริงแล้วอะ เราไม่รู้หรอกว่า feature ที่โดนสร้างโดน First Hidden Layer มันสร้างอะไรขึ้นมาทั้งๆที่เราบอกว่ามันอาจจะเป็น รูปร่างบลาๆ

แต่ที่เรารู้แน่ๆ เรารู้ว่า Nueral Nwtwork แตก data ออกมาเป็น chunksๆ

อย่างที่บอกว่า Neural Network เป็น black box model มารถรู้ได้จริงๆว่าอะไรเกิดขึ้นด้านใน

### Conclude

Each nueron in a neural network will be activated based on a mathematical formula called the activation function

activation function detemines how strong this neuron will fire

The nueral network is able to generate ots own features from the input data

This allow the neural network to solve both linear problems and nonlinear problems

It try all these combinations

The deeper the network the more complex and the more high level of features are generated at each layer

The pattern of learning for a neural  network is very similar to other machine learning algorithm. It make a prediction, it figures out how far off the prediction and adjust it

The process by which the error gets sent back down theough the network is called "Backpropagation"

#### Can we use neural network for everything

yes you can, you can solve all machine learning problem with the neural network

but would you want the newral network to solve every problem : NO

##### Disadvantages

##### Black Box Model

เราไม่รู้แบบแน่ชัดว่าทำไม neural network ให้เอาท์พุทออกมาแบบนี้

ลองคิดดูว่า neural network นี้ถูกใช้ในกระบวนการกฏหมาย ในการตัดสินว่าควรจำคุกทั้งหมดกี่ปีสำหรับแต่ละคน

ก็จะมีคำถามว่าทำไม นาย คนนี้ได้โทษจำคุกแค่ 2 ปีอีกคนได้ 200 ปี

นี้เป็นหนึ่งในตัวอย่างที่ black box model เสียเปรียบเป็นอย่างมาก

##### Cost

cost มาในวองรูปแบบ

1.) ปริมาณ data ที่ต้องการในจำนวนที่มาก
2.) คอมพิวเตอร์ที่แรงส์ >> ความไวในการเทรน

ทั้งสองนี้ก็มีความเชื่อมโยงกัน

ย้อนกลับไปดูที่ structure ของ neural network นี้

6 node || 6 node | 5 node | 4 node || 1 node

degree of complexity = จำนวนของ parameter
                     = (6*6) + (6*5) + (5*4) + (4*1)
                     = 90 connention == 90 parameter ==> neural network have to estimate

ปัญหาก็คือยิ่ง parameter มีค่ามากเท่าไหร จำนวน data ที่ต้องการก็มีค่ามากตามไปด้วย

เหมือนการทำเค้กเหมือนที่แม่คุณเคยทำให้อะ

ลองไปลองมา เอ้เกือบเหมือนแล้วนะ น้ำตาลน้อยไปหน่อยแป้งน้อยไปหน่อย อันไหนน้อยไปหน่อยเนี่ยยย

the more complex problrm, the more parameter you have

แล้วปริมาณ data แค่ไหนถึงจะเพียงพอ 

ในอุตสาหกรรม ส่วนมากใช้ rule of thumb หรือนั้นก็คือ 10 เท่าของ parameter 
90 para >> 900 data point

1 data point >> 900 pixel (30*30)

1 layer สำหรับทุกๆ node >> 6* 900 = 5400 pixel

แล้วถ้ามีการขยับเพิ่มขึ้นของ node เพียงอันเดียวตัวเลขแม่งกระโดดมากเลย

แล้ว data แต่ละอันต้องเป็น data ที่มีคุณภาพด้วยนะ

พอขั้นตอนการเทรน โอโห้เมื่อ data มีปริมาณมากก็ต้องมีการคำนวณที่มากตามไปด้วย ใช้ทรัพยากรในคอมพิวเตอร์ไปมากโดยสมควร

พอการมันใช้การคำนวณที่มาก เราก็จะต้องเปลี่ยนอุปกรณ์ที่ใช้ในการคำนวณแล้ว เป็น GPU

สามารถหาใช้ได้ตามงานที่เหมาะสมเลย colab, kraggle หรือ ที่เสียเงินก็ได้ ได้หมดตามที่เราต้องการ

ดังนั้นจงใช้เครื่องมือให้เหมาะกับงาน

# Pre Precessing Image Data and How RGB Work

อ่านหัวข้อย่อยในนี้แล้วไปอ่าน colab ได้เลยยย

## TensorFlow

open source machine learning framework

## Keras

Keras may be a bolt on to tensor flow

มันเป็น neural network ไลบารี่ เป็น module ที่ใช้ tensorflow ในเบื้องหลังในการคำนวณ

ไปที่ google drive ของเรา +new > more > google colabotary

ขนาดของรูปภาพที่เราจะเล่นด้วยจะอยู่ในรูปแบบ array w*h*channel

เมื่อเราไปอ่าน doc ของ keras เราสามารถใช้ pre train weight ง่ายๆโดยการที่ระบุ weight='imagenet'

ก็จะพบเจอกับสิ่งหนึ่งที่เรียกว่า imagenet

### imagenet

เป็น database รูปภาพขนาดใหญ่ ที่แต่ละรูปนั้นมี label แปะอยู่แล้ว

ก็เลยทำให้เห็นบ่อยมากใน opencv, machine learning

# coming