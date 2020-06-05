# introduction

## what is maching learning 

ย้อนกลับไปในอดีตคอมพิวเตอร์ถูกคิดค้นชึ้นจากความขี้เกียจของมนุษย์ แล้วก็พัฒนามาในรูปแบบต่าๆอยู่เรื่อยๆ
จนมาอยู่ในยุคที่เราเขียน programming language

ถ้าเราคิดเกี่ยวกับการแยกผลไม้ที่ไม่สุกบนสายพานการลำเลียงหล่ะโปรแกรมที่เราเขียนจะอยู่ในรูปแบบประมาณไหน

```python
if color == 'red':
    keep()
```

แล้วถ้าเกิดสปหน้าโรงงานของเราต้องการคัดพริกสีเขียวแทน

```python
if color == 'green':
    keep()
```

มันจะดีมากถ้าเราสามารถสอนคอมพิวเตอร์ได้ว่าอะไรคือผลไม้ชนิดไหน เช่น อาจจะเป็นถ้าวัตถุนั้นหลักประมาณ x g, รัศมีประมาณ y cm นี่คือพริก แต่! มีผลไม้มากมายบนโลกมันอาจจะซ้ำกันก็ได้

ถ้าเราจะสอนเด็กให้รู้จักกับผลไม้นั้นๆ เราก็ต้องให้ข้อมูลผลไม้นั้นมากๆ เช่น รูปของมันในมุมต่างๆ สี ขนาด น้ำหนัก เป็นต้น

ตัวอย่างด้านบนก็ใกล้เคียงกันกับ machine learning เหมือนกับการเรียนรู้ของเด็กน้อยเลย ให้ข้อมูลลักษณะต่างๆแก่คอมพิวเตอร์เยอะๆ 

เมื่อเราทำสิ่งที่เดียวกันกับที่ทำแบบนั้นให้กับคอมพิวเตอร์เราเรียกมันว่า "supervised learning" โดนการให้ข้อมูลที่จะนำไป train

เมื่อเรามี machine learning model ที่รู้แล้วว่าผลไม้ต่างๆเป็นยังไงก็สามารถนำไปใช้งานตามเราต้องการได้

## what is data science

data science เกี่ยวกับการทำให้ข้อมูลนั้นมีค่า

หลังจาก ibm ทำการคิดค้น relational database ขึ้นมานั้น เราก็คิดได้ว่าจากข้อมูลที่เรามีนั้นเราสามารถนำมันมาทำอะไรได้อีก

### data mining

"data mining is the application of specific algorithms for extracting patterns from data"

เมื่อก่อนเราทำการตีความมันด้วยวิธีการทางสถิติแบบเก่าๆ

### data mining + computer science = data science

ใช้เพื่อนทำนายเหตุการณ์ต่างๆได้ เพื่อนำมาใช้ใน business model 

by "data science", we mean almost everything that has something to do with data

บางคนก็แค่คิดว่ามันคือการเอาข้อมูลมายัดใส่ ai ml ให้มันประมวลผลตามที่เราต้องการ แต่จริงๆ data science มีมากกว่านั้น

เหมือนคนเรา ต้องมีอาหารกินก่อน ถึงจะเริ่มคิดถึงที่อยู่อาศัย พอเริ่มมีก็เริ่มต้องการ relation มากขึ้นไปเรื่อยๆ 

ใน data science นั้น มี hierarchy ดังนี้
1.al, deep learning                                                                             || machine learning expert
2.a/b testing, experimentation, simple ml algorithm                                             || machine learning expert (learn/optimize)
3.analytics, metrics, segments, aggreates, features, training data                              || data scientist (aggregate/label)
4.cleaning, anomaly detection, prep                                                             || data scientist (explore/transform)
5.reliable data flow, infrastructure, pipelines, etl, structured and unstructured data storage  || data engineer (move/store)
6.instrumentation, logging, sensors, external data, user generated content                      || data engineer (collect)

ในทุกๆ model เราพยายามที่จะแก้ไขปัญหาที่มีอยู่บนโลก ก่อนอื่นเราต้องเข้าใจ how to clean, segment and visualize raw data หลังจากนั้นก็หา algorithm เพื่อนมาใช้ถอดความหมายจาก data

## coming soon

# coming soon too