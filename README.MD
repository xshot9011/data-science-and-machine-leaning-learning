# introduction

more on deep learning --> [click](https://www.notion.so/Deep-Learning-6e79f494cbda4a6dae2db52cc15c91bd)

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

## processes to solve problem

แล้วเราจะเอา data science มาปรับใช้กับปัญหาได้ยังไง
1. formulate the question
นัก data science จะตั้งคำถามอย่างรอบคอบ
เพราะว่าคำถามที่ดีจะเป็นตัวกำหนดงานวิจัยและส่งผลต่อ data ที่เราออกไปเก็บ

2. gatherind the data
เก็บ data มาเพื่อนำมาใช้ในการตอบคำถามที่เราตั้งขึ้น

3. clean data
แต่ข้อมูลในโลกความเป็นจริงแม่งง๋องง๋อยมาก เราต้องทำการ clean มัน บางครั้งก็จะมีส่วนที่หายไปหรือส่วนที่ไม่สมบูรณ์ใน data นั้นๆ 
เราต้องหา error หรือกำหนดรูปแบบของข้อมูลใหม่

4. expore & visualise
ทำความเข้าใจเกี่ยวกับ data ที่ได้มา

5. training algorithm
เพื่อหารูปแบบใน data ที่เราทำการเก็บมา

ุ6. evaluate
ทำการประเมินผลว่าผลลัพธ์ที่ได้มานั้น ตอบคำถามตามที่เราตั้งไหมและความแม่นยำของมันมีเท่าไหร

# nice introduction

to solve a real life business and figuring out how to solve.
This problem will test your engineering mindset.

Common misconception these days is that to solve problems you need to use the most sophisticated model possible (a.i.).

However in reality, AI is not the only thing that matters and AI doesn't solve all problems.

Now, depending on your specific problem that you're faced with, AI system require amount of data and high cost to be built

As you are engineer, you should think about how to solve your problem in an efficient way

if you know many tools and use tools for the right jobs

# model Zoo

คือสถานที่ที่มีคนทำ model ไว้ให้เราแล้วเราแค่เอามันไปใช้งานต่อ เราไม่จำเป็นต้องทำการ train มันเอง

[https://www.modelzoo.co/](model zoo)

# at the end

practice on [kaggle](https://www.kaggle.com/)

One of the hardest parts will be finding the right data, so here are some of my favourite places to start looking: 

[UK goverment data](https://data.gov.uk/search)

[United States Census API](https://www.census.gov/data/developers/data-sets.html)

[Curated datasets by Figure-Eight.com](https://appen.com/resources/datasets/)

[List of Awesome Datasets](https://github.com/awesomedata/awesome-public-datasets)

[20 Cool Datasets by Oliver Cameron](https://gist.github.com/olivercameron/482dcfe8f34d66b536b1048eefe8b40d#file-datasets-csv)
