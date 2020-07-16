# spam filter part 1

## formulate question

เราต้องทำการเปลี่ยน business problem >> machine learning problem

ตอนนี้เรามี business problem จากบริษัทที่ชัดเจนคือการกรองอีเมลล์ที่เป็นสแปม

แล้วในมุมมองของ machine learning หล่ะ เราก็ต้องรู้ก่อว่าอันไหนคืออีเมลที่เป็นสแปมอันไหนไม่ใช่ 

ดังนั้นจุตประวงค์ของเราในตอนนี้ก็คือการไปหามาว่า ลักษณะของอีเมลที่เป็น spam เป็นยังไงเพื่อให้เราสามารถทำการ แยกมันออกจากเมลทั่วไปได้

งานเก่าที่เราทำการทำนายราคาบ้านกับอันนี้แม่งต่างกันมาก

อันเก่าคือการที่เราพยายามทำนาย "คุณภาพ" จาก "ลักษณะ" ของสิ่งที่เรามีอยู่

ซึ่งมันจัดอยู่ใน regression problem

แต่ตอนนี้เราต้องทำการจัดหมวดหมู่สิ่งของ มันจัดอยู่ใน classidication problem

ถ้าเป็น regression > we are fitting to the data 
แต่ถ้าเป็น classification > we are seperating the data

ก่อนหน้านี้เราทำเกี่ยวกับตัวเลข คำนวณมัน และยัดมันใส่ลงใน algorithm
ตอนนี้เราจะมายุ่งเกี่ยวกับ email แทนเกี่ยวข้องกับ text 

แล้วเราจะเทรน algorithm ของเรายังไงโดยใช้ text data

ยัดตัวมันเองเข้าไป ไม่เหมาะสมแน่ๆบอกเลย ดังนั้นเราต้องการวิธีการที่จะเปลี่ยนแปลงข้อมูล text ไปเป็นรูปแบบที่ algorithm เราสามารถเข้าใจมันได้

## gather data
 
ข้อมูลที่เราได้มาทั้งหมดมาจาก spam assassin หมดเลย ลองเข้าไปค้นหาใน google ได้ >> spam assassin public corpus


Glossary:

Corpus
:: is defined as a large and structured set of texts.

Document
:: in this context a document refers to a particular email in corpus

>>> before going to the next step go and read Naive Bayes Classifier

## clean data

jupyter notebook

## explor & visualisation

jupyter notebook

### Word Cloud

เป็นเหมือนขั้นตอนที่ทำให้เรามองเห็นภาพรวมของ data ได้มากขึ้น

แบบคำที่เยอะที่สุดจะปรากฏใหญ่ที่สุดในรูป อารมณ์แบบภาพโฆษณา

## training algorithm

### full matrix

เราจะสร้าง full matrix ที่ประกอบไปด้วย column

doc_id | word | label | occurence

doc_id คือ id ของ email แต่ละ email
word คือ คำ
label คือ ค่าที่บอกว่าเป็นสแปม(1) หรือไม่เป็น spam(0)
occurence คือ การปรากฎขึ้นมาของคำนั้นใน email นั้นๆ

สาเหตุที่เราเรียกมันว่า full matrix ก็เพราะว่า สำหรับทุกๆ email มันจะมีแถวข้อมูลของ คำใน vocab ทุกๆคำไม่ว่าใน email นั้นจะมีคำนั้นๆปรากฎอยู่หรือไม่ก็ตาม

ดังนั้นถ้าเรากำหนดขนาดของคำที่เราสนใจ = m

ใน 1 doc_id จะมีซ้ำกันทั้งหมด m ครั้งตามชนาดของคำ

### sparse matrix

จาก full matrix อะ ถ้าเราเอามาทุกอันมันก็เปลืองมันก็เลยเกิดสิ่งนี้ขึ้นมาาาา

อันนี้ก็เหมือนกับของอันด้านบนแต่ email ณ คำนั้นๆ ที่ไม่มีคำนั้นๆอยู่ใน email นั้น จะไม่เอาเข้ามาอยู่ใน matrix ด้วย

สรุป> เอาแต่คำที่มีซ้ำกันใน vocab ของเรา

jupyter notebook

## model evaluate

แล้วเราจะรู้ได้ไงว่า model ที่เราสร้างมานั้นมันดีแค่ไหน

ก็วัดตามอัตราส่วนที่ถูกจากทั้งหมดเป็นเท่าไหรนั้นคือเหตุผลที่เราทำการแบ่ง test กับ train ออกจากกัน

### Accuracy

ค่าความถูกต้องก็จะมีค่าเป็น

accuracy = ที่ทำนายถูก / ที่ทำนายทั้งหมด

### more on evaluation

ถ้าเราตั้งให้ positive คือการที่เมลนั้นเป็นสแปม
แล้ว negative คือการที่เมลนั้นไม่ใช่สแปม

เราก็จะมี 4 รูปแบบ ว่า จริงหรือเท็จ คือ

True Positive >> คือการที่ผลออกมาว่า เป็นสแปม และอีเมลนั้นก็เป็น สแปมจริงๆ

False Positive >> คือการที่ผลออกมาว่า เป็นสแปม แต่อีเมลนั้นไม่ได้เป็นสแปม

True Negative >> คือการที่ผลออกมาว่า ไม่เป็นสแปม และอีเมลนั้นไม่ใช่สแปม

False Negative >> คือการที่ผลออกมาว่า ไม่เป็นสแปม แต่จริงๆแล้วเมลนั้นเป็นสแปม

### Recall metric

recall is also known as the sensitivity 

recall score = True Positive / (True Positives + False Negatives)

ถ้ามองจากสมการจะรู้ว่าอะไรคือ key หลักของสมการนี้ นั้นก็คือ ค่าของ false negative

false negative คือโดนระบุว่าไม่เป็น spam ทั้งๆที่จริงๆแล้วตัวเองเป็น spam

เมื่อ false negative มีค่า = 0 จะทำให้ recall score มีค่าสูงสุดนั้นก็คือ 1  >> n/(n+0)
และในทางตรงกันข้ามถ้าค่า flase negative มีค่ามากขึ้นค่า recall score ก็จะน้อยลงเช่นกัน

recall score เป็นตัววัดคุณภาพหรือความเรียบร้อย ว่าเรากรองได้เท่าไหรจากเท่าไหร

แต่ recall score ก็ยังมีจุดอ่อนอยู่นั้นก็คือ การที่เราแปะแยกทุกอันว่า email ตอนนี้ Flase negative = 0 แน่นอนเพราะเราเอาทุกเมลเป็นสแปมเลย แต่ n / (n + 0) ยังไงก็มีค่าเท่ากับ 1

### Precision metric
 
Precision is also known as the positive prediction value

Precision = True Positive / (True Positives + False Positives)

เหมือนกับ recall เลยแค่ส่วนตอนท้ายไม่เหมือนกันที่เดียว

precision ถ้าอยากได้สูงก็ต้องมี false positive ที่น้อย

เป็นตัววัดคุณภาพรูปแบบหนึ่งเหมือนกัน ว่าเรากรองได้ของปลอมเท่าไหร 

กรองว่าเมลนั้นเป็นสแปมทั้งๆที่เมลนั้นไม่ใช่สแปม

#### สถานการณ์

ความไม่ balance ใน dataset มีผลต่อ model เราด้วยนะ ตอนเราแบ่งเพื่อทำการ train มันอะ มันจะมีกรณีที่เราแรนดอมแล้วค่า 1, 0 ที่เราแยก อยู่เกาะกลุ่มกันเกินไป เช่น กรณีตัวอย่างนี้

model ของตัวที่ใช้ทำนายมะเร็ง เบื้องต้นอะเรา labeled ทุกคนใน train data ว่าไม่ได้เป็นมะเร็งหมดเลยนะ(0) เป็นมะเร็ง(1)

train dataset
คนที่| label|
1  |   0  |
2  |   0  |
3  |   0  |
4  |   0  |
5  |   0  |

ซึ่งมันก็เป็นเรืองปกติอยู่แล้วหรือป่าวที่คนส่วนใหญ่ไม่เป็นมะเร็งอะ ถ้าเอา data มา test ของคนส่วนใหญ่ ที่มีคนไม่เป็นมะเร็งเป็นส่วนมากอยู่แล้ว model นั้นก็จะมีความแม่นยำที่สูงมากๆๆๆ เพราะตรงกับความจริงแบบปลอมๆ

คำถามต่อมารูปร่างของ recall metric กับ precision metric จะเป็นอย่างไร ?

precision = 0
recall = 0

เพราะว่ามันไม่มีคนไข้สักคนที่เราระบุว่าเป็น true positive ไง ในขณะที่มีค่า false neg, false pos

คำถามต่อมา มันมีความสัมพัมธ์กันระหว่าง recall กับ precision ไหม

ตอบ: มีถ้าอันหนึ่งเพิ่มอันหนึ่งจะลด เนื่องจากความผิดผลาดของ model ที่เราทำนายผิดเกิจขึ้นจาก 2 แบบ false neg, flase pos >> ความผิดพลาด

ถ้าอันใดเพิ่มอันใดต้องลดก็มีผลต่อ recall กับ precision ตามนั้น

# Naive Bayes Classifier

ข้อดีของโมเดลนี้คือเรียบง่ายและไว มักใช้ในการทำนายพยากรณ์อากาศ หรือ ทำการ filter spam

## how it actually work

เพื่อที่จะแยกว่า email นั้นเป็น spam หรือไม่ naive bayes จะทำการเทียบตามความน่าจะเป็น

มันจะคำนวณความน่าจะเป็นว่า email นั้นเป็น spam หรือไม่เป็น

ถ้าเกิด email นั้นถูกคำนวณว่ามีความน่าจะเป็น spam สูงกว่าไม่เป็น email นั้นจะถูกจัดอยู่ในหมวดของ spam

ง่ายๆก็คือดูแค่ตัวเลขความน่าจะเป็นสองเลขนี้

## where is that probability actually come from

มันจะใช้หลักการทางสิถิติมาช่วย วิชาสถิติอยู่ในทุก model ของ machine learning

ย้อนกลับไปดูที่ probability >> ne / ns เพื่อเอามาทำนายหัวก้อยๆ ที่เราเรียนมา แต่แค่นั้นก็ยังไม่เพียงพอต่อการ filter spam msg แน่นอน เพราะว่าถ้าแค่ดูเปอร์เซ็นที่ mail เป็น spam ต่อเมล 

การส่งเมลแต่ละครั้งมีปัญหาแน่ๆ ดังนั้นเราต้องเข้าไปดูที่เนื้อความของเมลนั้นด้วย

มันก็จะมีคำที่สามารถบ่งบอกได้ว่ามีโอกาศที่เป็น spam ได้อยู่ แต่ก็ต้องไปดูหัวข้อของ email นั้นด้วยว่ามันหมายถึงอะไร

เช่น email นี้มีคำว่า viagra อยู่ คำถามคือ โอกสเท่าไหรที่ email นี้จะเป็น spam >> condition probability

p a given b คือเหตุการณ์ทั้งสองเหตุการณ์ขึ้นอยู่ต่อกัน โดยเกิดเหตุการณ์ B ก่อนแล้ว A ตามมา

P(Spam | Viagra) >> โอกาสที่มี viagra ใน email แล้วเป็น spam

P(Spam|Viagra) = P(Spam intersect Viagra)
                        P(Viagra)

bayes theorem >> how to revert condition probability to make it easier

P(Spam|Viagra) = P(Viagra|Spam)P(Spam)
                        P(Viagra)

สิ่งที่เราควรจะคิดต่อไปก็คือใน ใน 1 email อะ ถ้ามันมีมากกว่า 1 คำหล่ะ เพราะมันไม่มีเมลไหนบอกหรอกว่าแบบ vigra อันเดียวเพียวๆ มันต้องมีอย่างอื่นแบบเช่น free, Cash, Expert บลาๆ

P(Spam|Viagra)
P(Spam|Free)
P(Spam|Cash)

ในกรณีที่เป็น Spam

P(Spam|Viagra) intersect P(Spam|Free) intercext P(Spam|Cash) 
is equal to >>joint probability<<
P(Spam|Viagra)*P(Spam|Free)*P(Spam|Cash) 

ในทางตรงกันข้ามที่เป็น email ปกติ

P(Normal|Viagra) intersect P(Normal|Free) intercext P(Normal|Cash) 
is equal to >>joint probability<<
P(Normal|Viagra)*P(Normal|Free)*P(Normal|Cash)

เสร็จแล้วทำการเปรียบเทียบ probability ว่าอันไหนสูงกว่า

ดังนั้นเราต้องหา "Bag of Words" เพื่อทำการจำแนกคำต่างๆ

ดังนั้น feature ของเราก็คือ frequency ของคำแต่ละคำ

เราจะมองเจาะจงไปที่คำแต่ละคำเลย ไม่สนแกรมม่าไม่สนรูปเต็มของมัน เช่น
New York >> New & York

นั้นเป็นเหตุผลว่าทำไม algorithm นี้ถึงถูกเรียกว่า Naive

# Natural Language Processing

เป็น sub field ของ ai อีกที

เราจะใช้มันเกี่ยวกับอะไรในงานนี้ :> เราจะใช้มันทำการเตรียมคำ เพื่อป้อนให้กับ training algorithm ของเรา

คือการ convert email body >> from ที่ algorithm เราสามารถเข้าใจได้ 

เพราะตามที่เคยบอกไปว่าเราไม่สามารถยัดข้อความลงไปทั้งก้อนให้กับ algorithm นี้ได้

## pre-process

1) converting to lowwer case
2) Tokenising แยกคำออกเป็นคำๆ
3) Removing stop words พวกอักขระพิเศษที่ไม่ช่วยในการสื่อสาร
4) Stripping oyt HTML Tags
5) Word Stemming เอาคำกลับมาเป็นต้นคำ go <<goes, going, gone>>
