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

## training algorithm

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