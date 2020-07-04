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