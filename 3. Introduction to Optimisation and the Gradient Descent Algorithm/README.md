# How a Machine Learns

มองแบบคนไม่รู้เรื่องก็คือการให้ข้อมูลจำนวนมากแก่คอม แล้วก็เอาผลลัพท์ที่ได้จากการคำนวณกลับมา

สิ่งที่ computer เรียนจริงๆนั้นคือการหาความ สัมพันธ์ของข้อมูล

เทคนิคของ machine learning มีพื้นฐานอยู่ 3 ขั้นตอน (loop)

### step 1 to make a prediction

the very first time this happens, the first prediction is pretty much like a completely random guesses.

### step 2 calculate error

we need to measure, how good the prediction was, how far off we were from the data

### step 3 learning 

this is where we adjust our initail prediction

## cost function

a very big partt of machine learning process is optimizing for solution that has the lowest costt

# Learning rate 

multipier in code >> jupter notebook >> how big was the step we take
เมื่อเราไม่ระวังในการเขียน algorithm ก็อาจจะตกอยู่ในสถาณการณ์ที่่ก้าวมันใหญ่เกินทำให้ไมสามารถ find minimun value ได้้เลย หรืออาจจะทำให้เกิด overflow ได้ในตอนเปลีย่นแปลงได้เลยเพราะวา่ step ที่เยอะไป

ดังนั้นเราต้้องหาทางเลือกค่่า learning rate ที่่เหมาะสม