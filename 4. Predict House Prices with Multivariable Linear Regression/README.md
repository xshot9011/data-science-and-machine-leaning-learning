# multivariable linear regression

## ขั้นตอนแรกใน data science หรือ machine laerning คือการเริ่มตั้งปัญหา

ปัญหาของเราอนนี้คือ บ้านหลังนั้นราคาเท่าไหรในย่านๆหนึ่ง ก็ลองมองดูว่า ราคาบ้านแถวนั้นอาจจะขึ้นอยู่กับหลายัจจัย ทำเล จำนวนห้อง อาชญากรรมแถวนั้น หรืออื่นๆอีกมากมาย

แล้วก็ไปดูว่าปัจจัยไหนมีผลมากผลน้อยตามไป

## gathering data

ถ้าเราเรียนอะ เขาก็จะมี data มาให้แล้ว ถ้าเราไปทำงานหรือทำเอง เราจะหา data ได้จากที่ไหน

ถ้าเพื่อการเรียนรู้ module บางตัวของ python มาพร้อมกัน dataset ที่เราสามารถใช้งานได้ >> scikit-learn ก็มีให้นา

[dataset](https://scikit-learn.org/stable/datasets/index.html)  

## clean data and explor & visualise data

ส่วนใหญ่เขาจะทำในขั้นตนเดียวกันได้เลย

6 คำถามแรกที่เราเริ่มทำกับ dataset ที่เราพึ่งได้มาสดๆร้อนๆ

### source of the data

dataset นั้นมาจากไหน 

### description of the data set

สามารถหาคำอธิบาเกี่ยวกับ dataset นั้นแบบย่อๆได้หรือไม่  

เพื่อให้รู้ว่าที่เก็บข้อมูลมาได้คือข้อมูลแบบไหน และ วิธีการเก็บข้อมูลคืออะไร

### number of data point

ทำกับ dataset เล็กใหญ่มีผลแตกต่างกัน ใช้เทคนิคที่ต่างกัน ใช้อุปกรณที่ต่างกัน

### number of features

สำหรับทุกๆ data point มี col เท่าไหร เพราะว่ามันบอกเราได้ว่า ในแต่ละ data point นั้น เรามีข้อมูลอะไรเกี่ยวกับมันบ้าง

### names of the features

### description of the features

เพื่อให้เราเข้าใจว่า data set นี้จริงๆแล้วมัน measure อะไร

#### correlation

ไปดูความสัมพันธ์ของข้อมูล อาจจะมีแบบเหมือนเส้นตรงเกาะกลุ่มกันแบบเพิ่มขึ้น หรือลดลง

correlation จะแข็งแรงแค่ไหนขึ้นอยู่กับช่วงของข้อมูล ว่าจะกระจายออกจากกันแค่ไหน

มีค่าอยู่ที่ระหว่าง -1 <= correlation <= 1

ทำไมเราต้องสนใจค่า correlation นี้ด้วย

เพราะว่ามันบอกถึงความเกาะกลุ่มของข้อมูลในเชิงความสัมพันณ์และทิศทางของข้อมูล

datafeame.corr()

ตอนนี้อะเราไม่ได้มี correlation แค่ระหว่าง feature กับ target แต่เรามี feature กับ feature ด้วย 

##### 1) คำถาม: ถ้า correlation ระหว่าง feature มีค่าสูง เป็นสิ่งที่ดีหรือไม่ดี

ตอบ: แล้วแต่ 55555+ อาจจะดีหรือไม่ดีก็ได้

ลองคิดเกี่ยวกับการทำนายมวลกระดูก จาก ข้อมูลดังนนี้

1) อายุ
2) เปอร์เซ็นไขมัน
3) น้ำหนัก

เห็นอะไรไหม เปอร์เซ็นไขมันกับน้ำหนัก โดยส่วนมากมักจะไปในทิศทางเดียวกัน เพราะคนอ้วน => น้ำหนักเยอะ
แต่ก็มีคนส่วนน้อยมาก พวกเล่นกล้ามที่น้ำหนักจะเยอะ แต่มวลไขมันจะน้อย แต่นั้นเป็นเพียงคนส่วนน้อยเท่านั้น

ความยากของมันคือการที่เราจะรู้ได้ยังไงว่า อันไหนส่งผลมากกว่ากัน หรือ ส่งผลในอัตราส่วนต่อกันเท่าไหร

มันเป็น redundant กัน

อันนี้ไม่ชัวจากไหน แต่มีคำศัพท์ที่่เรียกว่า Multicoillinearity

เกิดขึ้นเมื่อมี 2 หรือมากกว่า feature ที่มีความสัมพันธ์ต่อกันสูงมาก เหมือน น้ำหนักกับเปอร์เซ็นไขมันอันนั้น

เมื่อเกิดปัญหานี้อะ การประมาณ(predict) ของเราจะไม่ค่อยแม่นยำเท่าที่ควร

###### แต่ correlation สูง ไม่ได้หมายความว่าจะเกิด multicollinearity นะ

เราต้องทำการเช็คดู data ว่ามันมี high correlation ไหม ถ้ามีแล้วมันเกิดจากอะไร มันใช่ปัญหา multocollinearity ไหม

กลับเข้าไปดูใน df ที่เราทำการ mask นะจะเจอ NOX กับ INDUSH เนี่ยมันสูงป่ะ สูง 0.76 

ดังนั้นเมื่อเราเจอเราก็ทำการสำรวจดูมันคือ มลพิษ กับ โรงงานอุตสาหกรรม ซึ่งมันเป็นความ redendant ของข้อมูลแบบแน่นอน เพราะยิ่ง โรงงานเยอะ มลพิษก็ยิ่งเยอะ

ตัดภาพมาที่ค่า RAD จะเห็นได้ว่าข้อมูลจะประมาณนี้
   ||||                         "
   ||||                         |
   ||||                         |
  ||||||                        |
|||||||||                       |

จะเห็นได้ว่า RAD กับ TAX นั้นมีค่าสูงมาก พอกลับไปดูที่ RAD จะเห็นได้วาค่ามันไม่ continuous ตามด้านบนเลย

ดังนั้นมันสำคัญมากที่เราจะ

##### 2) ดู correlations กับ target

มันมีบางส่วนอะ ที่แบบเข้าใกล้ 0 มากๆเลยไม่ว่าทั้งทาง - และทาง +

เช่น CHAS DIS

พักมาดูความสัมพันธ์ของ DIS กับ PRICE จะเห็นได้ว่ามันต่ำมากกกกกก แต่ที่น่าสนใจของมันก็คือ ค่า correlatoin ระห่าง DIS กับ INDUS แม่ง(-)สูงมากเลยนะ เพราะว่าใน indestial area บางทีนั้นก็คือแหล่ที่เกิดการจ้างงานเยอะ ยิ่งไกลจากพื้นที่ที่เกิดการจ้างงานเยอะก็แสดงว่าจำนวนของ industial น้อยลงด้วย

ก็ต้องย้อนกลับมาดูว่าถ้าเราเอา DIS เข้าไปใน model ด้วยแล้วอะ มันจะเกิดผลในทางที่ดีขึ้นหรือแย่ลง?

#### what you should learn from above

1) identified strong correlations
2) simplify by excluding irrelevant data
3) test for multicollinearity

##### weakness > looking at correlation

1) required continuous data only!

ex. index, true and false value is not con. data

2) correllation does not imply causation

ของบางอย่างมันเคลื่อนที่ไปพร้อมกัน แต่มันไม่ได้หมายความว่าอีกคนทำให้เกิดอีกสิ่งหนึ่งขึ้น

เช่น คนที่ดื่มน้ำเปล่าทุกคนในปี 1850 ตายหมดแล้ว แต่มันไม่ได้หมายความว่าน้ำเปล่าฆ่าพวกเขานิ

เช่นตัวอย่างข้อมูลในเว็บ correlation อัตราการหย่าล้าง กับ การบริโภคมาการิน แม่ง correlation = 99% ซึ่งเยอะมาก แต่ นั้นก็ไม่เกี่ยวเลย เป็นเพียงโชชะตาที่พาเธอมาเจอกับฉันเฉยๆ

3) linear relationships only

มันมีกราฟข้อมูลหลายแบบเลยที่มีค่า correlation สูงแต่ ! ไม่ใช่ linear นั้นก็ใช้ไม่ได้นะ

เมื่อเราทำ data  visualisation เราก็จะเจอ outliner หรือคววามสัมพันธ์ที่อาจจะไม่ใช่ linear ใช้กราฟฟิคช่วยทำให้เราสามารถหาความสัมพันธ์ที่ซ่อนอยู่ภายใต้ของมูลนั้นๆได้

## training algorithm

we are going to combine explanatory feature to estimate price in boston 

and the model of choice is called multi variable regression

สมการที่มีตัวแปรเดียวของเราตอนแรกจะเป็น

y = theta0 + theta1*x

ตอนนี้เนื่องจากเป็น multi variabel สมการเราจะเป็น

y = theta0 + theta1*x1 + theta2*x2 ... thetan*xn 

where x1, .. xn is feature

แต่ก่อนที่จะเริ่มการเรียนรู้ สงสัยไหมว่า. . .

### wht this technique was called "regression"

ย้อนไปหาชายชื่อ francis galton สิ่งประดิษฐ์ที่โลกจดจำเขาได้ก็คือ กัลตันบอร์ด ไอบอร์ดที่คล้ายๆ นาฬิกาทราย เหมือนเครื่อเล่นเกมส์ เพื่อแสดงถึง normal distribution

galton เขาสนใจการเปลี่ยนแปลงของขนาดสิ่งต่างๆ ดูทั้งขนาดของเมล็ดพืช ขนาดของผู้คนเมื่อผ่านไปตามเจนเนอเรชั่น

เช่น เมื่อมีพ่อที่สูงมากๆๆ เมื่อมีลูก ลูกคนนั้นจะมักเตี้ยกว่าพ่อ galton เรียกสิ่งนี้ว่า "regression to the mean" 

เป็นที่มาของ regressions

## deploy and evaluate

ตอนนี้เราก็จะมาทำการ ประเมิน ผลงานของเราเพื่อดูว่ามีปัญหา หรือ เราสามารถปรับปรุงแก้ไขมันได้ไหม

เหมือนไปหาหมอเพื่อทำการเช็คร่างกาย เขาก็จะตรวขเรามากมายเพื่อทำการวิเคราห์ ว่ามีปัญหาอะไรไหม อะไรมากไปอะไรน้อยไป

ใน regression ก็จะเป็น r-square, p-values, VIF, BIC

### data transformations

ลองกลับไปดูที่กราฟ ราคาบ้าน ของเรา จะเห็นได้ว่ามันมีความเบ้อยู่ไม่เหมือน normal distribution สักเท่าไหร

ถ้าเบ้ขวาก็จะมี ค่า skew เป็น  + ถ้าเบ้ซ้ายก็ตรงกันข้าม 

แล้วเราจะสามาถเปลี่ยนแปลงยังไงได้บ้างกับข้อมูลเหล่านี้

เราก็สามารถทำการ คูณ 2 เข้าไปหรือ หารมันออกก็ได้

แต่ถ้าเราทำอย่างงั้นมันก็เหมือนไม่มีอะไรเกิดขึ้นอะดิ เพราะข้อมูลทั้งหมดถูกกระทำด้วยรูปแบบเดียวกัน เพราะฉนั้นเราต้องหาวิธีการที่มันกระเทือนส่วนหัวมากที่สุด(ที่เบ้ออกไป)

#### log

เราใช้ log เพื่อที่จะให้สมการเส้นโค้งมีความตรงมากขึ้นในทางคณิตศาสตร์

ถ้าราคา 7 >> ln >> 1.95
      50 >> ln >> 3.91

หลังจากการทำ data transform ด้วย log 

เราจะเห็นได้ว่าค่า  r-square ของเราดีขึ้นแบบเห็นได้ชัดเลย

แต่ถ้าเราเปลีย่นแปลงมันด้วยอะไร ตอนจะอ่านค่าอย่าลืมย้อนกลับมันละกัน ไม่ใช่ที่เราอ่นจะไม่ใช่ข้อมูลที่แท้จริงที่เราต้องการ

### evaluating coefficients & p-value

หลังจากที่เราทำ data transformations ไป เราไม่ได้สนใจเครื่องหมายของ theta หรือ coeff เท่าไหรหนัก แต่เราสนใจที่ ความสำคัญของมันแทน (ขนาดของตัวเลข)

เหมือนแต่ละการเช็คการใช้งานของหมอ ก็จะมีระดับบอกว่าประมาณไหนโอเค อันไหนต่ำไปอันไหนสูงไป

ในนี้เราจะเรียกสิ่งที่เป็นตัวกำหนกว่า p-value คล้ายกับในทางสถิติเลย คือ ถ้า น้อยกว่า แสดงว่า fail to reject null hypothesis 

เน้น บอกว่าคล้ายๆนะ 

ถ้าเราใช้ sklearn อย่างเดียวจะไม่พอต่อการคำนวณดังนั้นเราต้อง import model ทางสถิติเพิ่มเข้ามาอีก

เข้าไปดูใน jupyter notebook ได้เลย

### understandind VIF & Testing for Multicollinearity

ที่มี features สองตัวมีความสัมพันธ์ต่อกันสูง 

เมื่อเกิดปัญหานี้ขึ้นผลเสียที่ตามมาต่อ model ของเราก็คือ 

1) loss of reliability
เสียความน่าเชื่อถือในการประมาณผลกระทบของ feature นั้นๆ

2) high variability in theta(n) estimate
เมื่อมีการเปลี่ยนแปลงเล็กๆของตัวแปรจะทำใหเกิดผลกระทบที่ยิ่งใหญ่ขึ้น
ทำใ้ห theta ของเราไม่คงที่อาจจะเปลี่ยนจาก + >>> -  เลยก็เป็นได้

3) strange findings !!!
model ที่ได้มา ไม่เมคเซนต์ สิ่งนี้คือ เป็นอาการหลักเลยที่เกิดจากปัญหานี้

อันเก่าอะ เราแค่ดูจากค่า corr หรือ coef ที่ model เราทำนายออกมา แล้วมองด้วยตาว่ามันเมคเซนต์รึป่าว 

แต่ตอนนี้เราจะมาเรียนการใช้ matrix เพื่อระบุถึงปัญหา 

เราใช้วิชาสถิติที่ดูสิ่งที่เรียกว่า Varianace Inflation Factor (VIF) 

#### VIF

สงสัยละสิว่ามันทำงานยังไง ไม่บอกหรอก ขอก๊อปภาษาอังกฤษมาละกัน

measure of collinearity among the features with in multiple regression 

it will split up the number that will quantifies the severity of Multicollinearity

เหมือนกับที่เราทำตอน p-value ว่ามันมากกว่าหรือน้อยกว่า Threshould

##### step to calculate

1) regression run of all the other features against {var}
2) 1/(1-(rsquare of this regression))

for each VIFs, if it's VIF > 10, it's seem to cause the problem

#### features selection in context of regression

simple model is the good thing to use

แล้วจะทำอย่างไรให้มันเป็น model ที่เรียบง่ายหล่ะ ก็ตัดตัว features ที่ไม่จำเป็นออกไง

แล้วตัวแปรไหนละที่ควรโดนตัด

ถ้ามองดูที่ค่า  correlation ของ DIS กับ PRICE จะเห็นได้ว่ามีค่าแค่ 0.25 เท่านั้น แต่กลับมีค่า corr สูงมากเมื่ออยู่กับ INDUS = -0.71 เลยทีเดียว เราเลยไปดูมันที่ค่า p-value => 0.000 มีค่าสูงมากเลยทีเดียว เราเลยเก็บมันไว้

แต่ถ้าเราไปดูที่ค่า p-value ของ INDUS จะอยู่ที่ 0.438 ซึ่งถือว่าไม่ค่อยมีความสำคัญเท่าไหร 

แล้วเราควรลบแล้ว INDUS ออกไหม 

ถ้าเราลบออก แล้ว เหมือน INDUS เนี่ยส่งผลต่อ model เราด้าน a และมีเพียง feature INDUS อันเดียวที่ส่งผลต่อ model เราด้านนี้ เราก็จะสูญเสียความแม่นยำไป

เราก็จะดูผ่าน metric ที่ช่วยทำให้เราตัดสินใจได้ ที่ชื่อว่า Bayesian Information Criterion (BIC)

##### Bayesian Information Criterion (BIC)

ใข้ในการเปรียบเทียบ model 2 ตัวว่าอันไหนมีค่ามากกว่าน้อยกว่ากัน

ตัวเลขทั้งสองความสำคัญของมันคืออันไหนต่ำกว่าอันนั้นดี

เช่นตอนนี้เราก็ทำ 2 model >> อันที่เอา INDUS กับอันที่ไม่เอา INDUS

ไปดูใน jupyter notebook ได้เลย

ค่าตอนที่ยังไม่ดรอป INDUS
BIC : -139.74997769478875
r-squre : 0.7930234826697582
ค่าตอนที่ดรอป INDUS
BIC : -145.14508855591163
r-squre : 0.7927126289415163
ค่าตอนที่ดรอป INDUS และ AGE
BIC : -149.49934294224678
r-squre : 0.7918657661852815

จะเห็นได้ว่าการเอา feature บางส่วนออกทำให้ดีขึ้นด้วยซ้ำ r-square ก็แทบไม่เปลี่ยนแปลงมาก

ลองเอา LSTAT ออกสิเจ้าจะเจอความพังพินาศ


### plot regression residual

residual is (r)

formulas:> r = y - Y(predict)

>> residual is difference between the target value and the predicted value

#### why do residual matter

regression ของเราอะจะขึ้นอยู่กับเส้น regression line เป็นการทำนาย ที่แน่ๆอะมันไม่มีทางถูกต้อง 100% 

>> residual used to check if assumptions hold and model is valid

>> residual should be random, no pattern

ถ้าเรา plot ค่า แกน y = residual, x = ค่าที่ทำนายได้

รูปแบบของมันต้องไร้รูปแบบ regression ก็เหมือนคนมีหลายๆอย่างที่สามารถทำให้มันป่วยได้ เราต้องเป็นเหมือนหมอ ทำการวิเคราะห์ทำการรักษามัน

เราอาจจะต้องทำการย้อนกลับไปดู model เราว่ามันมีอะไรผิดพลาดไหม หรือเราตกหล่นอะไรไปหรือผ่าวเช่นการเลือก feature หรือเราต้อง transform data ใหม่อีกครั้ง

>> the perfect data and model cause residual to be normally distributed

แต่แบบมันก็ไม่จำเป็นต้อง normal เป้ะๆ อาจจะใกล้เคียงก็ได้ เลยมีคำกล่าวไว้ว่า

" the statisticain knows that in nature there never was a normal distribution, there never was a straight line, yet with normal and linear assumption, known to be false, he can approximation, thoes found in the real world"- George Box

"All models are wrong but some are useful"

go to jupyter notebook to learn more

### make prediction

ถ้ากราฟเราเป็น normal distribution 

mean - sd < mean < mean + sd

จากวิชาสถิติเราจะรู้ว่ามีข้อมูลอยู่ในช่วงนี้ประมาณ 65%

mean - 2sd < mean < mean + 2sd

จากวิชาสถิติเราจะรู้ว่ามีข้อมูลอยู่ในช่วงนี้ประมาณ 95%

mse = (1/n)*sum(y-y(predict))**2

root(mse) = root((1/n)*sum(y-y(predict))**2) "root mean square error"

ที่น่าสนใจก็คือ root mean squared error เนี่ยคือ 1 sd ของ residual distribution

## build a valuation tool

เราจะทำ model ที่เราทำมาอะ ให้เหมือน module เพื่อให้คนอื่นสามารถนำไปใช้ได้

จะได้ใช้การกำหนดค่าต่างๆที่ควรจะเป็น ค่าที่คนทั่วไปไม่รู้ เช่นค่า INDUS NOX แบบเนี่ยคนทั่วไปไม่ได้สนใจมันหรอก

เราก็ต้องทำการเซ็ทค่าที่เหมาะสมให้แก่มัน