#include <Servo.h>
Servo index; 
Servo thumb_abd;
Servo thumb_flex;

int pos = 0; 

void setup() {
  thumb_flex.write(0);
  thumb_flex.attach(4); 
  thumb_abd.write(0);
  thumb_abd.attach(5); 
  index.write(0);
  index.attach(6); 
  delay(100);

  LP();
  delay(5000);
  LP_open();

  // TVG();
  // delay(5000);
  // TVG_open();
}

void loop() {
}

void TVG() {
  // Abduct thumb
  for (pos = 0; pos <= 120; pos += 1) {
     thumb_abd.write(pos); 
     delay(15); 
  }

  delay(2000); 

  // Flex index/middle
  for (pos = 0; pos <= 100; pos += 1) { 
     index.write(pos);
     delay(15); 
   }
}

void LP(){
  // Flex index/middle
  for (pos = 0; pos <= 100; pos += 1) { 
     index.write(pos);
     delay(15); 
   }

  delay(2000);   

  // Abduct and flex thumb
  for (pos = 0; pos <= 100; pos += 1) {
    thumb_flex.write(pos);
    thumb_abd.write(pos); 
    delay(15); 
  }
}

void TVG_open() {
  // Return index/middle to 0
  for (pos = 100; pos >= 0; pos -= 1) { 
    index.write(pos);
    delay(15);                       
  }

  // Return thumb to 0
  for (pos = 120; pos >= 0; pos -= 1) { 
    thumb_abd.write(pos); 
    delay(15);                       
  }
}

void LP_open() {
  // Return all to 0
  for (pos = 100; pos >= 0; pos -= 1) { 
    thumb_abd.write(pos); 
    thumb_flex.write(pos);
    index.write(pos);
    delay(15);                       
  }
}
