def fibonacci():
  a = 0
  b = 1
  while True:             
    yield a             
    future = a + b
    a = b                 
    b = future           
