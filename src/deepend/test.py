y = 250
import time
for i in range(250):
	print(str(i)+'KB / '+str(y)+' KB downloaded!', end='\r')
	time.sleep(1) 