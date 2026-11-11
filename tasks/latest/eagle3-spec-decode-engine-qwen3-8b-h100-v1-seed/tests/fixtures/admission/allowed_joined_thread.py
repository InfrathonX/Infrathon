import threading
t=threading.Thread(target=lambda:None)
t.start()
t.join()
