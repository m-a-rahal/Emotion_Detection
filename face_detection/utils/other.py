import threading
import cv2

def pad_to_square(image, borderType = cv2.BORDER_CONSTANT): # v2.BORDER_CONSTANT, cv2.BORDER_REFLECT, cv2.BORDER_REFLECT_101, cv2.BORDER_DEFAULT, cv2.BORDER_REPLICATE, cv2.BORDER_WRAP
    w,h,c = image.shape
    size = max(w,h)
    x_pad,y_pad = 0,0
    if w == size:
        y_pad = size - h
    else:
        x_pad = size - w
    image = cv2.copyMakeBorder(image, x_pad//2, x_pad//2, y_pad//2, y_pad//2, borderType)
    return image[:size, :size]

def run_as_thread(func):
    ''' returns the thread to the function ! so you can gather them out in a list or something :3'''
    def wrapper(*args, **kwargs):
        t = threading.Thread(target=func, args=args, kwargs=kwargs)
        t.start()
        return t
    return wrapper

def turn_to_square(x, y, w, h):
    if w != h:
        print('not square, fixing')
        m = max(w,h)
        center = (x + w//2, y+h//2)
        x = center[0] - m//2
        y = center[1] - m//2
        w = h = m
    return (x, y, w, h)