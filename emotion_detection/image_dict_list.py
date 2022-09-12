import cv2
from emotion_detection.utils.drawing import Drawer
from emotion_detection.image_dict import ImageDict


class ImageDictList(list):
    def __init__(self, *args, emotions=None, drawer: Drawer = None):
        super().__init__(*args)
        self.emotions = emotions
        self.drawer = drawer

    def scroll_through_results(self, rotate_scroll=True):
        """
        :param rotate_scroll: rewind list of results when the index is out of bounds
        """
        print('press 4 to show previous image')
        print('press 6 to show next image')
        print('press 8 and 2 to scroll through faces')
        print('press 0 to exit')
        i = -1
        j = -1
        action = ord('r')
        while True:
            if action == ord('2'):
                j -= 1
            if action == ord('8'):
                j += 1
            if action == ord('6'):
                i += 1
            elif action == ord('4'):
                i -= 1
            if rotate_scroll:
                i = i % len(self)
            else:
                i = min(len(self) - 1, i)
                i = max(0, i)
            res: ImageDict = self[i]
            image = res.draw_boxes_and_margin(selected_idx=j % len(self[i].boxes))
            # res.show_square_images()
            cv2.imshow('image', image[:, :, ::-1])
            action = cv2.waitKey(0)
            if action in map(ord, '0q'):
                break
        cv2.destroyAllWindows()
