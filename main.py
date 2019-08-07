from reverse_frogger import Reverse_Frogger
from build_vocab import Vocabulary
import cv2
def run():
    rf = Reverse_Frogger()
    #rf.test_rf()
    img_path = "data/Frogger_Turk/Currrent_State/Screenshot_0.png"
    img_arr = cv2.imread(img_path)
    resized_img = cv2.resize(img_arr, (320,320))
    text = 'I moved up to get away from the car coming from the left'
                
    explanations = []
    images = []
    images.append(resized_img)
    explanations.append(text)
    output = rf.inference(explanations, images)
    print("output: ", output)
    return output
    
run()