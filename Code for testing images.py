from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

def predict_and_display(image_path):
    model = load_model('model here')

    class_labels = ['axial abnormal-0', 'axial abnormal-1', 'axial acl-0', 'axial acl-1', 'axial meniscus-0', 'axial meniscus-1', 'coronal abnormal-0', 'coronal abnormal-1', 'coronal acl-0', 'coronal acl-1', 'coronal menescus-0', 'coronal Menescus-1', 'sagittal abnormal-0', 'sagittal abnormal-1', 'sagittal acl-0', 'sagittal acl-1', 'sagittal meniscus-0', 'sagittal meniscus-1']

    img = image.load_img(image_path, target_size=(224, 224))

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)

    predicted_class = class_labels[np.argmax(predictions[0])]

    plt.imshow(img)
    plt.axis('off')

    plt.text(0, -20, 'Predicted MRI scan type: ' + predicted_class, fontsize=12, color='red')

    plt.show()

    print('Predicted MRI scan type:', predicted_class)

predict_and_display('test image with format')
