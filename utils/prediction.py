import numpy as np
import tensorflow as tf
from tabulate import tabulate


def seisfacies_predict(model, test_image, argmax=True, batch_size=1000): 
    predictions=[]
    size=batch_size #lower this if you have an memory overload in test
    for image in range(0,test_image.shape[0], size):
        limit=np.min([test_image.shape[0], (image+size)])
        if model.name=="unet3plus":
            prediction=model.predict(test_image[image:limit], batch_size=int(batch_size/5))[-1]
        else:
            prediction=model.predict(test_image[image:limit])
        if(argmax):prediction=np.argmax(prediction, axis=3)
        if image==0:
            predictions=prediction
        else:
            predictions=np.append(predictions, prediction, axis=0)
    return predictions
    
def calculate_micro_f1_score(true_positives, false_positives, false_negatives):
    micro_precision = true_positives / (true_positives + false_positives)
    micro_recall = true_positives / (true_positives + false_negatives)
    micro_f1_score = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)

    return micro_f1_score

def calculate_class_info(test_label, num_classes, predicted_label):
    true_positives = [0] * num_classes
    false_positives = [0] * num_classes
    false_negatives = [0] * num_classes


    for class_idx in range(num_classes):
        true_positive_mask = (predicted_label == class_idx) & (test_label == class_idx)
        false_positive_mask = (predicted_label == class_idx) & (test_label != class_idx)
        false_negative_mask = (predicted_label != class_idx) & (test_label == class_idx)

        true_positives[class_idx] += true_positive_mask.sum().item()
        false_positives[class_idx] += false_positive_mask.sum().item()
        false_negatives[class_idx] += false_negative_mask.sum().item()

    accuracy_by_class = {}
    for class_idx in range(num_classes):
        total_pixels = true_positives[class_idx] + false_positives[class_idx] + false_negatives[class_idx]
        if true_positives[class_idx]!=0:
            accuracy_by_class[class_idx] = true_positives[class_idx] / total_pixels
            precision = true_positives[class_idx] / (true_positives[class_idx] + false_positives[class_idx])
            recall = true_positives[class_idx] / (true_positives[class_idx] + false_negatives[class_idx])
        else:
            accuracy_by_class[class_idx]=0
            precision=0
            recall=0
        accuracy_by_class[class_idx] = [
            accuracy_by_class[class_idx],
            precision,
            recall
        ]

    total_true_positives = sum(true_positives)
    total_false_positives = sum(false_positives)
    total_false_negatives = sum(false_negatives)

    accuracy = (predicted_label == test_label).mean()

    micro_f1_score = calculate_micro_f1_score(total_true_positives, total_false_positives, total_false_negatives)

    return accuracy_by_class, micro_f1_score, accuracy

def calculate_macro_f1_score(class_info, num_classes=6):
    class_f1 = list(range(0,num_classes))
    for class_idx in class_info:
        precision = class_info[class_idx][1]
        recall = class_info[class_idx][2]
        if(precision+recall!=0):
            class_f1[class_idx] = 2 * (precision * recall) / (precision + recall)
        else:
            class_f1[class_idx]=0

    macro_f1_score = np.mean(list(class_f1))
    return macro_f1_score, class_f1

def calculate_iou_per_class(predicted_label, test_label, num_classes):
    iou_by_class = {}
    for class_idx in range(num_classes):
        intersection = np.logical_and(predicted_label == class_idx, test_label == class_idx).sum()
        union = np.logical_or(predicted_label == class_idx, test_label == class_idx).sum()
        if union != 0:
            iou_by_class[class_idx] = intersection / union
        else:
            iou_by_class[class_idx] = 0.0
    return iou_by_class

def calculate_mean_iou(iou_by_class, num_classes):
    total_iou = sum(iou_by_class.values())
    return total_iou / num_classes

def make_prediction(name, folder, model, test_image, test_label, num_classes=6):
    predicted_label = seisfacies_predict(model, test_image)
    
    if len(test_image.shape) > 3:
        class_info, micro_f1, accuracy = calculate_class_info(tf.squeeze(test_label).numpy(), num_classes, predicted_label)
        iou_by_class = calculate_iou_per_class(predicted_label, tf.squeeze(test_label).numpy(), num_classes)
    else:
        class_info, micro_f1, accuracy = calculate_class_info(test_label, num_classes, predicted_label)
        iou_by_class = calculate_iou_per_class(predicted_label, test_label, num_classes)
    
    macro_f1, class_f1 = calculate_macro_f1_score(class_info, num_classes)
    
    mean_iou = calculate_mean_iou(iou_by_class, num_classes)

    data = []
    for i in range(len(class_info)):
        data.append([
            "Classe " + str(i), 
            class_info[i][0],  # Accuracy
            class_info[i][1],  # Precision
            class_info[i][2],  # Recall
            class_f1[i],       # F1 Score
            iou_by_class[i]    # IoU Score
        ])

    col_names = ["Classe", "Accuracy", "Precision", "Recall", "F1 Score", "IoU"]

    with open(f"results/{folder}/tables/table_{name}.txt", "w", encoding="utf-8") as f:
        f.write(tabulate(data, headers=col_names, tablefmt="fancy_grid", floatfmt=".4f"))
        texto = f"\nMacro F1: {macro_f1:.4f}\nAccuracy: {accuracy:.4f}\nMicro F1: {micro_f1:.4f}\nMean IoU (mIoU): {mean_iou:.4f}"
        f.write(texto)