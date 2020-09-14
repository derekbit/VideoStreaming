import cv2
import numpy as np

def prepare_net(config_file, weights_file, labels_file):
    global net, output_layers, labels, colors

    labels = open(labels_file).read().strip().split('\n')

    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    net = cv2.dnn.readNetFromDarknet(config_file, weights_file)
    # Ref:
    # https://stackoverflow.com/questions/57706412/what-is-the-working-and-output-of-getlayernames-and-getunconnecteddoutlayers
    #
    # net.getLayerNames(): It gives you list of all layers used in a network.
    # Like I am currently working with yolov3. It gives me a list of 254 layers.
    layer_names = net.getLayerNames()

    # net.getUnconnectedOutLayers(): It gives you the final layers number in
    # the list from net.getLayerNames().
    # It gives the layers number that are unused (final layer). 
    #
    # For yolov3-tiny, it gave me three number, 200, 227, 254.
    # To get the corresponding indexes, we need to do layer_names[i[0] - 1].
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return net, output_layers, labels, colors

def draw_labels_and_boxes(frame, boxes, confidences, classids, idxs, colors, labels):
    if len(idxs) == 0:
        return frame

    # If there are any detections
    for i in idxs.flatten():
        # Get the bounding box coordinates
        x, y = boxes[i][0], boxes[i][1]
        w, h = boxes[i][2], boxes[i][3]
            
        # Get the unique color for this class
        color = [int(c) for c in colors[classids[i]]]

        # Draw the bounding box rectangle and label on the image
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        text = "{}: {:4f}".format(labels[classids[i]], confidences[i])
        cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

def generate_boxes_confidences_classids(outs, height, width, confidence):
    boxes = []
    confidences = []
    classids = []

    for out in outs:
        for detection in out:
            #print (detection)
            #a = input('GO!')
            
            # Get the scores, classid, and the confidence of the prediction
            scores = detection[5:]
            classid = np.argmax(scores)
            score = scores[classid]
            
            # Consider only the predictions that are above a certain confidence level
            if score > confidence:
                # TODO Check detection
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, bwidth, bheight = box.astype('int')

                # Using the center x, y coordinates to derive the top
                # and the left corner of the bounding box
                x = int(centerX - (bwidth / 2))
                y = int(centerY - (bheight / 2))

                # Append to list
                boxes.append([x, y, int(bwidth), int(bheight)])
                confidences.append(float(score))
                classids.append(classid)

    return boxes, confidences, classids


def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def process_frame(frame):
    confidence = 0.5
    threshold = 0.3

    height = frame.shape[0]
    width = frame.shape[1]

    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255, size=(416, 416), swapRB=True, crop=False)

    net.setInput(blob)
    preds = net.forward(getOutputsNames(net))

    boxes, confidences, classids = generate_boxes_confidences_classids(preds, height, width, confidence)

    # Apply Non-Maxima Suppression to suppress overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

    # Draw labels and boxes on the image
    frame = draw_labels_and_boxes(frame, boxes, confidences, classids, idxs, colors, labels)

    return frame
