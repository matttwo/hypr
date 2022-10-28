import csv
import requests
import os.path

metadata = []
with open('inaturalist_info.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for count, row in enumerate(spamreader):
        if count == 0:
            continue
        metadata.append((row[1], row[-2]))
print("Finished reading metadata")

with open('../dataset/labels/inat_labels.txt', 'w') as f:
    for count, img_data in enumerate(metadata):
        if count % 200 == 0:
            print(count)
            
        if os.path.exists('../dataset/images/image_{count}.jpg'):
            f.write(f"images/image_{count}.jpg {img_data[1]}\n")
            continue
        try:
            img = requests.get(img_data[0]).content
            with open(f"../dataset/images/image_{count}.jpg", 'wb') as handler:
                handler.write(img)
                f.write(f"images/image_{count}.jpg {img_data[1]}\n")
        except:
            print(count)
