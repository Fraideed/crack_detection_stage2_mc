import csv
import glob
import os

path= r'../mydata/test'
with open('../mydata/test_labels.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=' ',
                            quotechar=' ', quoting=csv.QUOTE_NONE)
    for file in glob.glob(path+'\*\*'):
        # dirname = os.path.join(os.path.dirname(file)[-1], os.path.split(file)[-1])
        dirname =os.path.dirname(file)[-1]+'/'+os.path.split(file)[-1]
        label = os.path.dirname(file)[-1]
        spamwriter.writerow([dirname+','+label])


