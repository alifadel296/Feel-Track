import csv 


def create_csv_structure (num_coords , path_of_csv):
    
    landmarks = ['class']
    
    for val in range(1, num_coords+1):
        landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
        
    f = open(path_of_csv, mode='w', newline='')
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)
    
    f.close()
