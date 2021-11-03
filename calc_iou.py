import csv
from IOU import IOU

if __name__ == '__main__':

    gt_csv_path = './save_else/csv/1026_001129_interpolated_1026_003615.csv'
    compare_csv_path = './save_else/csv/1029_004849_2d_2.csv'

    with open(gt_csv_path, 'r', newline='') as f:
        gt_csv = [line for line in csv.reader(f)]
        gt_csv = gt_csv[1:]

    with open(compare_csv_path, 'r', newline='') as f:
        compare_csv = [line for line in csv.reader(f)]
        compare_csv = compare_csv[1:]

    s = ''
    for i in range(71 - 1):
        aver = 0.0
        for j in range(3):
            bbox = [int(compare_csv[i][k]) for k in range(j * 4 + 1, j * 4 + 5)]
            bbox_label = [int(gt_csv[i][k]) for k in range(abs(j) * 4 + 1, abs(j) * 4 + 5)]
            ret = IOU(bbox, bbox_label)
            aver += ret * 100.0
        s += str(aver / 3.0) + '\n'
            # print(str(ret * 100.0) + '%', end=' ')
        # print()
    with open('save_else/calc_iou_2d_2.txt', 'w+') as f:
        f.write(s)

